#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import csv
import cv2
import numpy as np

# --- GPU libs ---
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cnd
    from cucim.skimage import exposure as cexp
    from cucim.skimage.measure import label as gpu_label, regionprops as gpu_regionprops
except Exception as e:
    raise ImportError(
        "GPU CC requires CuPy + cuCIM. Install e.g.: pip install 'cupy-cuda12x' 'cucim'"
    ) from e


# -------------------------
# Small helpers (CPU side)
# -------------------------
def _to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def _cpu_clahe(gray_stack: list[np.ndarray], clip: float, tile: int) -> list[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return [clahe.apply(g) for g in gray_stack]


# -------------------------
# GPU ops (batched)
# -------------------------
def _gpu_pad_batched(g: cp.ndarray, pad: int) -> cp.ndarray:
    # g: (N,H,W) uint8
    return cp.pad(g, ((0,0),(pad,pad),(pad,pad)), mode="edge")

def _gpu_tophat(g: cp.ndarray, use: bool, ksize: int) -> cp.ndarray:
    if not use or ksize <= 1:
        return g
    k = int(max(1, ksize))
    se = cp.ones((1, k, k), dtype=cp.bool_)
    opened = cnd.grey_opening(g, footprint=se, mode="reflect")
    out = g.astype(cp.int16) - opened.astype(cp.int16)
    return cp.clip(out, 0, 255).astype(cp.uint8)

def _gpu_dog(g: cp.ndarray, use: bool, s1: float, s2: float) -> cp.ndarray:
    if not use:
        return g
    # Gaussian blur in float, then difference, then per-frame rescale to 0..255
    gf1 = cnd.gaussian_filter(g.astype(cp.float32), sigma=(0, float(s1), float(s1)), mode="reflect")
    gf2 = cnd.gaussian_filter(g.astype(cp.float32), sigma=(0, float(s2), float(s2)), mode="reflect")
    diff = gf1 - gf2
    N = diff.shape[0]
    out = cp.empty_like(g)
    for i in range(N):
        vmin = diff[i].min()
        vmax = diff[i].max()
        denom = cp.maximum(vmax - vmin, cp.float32(1e-6))
        out[i] = cp.clip((diff[i] - vmin) * (255.0 / denom), 0, 255).astype(cp.uint8)
    return out

def _gpu_binary_open(mask: cp.ndarray, k: int) -> cp.ndarray:
    if k is None or k <= 1:
        return mask
    kk = int(k)
    if kk % 2 == 0:
        kk += 1
    se = cp.ones((kk, kk), dtype=cp.bool_)
    return cnd.binary_opening(mask, structure=se, iterations=1)

def _gpu_adaptive_threshold(g: cp.ndarray, method: str, c_offset: float = 2.0) -> cp.ndarray:
    """
    Adaptive mean/gaussian with OpenCV-like defaults:
      blockSize = 11, C = 2
    Returns boolean mask (N,H,W).
    """
    method = (method or "otsu").lower()
    block = 11
    C = float(c_offset)

    if method == "adaptive_mean":
        mean = cnd.uniform_filter(g.astype(cp.float32), size=(1, block, block), mode="reflect")
        return (g.astype(cp.float32) > (mean - C))
    if method == "adaptive_gaussian":
        sigma = block / 6.0
        mean = cnd.gaussian_filter(g.astype(cp.float32), sigma=(0, sigma, sigma), mode="reflect")
        return (g.astype(cp.float32) > (mean - C))

    # Otsu (per frame) fully on GPU
    if method == "otsu":
        N, H, W = g.shape
        mask = cp.zeros((N, H, W), dtype=cp.bool_)
        for i in range(N):
            img = g[i]
            hist = cp.histogram(img, bins=256, range=(0, 256))[0].astype(cp.float64)
            total = img.size
            idx = cp.arange(256, dtype=cp.float64)

            wb = cp.cumsum(hist)
            wf = total - wb
            # avoid zero division
            wbz = cp.maximum(wb, 1e-12)
            wfz = cp.maximum(wf, 1e-12)

            mu_b = cp.cumsum(hist * idx) / wbz
            mu_all = cp.cumsum(hist * idx)[-1] / total
            mu_f = (mu_all * total - cp.cumsum(hist * idx)) / wfz

            var_between = wb * wf * (mu_b - mu_f) ** 2
            thr = int(cp.argmax(var_between).item())
            mask[i] = img > thr
        return mask

    # Fallback (shouldn't hit here)
    return g > 0


# -------------------------
# Main entry
# -------------------------
def detect_stage1_to_csv(
    *,
    orig_path: Path,
    csv_path: Path,
    max_frames: int | None,
    # CC gates:
    min_area_px: int,
    max_area_scale: float,
    # Preproc options:
    use_clahe: bool,
    clahe_clip: float,
    clahe_tile: int,
    use_tophat: bool,
    tophat_ksize: int,
    use_dog: bool,
    dog_sigma1: float,
    dog_sigma2: float,
    # Segmentation + labeling:
    threshold_method: str,
    fixed_threshold: int,
    adaptive_c: float = 2.0,
    open_ksize: int,
    connectivity: int,
    # NEW (optional) batching/CLAHE backend:
    batch_size: int = 32,                 # number of frames per GPU batch
    preproc_backend: str = "cupy",        # 'cupy' (batched GPU CLAHE) or 'opencv_cuda' (per-frame cv2.cuda CLAHE)
    verbose: bool = False,
):
    """
    Writes CSV with: frame,x,y,w,h
    - Adds 2px replicate border, then unpads and clips boxes.
    - Everything heavy runs on GPU. cv2.cuda CLAHE is optional.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(orig_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {orig_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_area_px = int(W * H * float(max_area_scale)) if max_area_scale > 0 else (W * H)

    # cuCIM connectivity: 1=4-neigh, 2=8-neigh
    conn = 1 if int(connectivity) == 4 else 2

    # (Optional) pinned mem allocator can speed H->D a bit
    try:
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
    except Exception:
        pass

    use_cv2_cuda = (preproc_backend.lower() == "opencv_cuda") and hasattr(cv2, "cuda")
    if preproc_backend.lower() == "opencv_cuda" and not use_cv2_cuda:
        print("[cc_cuda] preproc_backend='opencv_cuda' requested but cv2.cuda is not available; falling back to 'cupy'.")
        use_cv2_cuda = False

    pad = 2
    rows: list[tuple[int,int,int,int,int]] = []
    t = 0

    try:
        while True:
            # ---- load up to batch_size frames (CPU) ----
            gray_batch: list[np.ndarray] = []
            frame_idx: list[int] = []
            for _ in range(max(1, int(batch_size))):
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and t >= max_frames:
                    break
                g = _to_gray(frame)
                gray_batch.append(g)
                frame_idx.append(t)
                t += 1

            if not gray_batch:
                break

            # ---- CLAHE ----
            if use_clahe:
                if use_cv2_cuda:
                    # per-frame GPU CLAHE via cv2.cuda; then proceed batched on CuPy
                    proc = []
                    clahe = cv2.cuda.createCLAHE(clipLimit=float(clahe_clip),
                                                 tileGridSize=(int(clahe_tile), int(clahe_tile)))
                    for g in gray_batch:
                        gm = cv2.cuda_GpuMat()
                        gm.upload(g)
                        g_eq = clahe.apply(gm)
                        proc.append(g_eq.download())
                    gray_batch = proc
                else:
                    # CPU CLAHE (still fine; we upload the batch once)
                    gray_batch = _cpu_clahe(gray_batch, clahe_clip, clahe_tile)

            # One batched upload to GPU: (N,H,W) uint8
            g = cp.asarray(np.stack(gray_batch, axis=0), dtype=cp.uint8)

            # ---- GPU preproc ----
            g = _gpu_pad_batched(g, pad=pad)  # (N,H+2p,W+2p)
            g = _gpu_tophat(g, use_tophat, tophat_ksize)
            g = _gpu_dog(g, use_dog, dog_sigma1, dog_sigma2)

            # ---- Threshold (GPU) ----
            thr_m = (threshold_method or "otsu").lower()
            if thr_m == "fixed":
                mask = g.astype(cp.int16) > int(fixed_threshold)
            else:
                if thr_m not in ("adaptive_gaussian", "adaptive_mean", "otsu"):
                    thr_m = "otsu"
                mask = _gpu_adaptive_threshold(g, thr_m, c_offset=adaptive_c)

            # ---- Binary open (GPU) ----
            mask = _gpu_binary_open(mask, open_ksize)

            # ---- Label each slice + emit bboxes ----
            N = mask.shape[0]
            for i in range(N):
                labeled = gpu_label(mask[i], connectivity=conn)
                for r in gpu_regionprops(labeled):
                    (minr, minc, maxr, maxc) = r.bbox
                    w = int(maxc - minc)
                    h = int(maxr - minr)
                    if w <= 0 or h <= 0:
                        continue
                    area = int(r.area)
                    if area < int(min_area_px) or area > max_area_px:
                        continue

                    # unpad + clip to original frame
                    x = int(minc) - pad
                    y = int(minr) - pad
                    if x < 0: x = 0
                    if y < 0: y = 0
                    if x + w > W: w = max(0, W - x)
                    if y + h > H: h = max(0, H - y)
                    if w <= 0 or h <= 0:
                        continue

                    rows.append((int(frame_idx[i]), x, y, w, h))

    finally:
        cap.release()

    # Stable ordering (nice for diffs)
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))

    with open(csv_path, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['frame', 'x', 'y', 'w', 'h'])
        wtr.writerows(rows)
