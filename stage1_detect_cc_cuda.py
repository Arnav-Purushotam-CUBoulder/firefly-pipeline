#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np
try:
    import cupy as cp
    from cucim.skimage.measure import label as gpu_label, regionprops as gpu_regionprops
except Exception as e:
    raise ImportError("GPU CC variant requires CuPy + cuCIM (match CUDA): "
                      "pip install 'cupy-cuda12x' 'cucim'") from e

def _clahe(gray, use, clip, tile):
    if not use: return gray
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return clahe.apply(gray)

def _tophat(gray, use, ksize):
    if not use: return gray
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ksize), int(ksize)))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

def _dog(gray, use, s1, s2):
    if not use: return gray
    g1 = cv2.GaussianBlur(gray, (0,0), max(1e-6, float(s1)))
    g2 = cv2.GaussianBlur(gray, (0,0), max(1e-6, float(s2)))
    out = cv2.subtract(g1, g2)
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def _binary_mask(gray: np.ndarray, method: str, fixed_thr: int, open_ksize: int) -> np.ndarray:
    if method == 'fixed':
        _, mask = cv2.threshold(gray, int(fixed_thr), 255, cv2.THRESH_BINARY)
    elif method == 'adaptive_mean':
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    elif method == 'adaptive_gaussian':
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    else:  # 'otsu'
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = max(1, int(open_ksize));  k += (k % 2 == 0)
    if k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask

def detect_stage1_to_csv(
    *,
    orig_path: Path, csv_path: Path, max_frames: int | None,
    min_area_px: int, max_area_scale: float,
    use_clahe: bool, clahe_clip: float, clahe_tile: int,
    use_tophat: bool, tophat_ksize: int,
    use_dog: bool, dog_sigma1: float, dog_sigma2: float,
    threshold_method: str, fixed_threshold: int, open_ksize: int, connectivity: int,
):
    csv_path = Path(csv_path); csv_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(orig_path))
    if not cap.isOpened(): raise RuntimeError(f"Could not open video: {orig_path}")

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_area_px = int(W*H * float(max_area_scale)) if max_area_scale > 0 else (W*H)
    conn = 1 if int(connectivity)==4 else 2  # cuCIM: 1=4n, 2=8n in 2D

    rows, t = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and t >= max_frames): break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = _clahe(gray, use_clahe, clahe_clip, clahe_tile)
        gray = _tophat(gray, use_tophat, tophat_ksize)
        gray = _dog(gray, use_dog, dog_sigma1, dog_sigma2)

        mask_cpu = _binary_mask(gray, threshold_method, fixed_threshold, open_ksize)
        mask_gpu = cp.asarray(mask_cpu > 0)

        labeled = gpu_label(mask_gpu, connectivity=conn)
        for r in gpu_regionprops(labeled):
            (minr, minc, maxr, maxc) = r.bbox
            w, h = int(maxc-minc), int(maxr-minr)
            area   = int(r.area)
            if area < int(min_area_px) or area > max_area_px or w<=0 or h<=0: continue
            rows.append((t, int(minc), int(minr), w, h))
        t += 1
    cap.release()

    import csv
    with open(csv_path, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['frame','x','y','w','h'])
        wtr.writerows(rows)
