
#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

def _clahe(gray, use, clip, tile):
    if not use:
        return gray
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return clahe.apply(gray)

def _tophat(gray, use, ksize):
    if not use:
        return gray
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ksize), int(ksize)))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

def _dog(gray, use, s1, s2):
    if not use:
        return gray
    g1 = cv2.GaussianBlur(gray, (0, 0), max(1e-6, float(s1)))
    g2 = cv2.GaussianBlur(gray, (0, 0), max(1e-6, float(s2)))
    out = cv2.subtract(g1, g2)
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def _binary_mask(
    gray: np.ndarray,
    method: str,
    fixed_thr: int,
    open_ksize: int,
) -> np.ndarray:
    # Threshold (adaptive methods use OpenCV defaults: blockSize=11, C=2)
    if method == 'fixed':
        _, mask = cv2.threshold(gray, int(fixed_thr), 255, cv2.THRESH_BINARY)
    elif method == 'adaptive_mean':
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == 'adaptive_gaussian':
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:  # 'otsu'
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Optional denoise via opening; k==1 => no-op
    k = max(1, int(open_ksize))
    if k > 1:
        if k % 2 == 0:
            k += 1
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
    return mask

def detect_stage1_to_csv(
    *,
    orig_path: Path,
    csv_path: Path,
    max_frames: int | None,
    # CC gates:
    min_area_px: int,
    max_area_scale: float,
    # Preproc:
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
    open_ksize: int,
    connectivity: int,
):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(orig_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {orig_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = W * H
    max_area_px = int(frame_area * float(max_area_scale)) if max_area_scale > 0 else frame_area

    pad = 2  # border pad in pixels

    rows = []
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and t >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # --- BORDER PAD BEFORE ANY PREPROCESS ---
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        # Preprocess
        gray = _clahe(gray, use_clahe, clahe_clip, clahe_tile)
        gray = _tophat(gray, use_tophat, tophat_ksize)
        gray = _dog(gray, use_dog, dog_sigma1, dog_sigma2)

        # Threshold to binary mask
        mask = _binary_mask(
            gray,
            threshold_method,
            fixed_threshold,
            open_ksize,
        )

        # CC on the padded mask
        num, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=4 if connectivity == 4 else 8
        )

        # stats columns: [x, y, w, h, area] -- coords are in PADDED space
        for lbl in range(1, num):
            x_p, y_p, w, h, area = map(int, stats[lbl])

            # Translate back to original coords
            x = x_p - pad
            y = y_p - pad

            # Clip bbox to original frame bounds
            if x < 0:
                w += x
                x = 0
            if y < 0:
                h += y
                y = 0
            if x + w > W:
                w = W - x
            if y + h > H:
                h = H - y
            if w <= 0 or h <= 0:
                continue

            # Area gate (area measured on padded mask is fine here)
            if area < int(min_area_px) or area > max_area_px:
                continue

            rows.append((t, x, y, w, h))
        t += 1

    cap.release()

    # Stable ordering for diffs / reproducibility
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))

    import csv
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["frame", "x", "y", "w", "h"])
        wtr.writerows(rows)
