#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
# Tiny progress bar (matches style used elsewhere)
_BAR = 50
def _progress(i, total, tag=''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar  = '=' * int(frac * _BAR) + ' ' * (_BAR - int(frac * _BAR))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i >= total: sys.stdout.write('\n')

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Image helpers (center-semantics aware; Stage 8 writes xy_semantics='center')
def _center_crop_clamped(img, cx: float, cy: float, w: int, h: int):
    H, W = img.shape[:2]
    w = max(1, int(round(w))); h = max(1, int(round(h)))
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w)); y0 = max(0, min(y0, H - h))
    return img[y0:y0+h, x0:x0+w], x0, y0

def _crop_from_row(frame_bgr, row: dict) -> Tuple[np.ndarray, int, int, int, int, int]:
    """
    Returns (crop_bgr, cx_int, cy_int, frame_idx, w, h).
    Robust to top-left vs center semantics; Stage 8 uses center semantics.
    """
    w = int(round(float(row.get('w', 10))))
    h = int(round(float(row.get('h', 10))))
    if str(row.get('xy_semantics','')).lower() == 'center':
        cx = float(row['x']); cy = float(row['y'])
    else:
        cx = float(row['x']) + w/2.0
        cy = float(row['y']) + h/2.0
    crop, x0, y0 = _center_crop_clamped(frame_bgr, cx, cy, w, h)
    return crop, int(round(cx)), int(round(cy)), int(row['frame']), w, h

def _largest_cc_area_pixels_over(gray: np.ndarray, strict_brightness_floor: int) -> int:
    """
    Area = number of pixels in the largest 8-connected component of pixels with intensity > strict_brightness_floor.
    """
    if gray is None or gray.size == 0:
        return 0
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    # THRESH_BINARY with thresh=T yields dst(x,y)=255 if src(x,y) > T, else 0.
    _, bin_img = cv2.threshold(gray, int(strict_brightness_floor), 255, cv2.THRESH_BINARY)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if num <= 1 or stats.shape[0] <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())

# ──────────────────────────────────────────────────────────────
# CSV helpers
def _read_csv_rows(csv_path: Path) -> List[dict]:
    with csv_path.open('r', newline='') as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]

def _write_csv_rows(csv_path: Path, rows: List[dict], preserve_order_from: Optional[List[str]] = None):
    if preserve_order_from is None:
        preserve_order_from = list(rows[0].keys()) if rows else []
    with csv_path.open('w', newline='') as f:
        wri = csv.DictWriter(f, fieldnames=preserve_order_from)
        wri.writeheader()
        for r in rows:
            wri.writerow({k: r.get(k, '') for k in preserve_order_from})

# ──────────────────────────────────────────────────────────────
# Main entry
def stage8_5_prune_by_blob_area(
    *,
    orig_video_path: Path,
    csv_path: Path,
    area_threshold_px: int,
    min_pixel_brightness_to_be_considered_in_area_calculation: int,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """
    After Stage 8:
      - For each *firefly* row in the main CSV, crop (x,y,w,h) and compute:
          area := pixels in largest 8-connected CC with intensity > min_pixel_brightness_to_be_considered_in_area_calculation
      - Remove rows with area < area_threshold_px.
      - Mirror deletions into the Stage-8 fireflies-only logits CSV (<stem>_fireflies_logits.csv).

    Returns: (deleted_from_main_fireflies, kept_main_fireflies, deleted_from_fireflies_logits)
    """
    rows = _read_csv_rows(csv_path)
    if not rows:
        if verbose:
            print("[stage8.5] No rows; nothing to do.")
        return (0, 0, 0)

    # Preserve original column order
    orig_cols = list(rows[0].keys())

    # Count fireflies prior to pruning
    has_class = 'class' in rows[0]
    firefly_idxs_all = [i for i, r in enumerate(rows) if (not has_class or r.get('class') == 'firefly')]

    # Group by frame for efficient video IO (only frames that have any rows)
    by_frame: Dict[int, List[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        try:
            f = int(r['frame'])
            if max_frames is not None and f >= max_frames:
                continue
            by_frame[f].append(idx)
        except Exception:
            continue

    # Walk the video, evaluate areas for firefly rows, mark for removal if area<thr
    keep_mask = [True] * len(rows)
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage8.5] Could not open video: {orig_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = min(max_frames, total_frames) if max_frames is not None else total_frames

    fr = 0
    while True:
        if limit is not None and fr >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break

        idxs = by_frame.get(fr, [])
        if idxs:
            for idx in idxs:
                r = rows[idx]
                # Only process class=='firefly' (leave background rows untouched)
                if has_class and r.get('class') != 'firefly':
                    continue
                try:
                    crop_bgr, cx, cy, fidx, w, h = _crop_from_row(frame, r)
                    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.size else None
                    area = _largest_cc_area_pixels_over(gray, min_pixel_brightness_to_be_considered_in_area_calculation)
                    if area < int(area_threshold_px):
                        keep_mask[idx] = False
                except Exception:
                    # If anything fails, keep the row (non-destructive)
                    continue

        _progress(fr+1, limit, 'stage8.5-area'); fr += 1

    cap.release()

    # Prepare filtered main rows
    before_fireflies = len(firefly_idxs_all)
    deleted_main_fireflies = sum(1 for i in firefly_idxs_all if not keep_mask[i])
    kept_main_fireflies = before_fireflies - deleted_main_fireflies

    kept_rows = [r for k, r in zip(keep_mask, rows) if k]
    _write_csv_rows(csv_path, kept_rows, preserve_order_from=orig_cols)

    if verbose:
        total_before = len(rows)
        total_after = len(kept_rows)
        print(f"[stage8.5] Main CSV: fireflies before {before_fireflies}  after {kept_main_fireflies}  "
              f"deleted {deleted_main_fireflies}  | total rows {total_before} → {total_after}")

    # Build key set for removed fireflies: (t, int(round(x)), int(round(y)))
    removed_keys = set()
    for keep, r in zip(keep_mask, rows):
        if not keep and (not has_class or r.get('class') == 'firefly'):
            try:
                t = int(r['frame'])
                x = int(round(float(r['x'])))
                y = int(round(float(r['y'])))
                removed_keys.add((t, x, y))
            except Exception:
                continue

    # Mirror deletions into <stem>_fireflies_logits.csv
    deleted_from_ff_csv = 0
    ff_csv = csv_path.with_name(csv_path.stem + '_fireflies_logits.csv')
    if ff_csv.exists():
        with ff_csv.open('r', newline='') as f:
            rdr = csv.DictReader(f)
            ff_rows = [dict(r) for r in rdr]
            ff_cols = rdr.fieldnames or ['x','y','t','background_logit','firefly_logit']

        ff_before = len(ff_rows)
        ff_kept: List[dict] = []
        for r in ff_rows:
            try:
                t = int(r['t']); x = int(r['x']); y = int(r['y'])
                if (t, x, y) in removed_keys:
                    continue
            except Exception:
                # If malformed, keep it (safer)
                pass
            ff_kept.append(r)

        with ff_csv.open('w', newline='') as f:
            wri = csv.DictWriter(f, fieldnames=ff_cols)
            wri.writeheader()
            for r in ff_kept:
                wri.writerow({k: r.get(k, '') for k in ff_cols})

        deleted_from_ff_csv = ff_before - len(ff_kept)
        if verbose:
            print(f"[stage8.5] Fireflies logits CSV: before {ff_before}  after {len(ff_kept)}  "
                  f"deleted {deleted_from_ff_csv}")
    else:
        if verbose:
            print(f"[stage8.5] Fireflies logits CSV not found at {ff_csv}; skipped mirroring.")

    return (deleted_main_fireflies, kept_main_fireflies, deleted_from_ff_csv)
