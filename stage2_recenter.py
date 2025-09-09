#!/usr/bin/env python3
# stage2_recenter.py (optimized but results-identical)
import csv, sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from audit_trail import AuditTrail

BAR_LEN = 50
def progress(i, total, tag=''):
    frac = i / total if total else 0
    bar  = '=' * int(frac * BAR_LEN) + ' ' * (BAR_LEN - int(frac * BAR_LEN))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i == total: sys.stdout.write('\n')

# ---- centroid helpers (cached indices to avoid per-patch allocations) ----
@lru_cache(maxsize=512)
def _cached_indices(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    # float32 matches original math and preserves results
    ys, xs = np.indices((h, w), dtype=np.float32)
    return ys, xs

def intensity_weighted_centroid_cached(gray_patch_f32: np.ndarray) -> Tuple[float, float]:
    h, w = gray_patch_f32.shape
    total = float(gray_patch_f32.sum())
    if total <= 1e-9:
        # identical fallback as original: center of patch (0-indexed)
        return (w - 1) / 2.0, (h - 1) / 2.0
    ys, xs = _cached_indices(h, w)
    cx = float((xs * gray_patch_f32).sum() / total)
    cy = float((ys * gray_patch_f32).sum() / total)
    return cx, cy

def recenter_boxes_with_centroid(
    orig_path: Path,
    csv_path: Path,
    max_frames=None,
    *,
    bright_max_threshold: int = 50,   # threshold still provided by orchestrator
    audit: AuditTrail | None = None,
    audit_video_path: Path | None = None,
):
    """
    For each CSV row (frame,x,y,w,h):
      • Drop the row if the brightest pixel in the crop < bright_max_threshold
      • Otherwise recenter using intensity-weighted centroid (on grayscale)
      • Overwrite the same CSV with refined x,y and only the rows that passed
    (Behavior is identical to the original implementation; this version reduces
     per-iteration allocations and duplicate color→gray conversions.)
    """
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        print("No detections to recenter."); return

    # Group rows by frame (unchanged)
    by_frame = defaultdict(list)
    max_frame_in_csv = 0
    for idx, r in enumerate(rows):
        f = int(r['frame'])
        by_frame[f].append((idx, int(r['x']), int(r['y']), int(r['w']), int(r['h'])))
        max_frame_in_csv = max(max_frame_in_csv, f)

    cap = cv2.VideoCapture(str(orig_path))
    if not cap.isOpened():
        sys.exit(f"Could not open original video: {orig_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    effective_limit = min(max_frame_in_csv, total)
    if max_frames is not None:
        effective_limit = min(effective_limit, max_frames)
    total = effective_limit

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    keep_mask = [True] * len(rows)
    logs_removed = []
    logs_kept = []

    fr = 0
    while True:
        if max_frames is not None and fr > max_frames:
            break

        ok, frame = cap.read()
        if not ok: break

        if fr in by_frame:
            # 1) one grayscale conversion per frame
            gray_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 2) float32 once per frame for centroid math (matches original)
            gray_f32 = gray_u8.astype(np.float32)

            for row_idx, x, y, w, h in by_frame[fr]:
                r = rows[row_idx]
                # clamp to frame
                w = max(1, min(w, W)); h = max(1, min(h, H))
                x = max(0, min(x, W - w)); y = max(0, min(y, H - h))

                # --- brightest-pixel guard (read from precomputed gray_u8) ---
                # this is IDENTICAL to max(cv2.cvtColor(color_patch, ...)) but avoids
                # another conversion, since gray_u8 was created from the same frame.
                patch_max = int(gray_u8[y:y+h, x:x+w].max()) if w > 0 and h > 0 else 0
                if patch_max < int(bright_max_threshold):
                    keep_mask[row_idx] = False
                    if audit is not None:
                        logs_removed.append({
                            'video': str(orig_path),
                            'frame': int(r['frame']),
                            'x': int(x), 'y': int(y),
                            'w': int(w), 'h': int(h),
                            'bright_max': int(patch_max),
                            'bright_min_thr': int(bright_max_threshold),
                        })
                    continue

                # --- recenter using intensity-weighted centroid on gray_f32 ---
                patch_gray_f32 = gray_f32[y:y+h, x:x+w]
                if patch_gray_f32.size == 0:
                    keep_mask[row_idx] = False
                    continue

                cx_rel, cy_rel = intensity_weighted_centroid_cached(patch_gray_f32)
                cx_full = x + cx_rel
                cy_full = y + cy_rel

                new_x = int(round(cx_full - w/2))
                new_y = int(round(cy_full - h/2))
                new_x = max(0, min(new_x, W - w))
                new_y = max(0, min(new_y, H - h))

                if audit is not None:
                    try:
                        shift_dx = int(new_x - int(r['x']))
                        shift_dy = int(new_y - int(r['y']))
                    except Exception:
                        shift_dx = int(new_x - x)
                        shift_dy = int(new_y - y)
                    logs_kept.append({
                        'video': str(orig_path),
                        'frame': int(r['frame']),
                        'x': int(new_x), 'y': int(new_y),
                        'w': int(w), 'h': int(h),
                        'shift_dx': shift_dx, 'shift_dy': shift_dy,
                        'bright_max': int(patch_max),
                    })

                rows[row_idx]['x'] = str(new_x)
                rows[row_idx]['y'] = str(new_y)

        progress(fr+1, total, 'recenter'); fr += 1

    cap.release()

    # Write back only rows that passed the brightness gate (unchanged)
    with csv_path.open('w', newline='') as f:
        fieldnames = ['frame','x','y','w','h']
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        for keep, r in zip(keep_mask, rows):
            if not keep:
                continue
            wri.writerow({
                'frame': int(r['frame']),
                'x': int(r['x']), 'y': int(r['y']),
                'w': int(r['w']), 'h': int(r['h'])
            })

    # ---- audit sidecars (unchanged) ----
    if audit is not None:
        if logs_removed:
            audit.log_removed('02_recenter', 'dim_seed', logs_removed, extra_cols=['bright_max','bright_min_thr'])
            if audit_video_path is not None:
                for rr in logs_removed[:2000]:
                    audit.save_crop(
                        audit_video_path,
                        rr['frame'], rr['x'], rr['y'], rr['w'], rr['h'],
                        '02_recenter/removed_dim',
                        f"t{rr['frame']:06d}_x{rr['x']}_y{rr['y']}_b{rr['bright_max']}"
                    )
        if logs_kept:
            audit.log_kept('02_recenter', logs_kept, extra_cols=['shift_dx','shift_dy','bright_max'])
