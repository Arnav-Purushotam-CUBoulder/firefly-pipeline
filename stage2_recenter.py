import csv, sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

BAR_LEN = 50
def progress(i, total, tag=''):
    frac = i / total if total else 0
    bar  = '=' * int(frac * BAR_LEN) + ' ' * (BAR_LEN - int(frac * BAR_LEN))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i == total: sys.stdout.write('\n')

def intensity_weighted_centroid(gray_patch: np.ndarray) -> tuple[float, float]:
    ys, xs = np.indices(gray_patch.shape, dtype=np.float32)
    total = float(gray_patch.sum())
    if total <= 1e-9:
        return (gray_patch.shape[1] - 1) / 2.0, (gray_patch.shape[0] - 1) / 2.0
    cx = float((xs * gray_patch).sum() / total)
    cy = float((ys * gray_patch).sum() / total)
    return cx, cy

def recenter_boxes_with_centroid(
    orig_path: Path,
    csv_path: Path,
    max_frames=None,
    *,
    bright_max_threshold: int = 50,   # ← threshold now comes from orchestrator
):
    """
    For each CSV row (frame,x,y,w,h):
      • Drop the row if the brightest pixel in the color crop < bright_max_threshold
      • Otherwise recenter using intensity-weighted centroid (on grayscale)
      • Overwrite the same CSV with refined x,y and only the rows that passed
    """
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        print("No detections to recenter."); return

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

    fr = 0
    while True:
        if max_frames is not None and fr > max_frames:
            break

        ok, frame = cap.read()
        if not ok: break

        if fr in by_frame:
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            for row_idx, x, y, w, h in by_frame[fr]:
                # clamp to frame
                w = max(1, min(w, W)); h = max(1, min(h, H))
                x = max(0, min(x, W - w)); y = max(0, min(y, H - h))

                color_patch = frame[y:y+h, x:x+w]
                if color_patch.size == 0:
                    keep_mask[row_idx] = False
                    continue

                # brightest-pixel guard (threshold comes from orchestrator)
                patch_max = int(cv2.cvtColor(color_patch, cv2.COLOR_BGR2GRAY).max())
                if patch_max < int(bright_max_threshold):
                    keep_mask[row_idx] = False
                    continue

                # recenter using centroid on grayscale
                patch_gray = gray_full[y:y+h, x:x+w]
                if patch_gray.size == 0:
                    keep_mask[row_idx] = False
                    continue

                cx_rel, cy_rel = intensity_weighted_centroid(patch_gray)
                cx_full = x + cx_rel
                cy_full = y + cy_rel

                new_x = int(round(cx_full - w/2))
                new_y = int(round(cy_full - h/2))
                new_x = max(0, min(new_x, W - w))
                new_y = max(0, min(new_y, H - h))

                rows[row_idx]['x'] = str(new_x)
                rows[row_idx]['y'] = str(new_y)

        progress(fr+1, total, 'recenter'); fr += 1

    cap.release()

    # write back only rows that passed the brightest-pixel threshold
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
