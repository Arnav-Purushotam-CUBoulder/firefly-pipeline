# stage7_prune_heaviest_unionfind.py  (now groups by centroid distance)
import csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import sys
import math
from typing import List, Dict

# ─── tiny progress bar ─────────────────────────────────────────
BAR_LEN = 50
def _progress(i, total, tag=''):
    frac = i / total if total else 0
    bar  = '=' * int(frac * BAR_LEN) + ' ' * (BAR_LEN - int(frac * BAR_LEN))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i == total:
        sys.stdout.write('\n')

def _clamp_box(x, y, w, h, W, H):
    w = max(1, min(int(w), W))
    h = max(1, min(int(h), H))
    x = max(0, min(int(x), W - w))
    y = max(0, min(int(y), H - h))
    return x, y, w, h

def _rgb_weight(frame_bgr, x, y, w, h):
    """Sum of R+G+B over the crop (float)."""
    crop = frame_bgr[y:y+h, x:x+w]
    if crop.size == 0:
        return -1.0
    return float(crop.astype(np.float64).sum())

def _centroid_xy(r):
    """Return (cx, cy) float centroid from a CSV row dict with x,y,w,h."""
    x, y, w, h = int(float(r['x'])), int(float(r['y'])), int(float(r['w'])), int(float(r['h']))
    return (x + w/2.0, y + h/2.0)

def _euclid(a, b):
    return float(math.hypot(a[0]-b[0], a[1]-b[1]))

def prune_overlaps_keep_heaviest_unionfind(
    orig_video_path: Path,
    csv_path: Path,
    *,
    dist_threshold_px: float = 10.0,   # NEW: max centroid distance to group boxes
    max_frames=None,
    verbose: bool = True,
):
    """
    For each frame:
      • Build graph connecting boxes whose centroid distance ≤ dist_threshold_px.
      • For each connected component, keep the single box with max RGB-sum weight.
      • Preserve any extra columns (e.g., class, logits) from the kept row.
    Overwrites the same csv_path with pruned rows.
    """
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return

    # Prepare video to compute weights
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {orig_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Group rows by frame
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        f = int(r['frame'])
        if max_frames is not None and f >= max_frames:
            continue
        by_frame[f].append(r)

    global_before = len(rows)
    out_rows: List[dict] = []

    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and fr >= max_frames:
            break

        dets = by_frame.get(fr, [])
        n = len(dets)
        if n == 0:
            fr += 1
            _progress(fr+1, max_frames or total_frames, 'stage7'); 
            continue

        # Precompute centroids and clamped boxes
        centroids = []
        boxes = []
        for r in dets:
            x, y, w, h = int(float(r['x'])), int(float(r['y'])), int(float(r['w'])), int(float(r['h']))
            x, y, w, h = _clamp_box(x, y, w, h, W, H)
            boxes.append((x,y,w,h))
            centroids.append((x + w/2.0, y + h/2.0))

        # Union-Find
        parent = list(range(n))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        # Connect if within threshold
        th = float(dist_threshold_px)
        for i in range(n):
            for j in range(i+1, n):
                if _euclid(centroids[i], centroids[j]) <= th:
                    union(i, j)

        # Buckets by root
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        # For each group choose max-weight index
        for root, idxs in groups.items():
            max_w = -1.0; max_idx = idxs[0]
            for i in idxs:
                x,y,w,h = boxes[i]
                wt = _rgb_weight(frame, x, y, w, h)
                if wt > max_w:
                    max_w = wt; max_idx = i
            kept = dict(dets[max_idx])  # preserve all columns
            out_rows.append(kept)

        _progress(fr, max_frames or total_frames, 'stage7'); fr += 1

    cap.release()

    global_after = len(out_rows)
    # Determine fieldnames: keep originals; ensure base columns at front
    orig_fields = list(rows[0].keys())
    base = ['frame','x','y','w','h']
    ordered = [c for c in base if c in orig_fields] + [c for c in orig_fields if c not in base]
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, '') for k in ordered})

    if verbose:
        print("\nStage-7 summary")
        print(f"  Boxes before: {global_before}")
        print(f"  Boxes after : {global_after}")
        print(f"  Deleted     : {global_before - global_after}")
