#!/usr/bin/env python3
from __future__ import annotations

"""
Day-time pipeline Stage 3 — Merge overlapping boxes via union-find on centroid distance.

Reads Stage 2 CSV (positives only), groups boxes per frame whose centroid distance
is <= MERGE_DIST_THRESHOLD_PX, and for each group keeps the single box with the
maximum RGB-sum weight measured on the original frame. Writes a pruned CSV with
the same schema as Stage 2.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import params
from utils import open_video, progress


def _center_to_tl(cx: float, cy: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    return x0, y0, int(w), int(h)


def _clamp_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    w = max(1, min(int(w), W))
    h = max(1, min(int(h), H))
    x = max(0, min(int(x), W - w))
    y = max(0, min(int(y), H - h))
    return x, y, w, h


def _rgb_weight(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return -1.0
    return float(crop.astype(np.float64).sum())


def run_for_video(video_path: Path) -> Path:
    stem = video_path.stem
    in_csv = (params.STAGE2_DIR / stem) / f"{stem}_patches.csv"
    assert in_csv.exists(), f"Missing Stage2 CSV for {stem}: {in_csv}"

    out_root = params.STAGE3_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"{stem}_merge.csv"

    # Load rows
    with in_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
    if not rows:
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "w", "h", "frame_number", "global_trajectory_id"])
        print("Stage3  NOTE: No Stage2 rows; wrote empty merge CSV.")
        return out_csv

    # Prepare
    cap, W, H, fps, total = open_video(video_path)
    max_frames = int(params.MAX_FRAMES) if (params.MAX_FRAMES is not None) else total

    # Group rows by frame_number
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        t = int(r.get("frame_number", r.get("frame", 0)))
        if params.MAX_FRAMES is not None and t >= max_frames:
            continue
        by_frame[t].append(r)

    kept_rows: List[dict] = []
    frames_with_dets_before = len([k for k, v in by_frame.items() if v])
    fr = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
    try:
        while fr < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            dets = by_frame.get(fr, [])
            n = len(dets)
            if n == 0:
                fr += 1
                if fr % 50 == 0:
                    progress(fr, max_frames or fr, "Stage3 merge")
                continue

            # Precompute centers and clamped boxes
            centers: List[Tuple[float, float]] = []
            boxes_tl: List[Tuple[int, int, int, int]] = []
            for r in dets:
                cx = float(r["x"])  # center coords per day-time pipeline
                cy = float(r["y"])
                w = int(r["w"]) if "w" in r else int(params.PATCH_SIZE_PX)
                h = int(r["h"]) if "h" in r else int(params.PATCH_SIZE_PX)
                x0, y0, w0, h0 = _center_to_tl(cx, cy, w, h)
                x0, y0, w0, h0 = _clamp_box(x0, y0, w0, h0, W, H)
                boxes_tl.append((x0, y0, w0, h0))
                centers.append((cx, cy))

            # Union-Find by centroid distance
            parent = list(range(n))

            def find(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def union(i: int, j: int) -> None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

            th = float(getattr(params, 'MERGE_DIST_THRESHOLD_PX', 10.0))
            for i in range(n):
                (cxi, cyi) = centers[i]
                for j in range(i + 1, n):
                    (cxj, cyj) = centers[j]
                    dx = cxi - cxj
                    dy = cyi - cyj
                    if (dx * dx + dy * dy) <= (th * th):
                        union(i, j)

            # Buckets
            groups: Dict[int, List[int]] = defaultdict(list)
            for i in range(n):
                groups[find(i)].append(i)

            # Keep max-weight per group; preserve all columns
            for root, idxs in groups.items():
                best_idx = idxs[0]
                best_wt = -1.0
                for i in idxs:
                    x0, y0, w0, h0 = boxes_tl[i]
                    wt = _rgb_weight(frame, x0, y0, w0, h0)
                    if wt > best_wt:
                        best_wt = wt
                        best_idx = i
                kept = dict(dets[best_idx])
                # Normalize fields: ensure required columns exist and are typed
                kept['x'] = float(kept['x'])
                kept['y'] = float(kept['y'])
                kept['w'] = int(float(kept.get('w', params.PATCH_SIZE_PX)))
                kept['h'] = int(float(kept.get('h', params.PATCH_SIZE_PX)))
                kept['frame_number'] = int(kept.get('frame_number', fr))
                kept_rows.append(kept)

            fr += 1
            if fr % 50 == 0:
                progress(fr, max_frames or fr, "Stage3 merge")
        progress(fr, max_frames or fr or 1, "Stage3 merge done")
    finally:
        cap.release()

    # Determine columns: keep originals plus ensure the base columns are first
    orig_fields = list(rows[0].keys())
    base = ['x', 'y', 'w', 'h', 'frame_number', 'global_trajectory_id']
    ordered = [c for c in base if c in orig_fields] + [c for c in orig_fields if c not in base]
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in kept_rows:
            w.writerow({k: r.get(k, '') for k in ordered})

    print("Stage3  Merge summary:")
    print(f"  Input rows : {len(rows)}  frames_with_dets={frames_with_dets_before}")
    print(f"  Output rows: {len(kept_rows)}  reduction={(1.0 - (len(kept_rows)/max(1,len(rows))))*100:.1f}%")
    print(f"  Wrote CSV  → {out_csv}")
    return out_csv


__all__ = ["run_for_video"]
