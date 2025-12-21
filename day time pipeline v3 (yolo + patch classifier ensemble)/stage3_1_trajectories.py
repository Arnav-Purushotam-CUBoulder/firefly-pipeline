#!/usr/bin/env python3
from __future__ import annotations

"""
Stage 3.1: trajectory grouping for Stage 3 detections (no filtering).

Reads the Stage 3 CSV for a video:

  frame_idx, video_name, x, y, w, h, conf, det_id

and treats each detection as a 3D point:

  (cx, cy, t) where cx = x + w/2, cy = y + h/2, t = frame_idx.

Detections that are close in (x,y,t) space are grouped into trajectories
using a simple union-find scheme. A new CSV is written with an extra
column 'traj_id' indicating which trajectory each detection belongs to:

  STAGE3_DIR/<stem>/<stem>_patches_traj.csv

No detections are removed or modified; this stage is purely for analysis.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import math

import params


@dataclass
class _Det3D:
    row_idx: int
    frame: int
    cx: float
    cy: float


def _read_stage3(stem: str) -> Tuple[Path, List[dict], List[_Det3D], List[str]]:
    """Read Stage 3 CSV and return (path, rows, 3D detections, fieldnames)."""
    s3_csv = (params.STAGE3_DIR / stem) / f"{stem}_patches.csv"
    if not s3_csv.exists():
        raise FileNotFoundError(f"Stage3 CSV not found for stem '{stem}': {s3_csv}")

    rows: List[dict] = []
    dets: List[_Det3D] = []

    with s3_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or [
            "frame_idx",
            "video_name",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "det_id",
        ]
        for row_idx, row in enumerate(reader):
            try:
                t = int(row.get("frame_idx") or row.get("frame_number") or 0)
                x = float(row["x"])
                y = float(row["y"])
                w = float(row.get("w", params.PATCH_SIZE_PX))
                h = float(row.get("h", params.PATCH_SIZE_PX))
            except Exception:
                continue
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            rows.append(row)
            dets.append(_Det3D(row_idx=len(rows) - 1, frame=t, cx=cx, cy=cy))

    return s3_csv, rows, dets, fieldnames


def _build_trajectories(dets: List[_Det3D]) -> Dict[int, List[int]]:
    """Group detections into trajectories via union-find in (x,y,t).

    Returns mapping root_id -> list of indices into dets.
    """
    if not dets:
        return {}

    link_radius = float(getattr(params, "STAGE3_1_LINK_RADIUS_PX", 12.0))
    max_gap = int(getattr(params, "STAGE3_1_MAX_FRAME_GAP", 3))
    time_scale = float(getattr(params, "STAGE3_1_TIME_SCALE", 1.0))

    n = len(dets)
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

    # Index by frame to avoid O(n^2) across all points
    by_frame: Dict[int, List[int]] = {}
    for i, d in enumerate(dets):
        by_frame.setdefault(d.frame, []).append(i)

    frames_sorted = sorted(by_frame.keys())

    for t in frames_sorted:
        src_idxs = by_frame.get(t, [])
        if not src_idxs:
            continue
        for dt in range(1, max_gap + 1):
            t2 = t + dt
            tgt_idxs = by_frame.get(t2)
            if not tgt_idxs:
                continue
            for i in src_idxs:
                di = dets[i]
                for j in tgt_idxs:
                    dj = dets[j]
                    dx = dj.cx - di.cx
                    dy = dj.cy - di.cy
                    dz = (dj.frame - di.frame) * time_scale
                    dist_sq = dx * dx + dy * dy + dz * dz
                    if dist_sq <= link_radius * link_radius:
                        union(i, j)

    trajs: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        trajs.setdefault(r, []).append(i)
    return trajs


def run_for_video(video_path: Path) -> Path:
    """Group Stage 3 detections into trajectories for a single video.

    Returns path to the trajectory CSV:
      STAGE3_DIR/<stem>/<stem>_patches_traj.csv
    """
    stem = video_path.stem
    s3_csv, rows, dets, fieldnames = _read_stage3(stem)

    out_root = params.STAGE3_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"{stem}_patches_traj.csv"

    if not rows or not dets:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(fieldnames) + ["traj_id"],
            )
            writer.writeheader()
        print(f"Stage3.1  NOTE: No Stage3 rows for {stem}. Wrote empty traj CSV → {out_csv}")
        return out_csv

    trajs = _build_trajectories(dets)

    # Assign compact trajectory IDs
    traj_ids: Dict[int, int] = {}
    for new_id, root in enumerate(sorted(trajs.keys())):
        for det_idx in trajs[root]:
            ri = dets[det_idx].row_idx
            traj_ids[ri] = new_id

    # Write CSV with traj_id column
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fieldnames) + ["traj_id"],
        )
        writer.writeheader()
        for idx, row in enumerate(rows):
            row_out = dict(row)
            row_out["traj_id"] = int(traj_ids.get(idx, -1))
            writer.writerow(row_out)

    total_traj = len(trajs)
    print(
        f"Stage3.1  {stem}: detections={len(rows)} trajectories={total_traj} "
        f"(link_radius={getattr(params,'STAGE3_1_LINK_RADIUS_PX',None)}, "
        f"max_gap={getattr(params,'STAGE3_1_MAX_FRAME_GAP',None)}, "
        f"time_scale={getattr(params,'STAGE3_1_TIME_SCALE',None)})"
    )
    print(f"Stage3.1  Wrote trajectory CSV → {out_csv}")
    return out_csv


__all__ = ["run_for_video"]

