#!/usr/bin/env python3
from __future__ import annotations

"""
Stage 5: 3D visualization of detections over time (geometry export).

For each video, this stage reads the Stage 3 CSV of per-frame positive
patch detections:

    frame_idx, video_name, x, y, w, h, conf, det_id

Treats time (frame_idx) as the third dimension (Z) and builds a 3D
point cloud where each detection is represented as a sphere at (x, y, t).

Detections are grouped into temporal blocks of
STAGE5_BLOCK_SIZE_FRAMES frames (e.g. 0–999, 1000–1999, …) and each
block is exported as a separate 3D geometry file:

    STAGE5_DIR/<video_stem>/<video_stem>_block_<start>-<end>.vtp

The .vtp format is VTK PolyData. You can open these in PyVista,
ParaView, or other VTK-compatible 3D viewers and freely rotate/zoom.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import params


def _load_stage3_detections(stem: str) -> List[Tuple[int, float, float, int]]:
    """Load Stage 3 detections for a video stem.

    Returns list of (frame_idx, x_center, y_center, rejected),
    where rejected=1 for trajectories not selected by Stage3.1 (if available)
    and 0 otherwise.
    """
    s3_dir = params.STAGE3_DIR / stem
    preferred_all = s3_dir / f"{stem}_patches_motion_all.csv"
    preferred = s3_dir / f"{stem}_patches_motion.csv"
    fallback = s3_dir / f"{stem}_patches.csv"
    if preferred_all.exists():
        s3_csv = preferred_all
    elif preferred.exists():
        s3_csv = preferred
    else:
        s3_csv = fallback
    if not s3_csv.exists():
        raise FileNotFoundError(f"Stage 3 CSV not found for stem '{stem}': {s3_csv}")

    dets: List[Tuple[int, float, float, int]] = []
    with s3_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
            rej = 0
            # If stage3.1 wrote traj_is_selected, treat unselected as rejected.
            sel_raw = row.get("traj_is_selected")
            if sel_raw is not None:
                s = str(sel_raw).strip()
                if s in {"", "0", "False", "false"}:
                    rej = 1
            dets.append((t, cx, cy, rej))
    return dets


def _group_into_blocks(
    dets: List[Tuple[int, float, float, int]], block_size: int
) -> Dict[int, List[Tuple[int, float, float, int]]]:
    """Group detections into temporal blocks of length block_size.

    Keys are block indices (0, 1, 2, ...), values are lists of
    (frame_idx, cx, cy) in that block.
    """
    blocks: Dict[int, List[Tuple[int, float, float, int]]] = {}
    for t, cx, cy, rej in dets:
        if t < 0:
            continue
        block_idx = t // block_size
        blocks.setdefault(block_idx, []).append((t, cx, cy, rej))
    return blocks


def _ensure_pyvista():
    try:
        import pyvista as pv  # type: ignore[import]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Stage 5 (3D render) requires pyvista. Install with:\n\n"
            "    pip install pyvista\n"
        ) from e
    return pv


def _render_block(
    stem: str,
    block_idx: int,
    block_dets: List[Tuple[int, float, float, int]],
    block_size: int,
    out_dir: Path,
    sphere_radius: float,
) -> Path:
    """Export one temporal block of detections as a 3D geometry (.vtp)."""
    pv = _ensure_pyvista()

    if not block_dets:
        raise ValueError("Block has no detections to render.")

    # Convert to arrays
    t_vals = np.array([t for (t, _, _, _) in block_dets], dtype=np.float32)
    x_vals = np.array([cx for (_, cx, _, _) in block_dets], dtype=np.float32)
    y_vals = np.array([cy for (_, _, cy, _) in block_dets], dtype=np.float32)
    rejected_vals = np.array([rej for (_, _, _, rej) in block_dets], dtype=np.int8)

    # Normalize time within this block from 0..block_size-1
    block_start = int(block_idx * block_size)
    z_vals = t_vals - float(block_start)

    points = np.vstack([x_vals, y_vals, z_vals]).T

    # Build glyph spheres as a single PolyData (with 'rejected' flag).
    cloud = pv.PolyData(points)
    cloud["rejected"] = rejected_vals
    sphere = pv.Sphere(radius=float(sphere_radius))
    glyphs = cloud.glyph(scale=False, geom=sphere)

    out_dir.mkdir(parents=True, exist_ok=True)
    start_f = block_start
    end_f = block_start + block_size - 1
    out_path = out_dir / f"{stem}_block_{start_f:06d}-{end_f:06d}.vtp"

    glyphs.save(str(out_path))
    return out_path


def run_for_video(video_path: Path) -> List[Path]:
    """Generate 3D geometry file(s) for a single video's detections.

    Returns list of output file paths for that video.
    """
    assert video_path.exists(), f"Video not found: {video_path}"
    stem = video_path.stem

    dets = _load_stage3_detections(stem)
    if not dets:
        print(f"Stage5  NOTE: No Stage3 detections found for {stem}.")
        return []

    block_size = int(getattr(params, "STAGE5_BLOCK_SIZE_FRAMES", 1000))
    sphere_radius = float(getattr(params, "STAGE5_SPHERE_RADIUS", 5.0))

    blocks = _group_into_blocks(dets, block_size)
    if not blocks:
        print(f"Stage5  NOTE: After grouping, no blocks to render for {stem}.")
        return []

    out_root = params.STAGE5_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)

    out_files: List[Path] = []
    print(
        f"Stage5  Rendering 3D detections for {stem}: "
        f"{len(dets)} detections, block_size={block_size} frames, blocks={len(blocks)}"
    )
    for block_idx in sorted(blocks.keys()):
        block_dets = blocks[block_idx]
        if not block_dets:
            continue
        try:
            out_file = _render_block(
                stem=stem,
                block_idx=block_idx,
                block_dets=block_dets,
                block_size=block_size,
                out_dir=out_root,
                sphere_radius=sphere_radius,
            )
            out_files.append(out_file)
            print(f"Stage5  Block {block_idx}: {len(block_dets)} detections → {out_file}")
        except Exception as e:
            print(f"Stage5  Warning: failed to render block {block_idx} for {stem}: {e}")

    if not out_files:
        print(f"Stage5  NOTE: No 3D files produced for {stem}.")
    else:
        print(f"Stage5  Done for {stem}. Files: {len(out_files)} → {out_root}")
    return out_files


__all__ = ["run_for_video"]
