#!/usr/bin/env python3
"""
Stage 8.6 — Iterative blackout + mini-pipeline re-run

For each RUN:
  • Build a blacked-out video: for every frame, fill 10×10 black squares
    at all centers of current firefly detections from the main CSV.
  • Re-run your standard detection pipeline on that blacked video:
      Stage1 Detect → Stage2 Recenter → Stage3 Area Filter
      → Stage4 CNN → Stage7 Merge → Stage8 Gaussian Centroid → Stage8.5 Blob Filter
    (using the SAME params you already use elsewhere).
  • Merge newly found fireflies from the run's final CSV back into the original
    main CSV (and append logits into <stem>_fireflies_logits.csv).
  • Repeat for N runs (configurable).

Notes
-----
- This file does NOT know individual stage params; it receives a callback
  `run_subpipeline_fn(video_path, stem_override)` implemented in orchestrator
  that runs Stage1→8.5 and returns (final_csv_path, fireflies_logits_csv_path).
- Minimal, internal de-dup safeguards prevent exact near-duplicate rows
  when merging back into the original CSV (2 px default). This is purely to
  avoid accidental duplicates across runs.

API
---
stage8_6_iterative_blackout(
    orig_video_path: Path,
    csv_path: Path,
    num_runs: int,
    run_subpipeline_fn: Callable[[Path, str], tuple[Path, Path | None]],
    temp_root: Optional[Path] = None,
    merge_dedupe_dist_px: float = 2.0,
    verbose: bool = True,
) -> int   # returns total newly-added rows across all runs
"""

from __future__ import annotations
import csv
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ──────────────────────────── CSV helpers ────────────────────────────
def _read_csv_rows(p: Path) -> List[dict]:
    try:
        with p.open("r", newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []

def _write_csv_rows(p: Path, rows: List[dict], field_order: Sequence[str]):
    fieldnames = list(field_order)
    # Ensure these fields exist at write time (harmless if absent in originals)
    for extra in ["class","background_logit","firefly_logit","firefly_confidence","xy_semantics"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def _frame_firefly_centers(rows: List[dict], has_class: bool) -> Dict[int, List[Tuple[float,float]]]:
    out: Dict[int, List[Tuple[float,float]]] = {}
    for r in rows:
        try:
            if has_class and r.get("class") != "firefly":
                continue
            f = int(r["frame"])
            w = int(round(float(r.get("w", 10))))
            h = int(round(float(r.get("h", 10))))
            if str(r.get("xy_semantics","")).lower() == "center":
                cx = float(r["x"]); cy = float(r["y"])
            else:
                cx = float(r["x"]) + w/2.0
                cy = float(r["y"]) + h/2.0
            out.setdefault(f, []).append((cx, cy))
        except Exception:
            continue
    return out


# ──────────────────────────── Video helpers ────────────────────────────
def _open_writer_like(cap: cv2.VideoCapture, out_path: Path) -> cv2.VideoWriter:
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isclose(float(fps), 0.0):
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H))

def _clamped_center_box(cx: float, cy: float, w: int, h: int, W: int, H: int) -> Tuple[int,int,int,int]:
    w = max(1, int(round(w))); h = max(1, int(round(h)))
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w)); y0 = max(0, min(y0, H - h))
    return x0, y0, x0 + w, y0 + h

def _build_blacked_video(
    *,
    orig_video_path: Path,
    centers_by_frame: Dict[int, List[Tuple[float,float]]],
    out_video_path: Path,
    blackout_w: int = 10,
    blackout_h: int = 10,
    verbose: bool = True,
) -> Tuple[int,int,int]:
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage8.6] Could not open video: {orig_video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out = _open_writer_like(cap, out_video_path)
    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # blackout all current fireflies for this frame
        for (cx, cy) in centers_by_frame.get(fr, []):
            x0, y0, x1, y1 = _clamped_center_box(cx, cy, blackout_w, blackout_h, W, H)
            frame[y0:y1, x0:x1] = 0  # fill black
        out.write(frame)
        fr += 1

    cap.release(); out.release()
    if verbose:
        print(f"[stage8.6] Built blacked video: {out_video_path.name}  frames={fr}  size=({W}x{H})")
    return W, H, fr


# ──────────────────────────── Merge helpers ────────────────────────────
def _euclid2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    dx = float(a[0]) - float(b[0]); dy = float(a[1]) - float(b[1])
    return dx*dx + dy*dy

def _collect_existing_centers_by_frame(rows: List[dict], has_class: bool) -> Dict[int, List[Tuple[float,float]]]:
    return _frame_firefly_centers(rows, has_class)

def _collect_new_fireflies(rows: List[dict]) -> List[dict]:
    """From a *mini-pipeline* final CSV (post Stage 8/8.5), collect FIRELY rows."""
    out = []
    has_class = ("class" in rows[0].keys()) if rows else False
    for r in rows:
        try:
            if has_class and r.get("class") != "firefly":
                continue
            # Expect center semantics and w=h=10 after Stage 8
            out.append(r)
        except Exception:
            continue
    return out

def _dedupe_merge(
    *,
    base_rows: List[dict],
    add_rows: List[dict],
    dist_px: float = 2.0,
) -> Tuple[List[dict], List[dict]]:
    """Return (kept_add_rows, dropped_add_rows) after checking proximity to base_rows in same frames."""
    if not add_rows:
        return [], []
    has_class = ("class" in base_rows[0].keys()) if base_rows else False
    centers_by_frame = _collect_existing_centers_by_frame(base_rows, has_class)

    thr2 = float(dist_px*dist_px)
    kept, dropped = [], []
    acc_by_frame: Dict[int, List[Tuple[float,float]]] = {}  # also dedupe among the new additions
    for r in add_rows:
        try:
            f = int(r["frame"])
            w = int(round(float(r.get("w", 10)))); h = int(round(float(r.get("h", 10))))
            if str(r.get("xy_semantics","")).lower() == "center":
                cx = float(r["x"]); cy = float(r["y"])
            else:
                cx = float(r["x"]) + w/2.0; cy = float(r["y"]) + h/2.0
            # against base
            clash = any(_euclid2((cx,cy), c) <= thr2 for c in centers_by_frame.get(f, []))
            if clash:
                dropped.append(r); continue
            # against previously kept new rows in same frame
            clash2 = any(_euclid2((cx,cy), c) <= thr2 for c in acc_by_frame.get(f, []))
            if clash2:
                dropped.append(r); continue
            # keep
            kept.append(r)
            acc_by_frame.setdefault(f, []).append((cx,cy))
        except Exception:
            dropped.append(r)
    return kept, dropped


# ──────────────────────────── Public API ────────────────────────────
def stage8_6_iterative_blackout(
    *,
    orig_video_path: Path,
    csv_path: Path,
    num_runs: int,
    run_subpipeline_fn: Callable[[Path, str], Tuple[Path, Optional[Path]]],
    temp_root: Optional[Path] = None,
    merge_dedupe_dist_px: float = 2.0,
    verbose: bool = True,
) -> int:
    """
    See module docstring. Returns total # of rows newly added to the main CSV across all runs.
    """
    assert num_runs >= 1, "num_runs must be >= 1"

    base_dir = csv_path.parent
    stem = csv_path.stem
    video_stem = Path(orig_video_path).stem
    tmp_root = Path(temp_root) if temp_root is not None else (base_dir / "stage8_6_iter" / video_stem)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Load the *current* CSV once; then update in-memory as we add rows each run
    base_rows = _read_csv_rows(csv_path)
    if not base_rows:
        if verbose:
            print("[stage8.6] Current CSV is empty — nothing to blackout; skipping.")
        return 0
    field_order = list(base_rows[0].keys())
    has_class = ("class" in base_rows[0].keys())

    total_added = 0

    for run_idx in range(1, num_runs + 1):
        if verbose:
            print(f"[stage8.6] RUN {run_idx}/{num_runs}")

        # 1) Build centers index from *current* base_rows (includes additions from prior runs)
        centers_by_frame = _frame_firefly_centers(base_rows, has_class)

        # 2) Create blacked-out video for this run
        run_tag = f"{stem}__s8_6run{run_idx}"
        out_video = tmp_root / f"{run_tag}.mp4"
        _build_blacked_video(
            orig_video_path=orig_video_path,
            centers_by_frame=centers_by_frame,
            out_video_path=out_video,
            blackout_w=10,
            blackout_h=10,
            verbose=verbose,
        )

        # 3) Run your Stage1→Stage8.5 mini-pipeline on the blacked video
        #    (Orchestrator provides this callback and returns final CSVs)
        mini_csv, mini_ff_logits = run_subpipeline_fn(out_video, run_tag)
        mini_rows_all = _read_csv_rows(mini_csv)
        mini_fireflies = _collect_new_fireflies(mini_rows_all)

        if verbose:
            print(f"[stage8.6] RUN {run_idx}: mini-pipeline produced {len(mini_fireflies)} fireflies")

        # 4) Dedupe vs the original CSV (and among the new set), then merge
        to_add, _dropped = _dedupe_merge(
            base_rows=base_rows,
            add_rows=mini_fireflies,
            dist_px=merge_dedupe_dist_px,
        )

        if not to_add:
            if verbose:
                print(f"[stage8.6] RUN {run_idx}: nothing to add after dedupe.")
            continue

        # 5) Append to main CSV (in-memory), then write to disk
        base_rows = base_rows + to_add
        _write_csv_rows(csv_path, base_rows, field_order)

        # 6) Append matching rows into <stem>_fireflies_logits.csv
        ff_csv = csv_path.with_name(csv_path.stem + "_fireflies_logits.csv")
        ff_cols = ["x","y","t","background_logit","firefly_logit"]
        existing = []
        if ff_csv.exists():
            try:
                with ff_csv.open("r", newline="") as f:
                    rd = csv.DictReader(f)
                    if rd.fieldnames:
                        ff_cols = rd.fieldnames
                    existing = list(rd)
            except Exception:
                pass

        # Read mini ff logits, filter to rows that correspond to the kept additions (frame & rounded coords)
        new_ff_rows = []
        if mini_ff_logits and Path(mini_ff_logits).exists():
            kept_keys = set()
            for r in to_add:
                try:
                    k = (int(r["frame"]), int(round(float(r["x"]))), int(round(float(r["y"]))))

                    kept_keys.add(k)
                except Exception:
                    continue
            with Path(mini_ff_logits).open("r", newline="") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    try:
                        k = (int(r["t"]), int(r["x"]), int(r["y"]))
                        if k in kept_keys:
                            new_ff_rows.append({k2: r.get(k2, "") for k2 in ff_cols})
                    except Exception:
                        continue

        with ff_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ff_cols)
            w.writeheader()
            for r in existing:
                w.writerow({k: r.get(k, "") for k in ff_cols})
            for r in new_ff_rows:
                w.writerow({k: r.get(k, "") for k in ff_cols})

        added_now = len(to_add)
        total_added += added_now
        if verbose:
            print(f"[stage8.6] RUN {run_idx}: added {added_now} new rows → {csv_path.name}")

    if verbose:
        print(f"[stage8.6] TOTAL new rows added across runs: {total_added}")
    return total_added


# Backwards-compat alias (so existing imports still work if name wasn't updated)
stage8_6_neighbor_hunt = stage8_6_iterative_blackout
