#!/usr/bin/env python3
"""
Stage 8.6 Iterative blackout + full re-run (uses orchestrator's params directly)

Per RUN:
  1) Read the current MAIN CSV (already post Stage 8/8.5).
  2) Build a blacked video by painting 10�10 black squares at every firefly center (per frame).
  3) Run Stage1�2�3�4�7�8�8.5 directly on that blacked video using the SAME params you set in the orchestrator.
  4) Merge the run's final CSV into the MAIN CSV (+ append matching rows into main *_fireflies_logits.csv)
     with a tiny proximity dedupe (default 2 px).

All Stage 8.6 artifacts live under:
  ROOT / "stage8.6" / <video_stem> / run_XX/
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import cv2
import numpy as np
from audit_trail import AuditTrail

#####################################################################################
# Import the stage functions (same names you use in orchestrator)
#####################################################################################
from stage2_recenter import recenter_boxes_with_centroid as STAGE2_RECENTER
from stage3_area_filter import filter_boxes_by_area as STAGE3_AREA_FILTER
from stage4_cnn_filter import classify_and_filter_csv as STAGE4_CNN_FILTER
from stage7_merge import prune_overlaps_keep_heaviest_unionfind as STAGE7_MERGE
from stage8_gaussian_centroid import recenter_gaussian_centroid as STAGE8_GAUSSIAN_CENTROID
from stage8_5_blob_area_filter import stage8_5_prune_by_blob_area as STAGE8_5_BLOB_FILTER

#####################################################################################
# Stage-1 resolver and safe-call (handles differing signatures)
#####################################################################################
import inspect

def _resolve_stage1(stage1_impl: str):
    if stage1_impl == 'blob':
        from stage1_detect import detect_blobs_to_csv as f
    elif stage1_impl == 'cc_cpu':
        from stage1_detect_cc_cpu import detect_stage1_to_csv as f
    elif stage1_impl == 'cc_cuda':
        from stage1_detect_cc_cuda import detect_stage1_to_csv as f
    else:
        raise ValueError(f"Unknown stage1_impl={stage1_impl!r} (expected 'blob'|'cc_cpu'|'cc_cuda')")
    return f

def _filtered_call(fn, **kwargs):
    sig = inspect.signature(fn)
    call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**call_kwargs)

#####################################################################################
# Pull ALL params/paths directly from the running orchestrator.
# (Avoid circular imports by reading __main__ at call time.)
#####################################################################################
def _orc():
    """Return the running orchestrator module (loaded as __main__)."""
    import __main__ as ORC  # orchestrator.py when executed is __main__
    return ORC

def _pack_stage1_params_for(ORC, variant: str) -> dict:
    """Build the right Stage-1 kwargs from orchestrator constants for the chosen variant."""
    if variant == 'blob':
        return dict(
            sbd_min_area_px=ORC.SBD_MIN_AREA_PX,
            sbd_max_area_scale=ORC.SBD_MAX_AREA_SCALE,
            sbd_min_dist=ORC.SBD_MIN_DIST,
            sbd_min_repeat=ORC.SBD_MIN_REPEAT,
            use_clahe=ORC.USE_CLAHE, clahe_clip=ORC.CLAHE_CLIP, clahe_tile=ORC.CLAHE_TILE,
            use_tophat=ORC.USE_TOPHAT, tophat_ksize=ORC.TOPHAT_KSIZE,
            use_dog=ORC.USE_DOG, dog_sigma1=ORC.DOG_SIGMA1, dog_sigma2=ORC.DOG_SIGMA2,
        )
    elif variant in ('cc_cpu', 'cc_cuda'):
        d = dict(
            min_area_px=ORC.CC_MIN_AREA_PX, max_area_scale=ORC.CC_MAX_AREA_SCALE,
            use_clahe=ORC.CC_USE_CLAHE, clahe_clip=ORC.CC_CLAHE_CLIP, clahe_tile=ORC.CC_CLAHE_TILE,
            use_tophat=ORC.CC_USE_TOPHAT, tophat_ksize=ORC.CC_TOPHAT_KSIZE,
            use_dog=ORC.CC_USE_DOG, dog_sigma1=ORC.CC_DOG_SIGMA1, dog_sigma2=ORC.CC_DOG_SIGMA2,
            threshold_method=ORC.CC_THRESHOLD_METHOD, fixed_threshold=ORC.CC_FIXED_THRESHOLD,
            open_ksize=ORC.CC_OPEN_KSIZE, connectivity=ORC.CC_CONNECTIVITY,
        )
        if variant == 'cc_cuda':
            # only cc_cuda understands these; guard with hasattr for safety
            if hasattr(ORC, "CC_BATCH_SIZE"):
                d["batch_size"] = ORC.CC_BATCH_SIZE
            if hasattr(ORC, "CC_PREPROC_BACKEND"):
                d["preproc_backend"] = ORC.CC_PREPROC_BACKEND
            if hasattr(ORC, "CC_ADAPTIVE_C"):
                d["adaptive_c"] = ORC.CC_ADAPTIVE_C
        return d
    else:
        raise ValueError(f"Unknown Stage-1 variant: {variant!r}")

#####################################################################################
# CSV helpers
#####################################################################################
def _read_csv_rows(p: Path) -> List[dict]:
    try:
        with p.open("r", newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []

def _write_csv_rows(p: Path, rows: List[dict], field_order: Sequence[str]):
    fieldnames = list(field_order)
    # Keep stage-added columns stable
    for extra in ["class","background_logit","firefly_logit","firefly_confidence","xy_semantics"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def _frame_firefly_centers(rows: List[dict]) -> Dict[int, List[Tuple[float,float]]]:
    out: Dict[int, List[Tuple[float,float]]] = {}
    has_class = ("class" in rows[0].keys()) if rows else False
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

#####################################################################################
# Video helpers
#####################################################################################
def _open_writer_like(cap: cv2.VideoCapture, out_path: Path) -> cv2.VideoWriter:
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or float(fps) <= 1e-3:
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H))

def _clamped_center_box(cx: float, cy: float, w: int, h: int, W: int, H: int) -> Tuple[int,int,int,int]:
    w = max(1, int(round(w))); h = max(1, int(round(h)))
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w)); y0 = max(0, min(y0, H - h))
    return x0, y0, x0 + w, y0 + h

def _build_blacked_video(
    *, orig_video_path: Path, centers_by_frame: Dict[int, List[Tuple[float,float]]],
    out_video_path: Path, blackout_w: int = 10, blackout_h: int = 10, max_frames=None
):
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage8.6] Could not open video: {orig_video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None:
        total = min(total, int(max_frames))

    out = _open_writer_like(cap, out_video_path)

    fr = 0
    BAR = 50
    while True:
        if max_frames is not None and fr >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        for (cx, cy) in centers_by_frame.get(fr, []):
            x0, y0, x1, y1 = _clamped_center_box(cx, cy, blackout_w, blackout_h, W, H)
            frame[y0:y1, x0:x1] = 0  # fill black
        out.write(frame)

        # progress
        frac = 0 if total == 0 else min(1.0, max(0.0, (fr+1) / total))
        bar  = '=' * int(frac * BAR) + ' ' * (BAR - int(frac * BAR))
        try:
            import sys
            sys.stdout.write(f'\r[stage8.6] build blacked video [{bar}] {int(frac*100):3d}%')
            sys.stdout.flush()
        except Exception:
            pass
        fr += 1

    cap.release(); out.release()
    try:
        import sys; sys.stdout.write('\n')
    except Exception:
        pass
    print(f"[stage8.6] Blacked video � {out_video_path}")

#####################################################################################
# Merge helpers
#####################################################################################
def _euclid2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    dx = float(a[0]) - float(b[0]); dy = float(a[1]) - float(b[1])
    return dx*dx + dy*dy

def _dedupe_merge(base_rows: List[dict], add_rows: List[dict], dist_px: float) -> Tuple[List[dict], List[dict]]:
    if not add_rows:
        return [], []
    has_class = ("class" in base_rows[0].keys()) if base_rows else False
    centers_by_frame: Dict[int, List[Tuple[float,float]]] = {}
    for r in base_rows:
        try:
            if has_class and r.get("class") != "firefly":
                continue
            f = int(r["frame"])
            w = int(round(float(r.get("w", 10)))); h = int(round(float(r.get("h", 10))))
            if str(r.get("xy_semantics","")).lower() == "center":
                cx = float(r["x"]); cy = float(r["y"])
            else:
                cx = float(r["x"]) + w/2.0; cy = float(r["y"]) + h/2.0
            centers_by_frame.setdefault(f, []).append((cx, cy))
        except Exception:
            continue

    thr2 = float(dist_px * dist_px)
    kept, dropped = [], []
    acc_by_frame: Dict[int, List[Tuple[float,float]]] = {}
    for r in add_rows:
        try:
            f = int(r["frame"])
            w = int(round(float(r.get("w", 10)))); h = int(round(float(r.get("h", 10))))
            if str(r.get("xy_semantics","")).lower() == "center":
                cx = float(r["x"]); cy = float(r["y"])
            else:
                cx = float(r["x"]) + w/2.0; cy = float(r["y"]) + h/2.0

            # against base
            if any(_euclid2((cx,cy), c) <= thr2 for c in centers_by_frame.get(f, [])):
                dropped.append(r); continue
            # among kept additions
            if any(_euclid2((cx,cy), c) <= thr2 for c in acc_by_frame.get(f, [])):
                dropped.append(r); continue

            kept.append(r)
            acc_by_frame.setdefault(f, []).append((cx,cy))
        except Exception:
            dropped.append(r)
    return kept, dropped

#####################################################################################
# Public API
#####################################################################################
def stage8_6_run(
    *,
    orig_video_path: Path,
    main_csv_path: Path,
    num_runs: int | None = None,
    audit: Optional[AuditTrail] = None,
    stage1_impl: Optional[str] = None,
    stage1_params: Optional[dict] = None,
) -> int:
    """
    Execute Stage 8.6 end-to-end using orchestrator params. Returns total rows added to MAIN CSV.
    """
    ORC = _orc()

    # derive knobs from orchestrator (single source of truth)
    ROOT                     = ORC.ROOT
    MAX_FRAMES               = ORC.MAX_FRAMES
    RUN_STAGE4               = ORC.RUN_STAGE4
    USE_CNN_FILTER           = ORC.USE_CNN_FILTER

    # Stage-2
    BRIGHT_MAX_THRESHOLD     = ORC.BRIGHT_MAX_THRESHOLD

    # Stage-3
    AREA_THRESHOLD_PX        = ORC.AREA_THRESHOLD_PX

    # Stage-4
    CNN_MODEL_PATH           = ORC.CNN_MODEL_PATH
    CNN_BACKBONE             = ORC.CNN_BACKBONE
    CNN_CLASS_TO_KEEP        = ORC.CNN_CLASS_TO_KEEP
    CNN_PATCH_W              = ORC.CNN_PATCH_W
    CNN_PATCH_H              = ORC.CNN_PATCH_H
    FIREFLY_CONF_THRESH      = ORC.FIREFLY_CONF_THRESH
    DROP_BACKGROUND_ROWS     = ORC.DROP_BACKGROUND_ROWS
    IMAGENET_NORMALIZE       = ORC.IMAGENET_NORMALIZE
    PRINT_LOAD_STATUS        = ORC.PRINT_LOAD_STATUS
    FAIL_IF_WEIGHTS_MISSING  = ORC.FAIL_IF_WEIGHTS_MISSING
    DEBUG_SAVE_PATCHES_DIR   = ORC.DEBUG_SAVE_PATCHES_DIR

    # Stage-7
    STAGE7_DIST_THRESHOLD_PX = ORC.STAGE7_DIST_THRESHOLD_PX
    STAGE7_VERBOSE           = ORC.STAGE7_VERBOSE

    # Stage-8
    STAGE8_PATCH_W           = ORC.STAGE8_PATCH_W
    STAGE8_PATCH_H           = ORC.STAGE8_PATCH_H
    STAGE8_GAUSSIAN_SIGMA    = ORC.STAGE8_GAUSSIAN_SIGMA
    STAGE8_VERBOSE           = ORC.STAGE8_VERBOSE
    DIR_STAGE8_CROPS         = ORC.DIR_STAGE8_CROPS

    # Stage-8.5 / 9 shared
    MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION = ORC.MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION

    # Stage-8.6 knobs
    STAGE8_6_RUNS            = getattr(ORC, "STAGE8_6_RUNS", 1)
    MERGE_DEDUPE_PX          = getattr(ORC, "STAGE8_6_DEDUPE_PX", 2.0)

    runs = int(num_runs if num_runs is not None else STAGE8_6_RUNS)
    if runs <= 0:
        print("[stage8.6] Skipping (STAGE8_6_RUNS <= 0)")
        return 0

    # Stage-1 choice and params: default to orchestrator if not provided
    if stage1_impl is None:
        stage1_impl = getattr(ORC, "STAGE1_VARIANT", "blob")
    if stage1_params is None:
        stage1_params = _pack_stage1_params_for(ORC, stage1_impl)

    # Resolve Stage-1 function once
    STAGE1_DETECT = _resolve_stage1(stage1_impl)

    # Load MAIN CSV
    base_rows = _read_csv_rows(main_csv_path)
    if not base_rows:
        print("[stage8.6] Main CSV is empty nothing to blackout.")
        return 0
    field_order = list(base_rows[0].keys())

    video_stem = Path(orig_video_path).stem
    stage86_root = ROOT / "stage8.6" / video_stem
    stage86_root.mkdir(parents=True, exist_ok=True)

    main_ff_logits_path = main_csv_path.with_name(main_csv_path.stem + "_fireflies_logits.csv")

    total_added = 0
    for run_idx in range(1, runs + 1):
        print(f"[stage8.6] RUN {run_idx}/{runs}")

        # Paths for this run
        run_dir = stage86_root / f"run_{run_idx:02d}"
        (run_dir / "csv files").mkdir(parents=True, exist_ok=True)
        (run_dir / "stage8 crops").mkdir(parents=True, exist_ok=True)

        run_tag        = f"{main_csv_path.stem}__s8_6run{run_idx}"
        blacked_video  = run_dir / "blacked.mp4"
        sub_csv        = run_dir / "csv files" / f"{run_tag}.csv"
        sub_ff_logits  = run_dir / "csv files" / f"{run_tag}_fireflies_logits.csv"
        sub_area_snap  = run_dir / "csv files" / f"{run_tag}_area_snapshot.csv"
        sub_s8_crops   = run_dir / "stage8 crops"  # for Stage 8

        # 1) Build blacked video from CURRENT base_rows
        centers_by_frame = _frame_firefly_centers(base_rows)
        _build_blacked_video(
            orig_video_path=orig_video_path,
            centers_by_frame=centers_by_frame,
            out_video_path=blacked_video,
            blackout_w=10, blackout_h=10,
            max_frames=MAX_FRAMES,
        )

        # 2) Run standard pipeline on the blacked video (using orchestrator's params)

        # Stage 1 (resolved variant + filtered kwargs)
        _filtered_call(
            STAGE1_DETECT,
            orig_path=blacked_video,
            csv_path=sub_csv,
            max_frames=MAX_FRAMES,
            **(stage1_params or {}),
        )

        # Stage 2
        STAGE2_RECENTER(
            orig_path=blacked_video,
            csv_path=sub_csv,
            max_frames=MAX_FRAMES,
            bright_max_threshold=BRIGHT_MAX_THRESHOLD,
            audit=audit,
            audit_video_path=blacked_video,
        )

        # Stage 3
        STAGE3_AREA_FILTER(
            csv_path=sub_csv,
            area_threshold_px=AREA_THRESHOLD_PX,
            snapshot_csv_path=sub_area_snap,
            audit=audit,
        )

        # Stage 4 (only if enabled in orchestrator)
        if ORC.RUN_STAGE4 and USE_CNN_FILTER:
            STAGE4_CNN_FILTER(
                orig_path=blacked_video,
                csv_path=sub_csv,
                max_frames=MAX_FRAMES,
                use_cnn_filter=USE_CNN_FILTER,
                model_path=CNN_MODEL_PATH,
                backbone=CNN_BACKBONE,
                class_to_keep=CNN_CLASS_TO_KEEP,
                patch_w=CNN_PATCH_W,
                patch_h=CNN_PATCH_H,
                firefly_conf_thresh=FIREFLY_CONF_THRESH,
                drop_background_rows=DROP_BACKGROUND_ROWS,
                imagenet_normalize=IMAGENET_NORMALIZE,
                print_load_status=PRINT_LOAD_STATUS,
                fail_if_weights_missing=FAIL_IF_WEIGHTS_MISSING,
                debug_save_patches_dir=DEBUG_SAVE_PATCHES_DIR,
                audit=audit,
            )

        # Stage 7
        STAGE7_MERGE(
            orig_video_path=blacked_video,
            csv_path=sub_csv,
            dist_threshold_px=STAGE7_DIST_THRESHOLD_PX,
            max_frames=MAX_FRAMES,
            verbose=STAGE7_VERBOSE,
            audit=audit,
        )

        # Stage 8
        STAGE8_GAUSSIAN_CENTROID(
            orig_video_path=blacked_video,
            csv_path=sub_csv,
            centroid_patch_w=STAGE8_PATCH_W,
            centroid_patch_h=STAGE8_PATCH_H,
            gaussian_sigma=STAGE8_GAUSSIAN_SIGMA,
            max_frames=MAX_FRAMES,
            verbose=STAGE8_VERBOSE,
            crop_dir=sub_s8_crops,
            audit=audit,
        )

        # Stage 8.5
        STAGE8_5_BLOB_FILTER(
            orig_video_path=blacked_video,
            csv_path=sub_csv,
            area_threshold_px=AREA_THRESHOLD_PX,
            min_pixel_brightness_to_be_considered_in_area_calculation=MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION,
            max_frames=MAX_FRAMES,
            verbose=True,
            audit=audit,
        )

        # 3) Merge run results into MAIN CSV (+ logits)
        run_rows = _read_csv_rows(sub_csv)
        if not run_rows:
            print(f"[stage8.6] RUN {run_idx}: produced no rows.")
            continue

        # Collect fireflies (post Stage 8/8.5: center semantics, w=h=10)
        add_rows = []
        has_class = ("class" in run_rows[0].keys())
        for r in run_rows:
            try:
                if has_class and r.get("class") != "firefly":
                    continue
                add_rows.append(r)
            except Exception:
                continue

        kept, _dropped = _dedupe_merge(base_rows, add_rows, MERGE_DEDUPE_PX)

        # audit: what we tried to add vs what dedupe rejected
        if audit is not None:
            if kept:
                audit.log_kept('08_6_neighbor_hunt', [dict(r, video=str(orig_video_path)) for r in kept])
            if _dropped:
                audit.log_removed('08_6_neighbor_hunt', 'dedupe', [dict(r, video=str(orig_video_path)) for r in _dropped])

        if not kept:
            print(f"[stage8.6] RUN {run_idx}: nothing to add after dedupe.")
            continue

        # append to main CSV
        base_rows = base_rows + kept
        _write_csv_rows(main_csv_path, base_rows, field_order)

        # append to main fireflies_logits.csv (match by frame,int(x),int(y))
        if sub_ff_logits.exists():
            ff_cols = ["x","y","t","background_logit","firefly_logit"]
            existing = []
            if main_ff_logits_path.exists():
                with main_ff_logits_path.open("r", newline="") as f:
                    rd = csv.DictReader(f)
                    if rd.fieldnames:
                        ff_cols = rd.fieldnames
                    existing = list(rd)

            kept_keys = set()
            for r in kept:
                try:
                    kept_keys.add((int(r["frame"]), int(round(float(r["x"]))), int(round(float(r["y"])))))
                except Exception:
                    continue

            new_ff_rows = []
            with sub_ff_logits.open("r", newline="") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    try:
                        k = (int(r["t"]), int(r["x"]), int(r["y"]))
                        if k in kept_keys:
                            new_ff_rows.append({k2: r.get(k2, "") for k2 in ff_cols})
                    except Exception:
                        continue

            with main_ff_logits_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=ff_cols)
                w.writeheader()
                for r in existing:
                    w.writerow({k: r.get(k, "") for k in ff_cols})
                for r in new_ff_rows:
                    w.writerow({k: r.get(k, "") for k in ff_cols})

        print(f"[stage8.6] RUN {run_idx}: added {len(kept)} new rows � {main_csv_path.name}")
        total_added += len(kept)

    print(f"[stage8.6] TOTAL new rows added across runs: {total_added}")
    return total_added

# Back-compat alias if old name is referenced anywhere
stage8_6_iterative_blackout = stage8_6_run
