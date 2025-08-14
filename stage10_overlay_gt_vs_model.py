#!/usr/bin/env python3
import csv, sys, re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BAR_LEN = 50
def _progress(i, total, tag=''):
    total = max(1, int(total or 1))
    frac  = min(1.0, max(0.0, i / float(total)))
    bar   = '=' * int(frac * BAR_LEN) + ' ' * (BAR_LEN - int(frac * BAR_LEN))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i >= total: sys.stdout.write('\n')

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _clamp_box(x0:int, y0:int, w:int, h:int, W:int, H:int) -> Tuple[int,int,int,int]:
    w = max(1, min(int(w), W))
    h = max(1, min(int(h), H))
    x0 = max(0, min(int(x0), W - w))
    y0 = max(0, min(int(y0), H - h))
    return x0, y0, w, h

# ──────────────────────────────────────────────────────────────
# Original overlay helpers (GT vs Model, pixel-overlap = yellow)
# ──────────────────────────────────────────────────────────────

def _read_preds_by_frame(csv_path: Path, only_firefly: bool=True, max_frames: Optional[int]=None) -> Dict[int, List[Tuple[int,int,int,int,str,str]]]:
    """
    Return: frame -> list of (x,y,w,h,cls,xy_semantics)
    """
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return {}
    has_class  = ('class' in rows[0].keys())
    has_xysem  = ('xy_semantics' in rows[0].keys())
    has_w      = ('w' in rows[0].keys())
    has_h      = ('h' in rows[0].keys())

    by_frame: Dict[int, List[Tuple[int,int,int,int,str,str]]] = defaultdict(list)
    for r in rows:
        try:
            f = int(r.get('frame', r.get('t')))
            if max_frames is not None and f >= max_frames:
                continue
            cls = (r.get('class','firefly') if has_class else 'firefly')
            if only_firefly and cls != 'firefly':
                continue
            x = float(r['x']); y = float(r['y'])
            w = int(float(r['w'])) if has_w else 10
            h = int(float(r['h'])) if has_h else 10
            xy_sem = (r.get('xy_semantics','') if has_xysem else '')
            by_frame[f].append((x,y,w,h,cls,xy_sem))
        except Exception:
            continue
    return by_frame

def _read_gt_by_frame(gt_norm_csv: Path, max_frames: Optional[int]=None) -> Dict[int, List[Tuple[int,int]]]:
    """
    gt_norm_csv must have columns x,y,t (t is zero-based frame index).
    Returns: frame -> list of (x,y) centers for GT.
    """
    rows = list(csv.DictReader(gt_norm_csv.open()))
    by_frame: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    for r in rows:
        try:
            f = int(r['t'])
            if max_frames is not None and f >= max_frames:
                continue
            x = int(round(float(r['x'])))
            y = int(round(float(r['y'])))
            by_frame[f].append((x,y))
        except Exception:
            continue
    return by_frame

def _find_normalized_gt(csv_dir: Path) -> Optional[Path]:
    """
    Try to locate a *_norm_offset*.csv written by Stage 9 in the same CSV folder.
    """
    try:
        cands = sorted(csv_dir.glob('*_norm_offset*.csv'))
        return cands[0] if cands else None
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────
# NEW: Per-threshold TP/FP/FN overlay helpers (Stage 9 outputs)
# ──────────────────────────────────────────────────────────────

def _sanitize_thr_name(thr_dir_name: str) -> str:
    # thr_5.0px -> thr5.0px
    return re.sub(r'[^0-9A-Za-z_.-]+', '', thr_dir_name.replace('thr_', 'thr'))

def _find_stage9_dir_for_base(base: str, hints: List[Path]) -> Optional[Path]:
    """
    Best-effort discovery of Stage 9 output folder for a given video 'base'.
    Looks through a few common locations, plus user hints.
    """
    candidates: List[Path] = []
    for h in hints:
        candidates.append(h / base)
    cwd = Path.cwd()
    candidates.extend([
        cwd / 'stage9 validation' / base,
        cwd / 'stage9_validation' / base,
        cwd / 'stage9' / base,
    ])
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None

def _read_thr_partitions(thr_dir: Path, max_frames: Optional[int]) -> Tuple[Dict[int,List[Tuple[int,int]]], Dict[int,List[Tuple[int,int]]], Dict[int,List[Tuple[int,int]]]]:
    """
    Reads fps.csv, tps.csv, fns.csv in a threshold folder.
    Returns dicts by frame:
      fps_by_t: frame -> list[(x,y)]
      tps_by_t: frame -> list[(x,y)]
      fns_by_t: frame -> list[(x,y)]
    """
    def _read_xy(path: Path) -> Dict[int, List[Tuple[int,int]]]:
        by_t: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
        if not path.exists():
            return by_t
        rows = list(csv.DictReader(path.open()))
        for r in rows:
            try:
                t = int(r.get('t'))
                if max_frames is not None and t >= max_frames:
                    continue
                x = int(round(float(r['x'])))
                y = int(round(float(r['y'])))
                by_t[t].append((x,y))
            except Exception:
                continue
        return by_t

    fps_by_t = _read_xy(thr_dir / 'fps.csv')
    tps_by_t = _read_xy(thr_dir / 'tps.csv')
    fns_by_t = _read_xy(thr_dir / 'fns.csv')
    return fps_by_t, tps_by_t, fns_by_t

def _list_threshold_dirs(stage9_dir: Path) -> List[Tuple[str, Path]]:
    # Expect subfolders like 'thr_5.0px'
    dirs = []
    for p in sorted(stage9_dir.iterdir()):
        if p.is_dir() and p.name.startswith('thr_') and p.name.endswith('px'):
            dirs.append((p.name, p))
    return dirs

def _render_tp_fp_fn_video_for_thr(
    *,
    orig_video_path: Path,
    thr_name: str,
    thr_dir: Path,
    out_dir: Path,
    box_w: int,
    box_h: int,
    thickness: int,
    max_frames: Optional[int],
):
    RED    = (0,0,255)
    GREEN  = (0,255,0)
    YELLOW = (0,255,255)

    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage10] Could not open video: {orig_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _ensure_dir(out_dir)

    base = orig_video_path.stem
    safe_thr = _sanitize_thr_name(thr_name)
    out_path = out_dir / f"{base}_tp_fp_fn_{safe_thr}.mp4"
    out  = cv2.VideoWriter(str(out_path), fourcc, fps, (W,H))

    # Read lists from fps.csv/tps.csv/fns.csv
    fps_by_t, tps_by_t, fns_by_t = _read_thr_partitions(thr_dir, max_frames)

    limit = total_frames if max_frames is None else min(total_frames, max_frames)
    fr = 0
    while True:
        if fr >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break

        # FP: predicted-only => RED
        for (x,y) in fps_by_t.get(fr, []):
            x0 = int(round(x - box_w/2.0))
            y0 = int(round(y - box_h/2.0))
            x0,y0,w,h = _clamp_box(x0,y0,box_w,box_h,W,H)
            cv2.rectangle(frame, (x0,y0), (x0+w, y0+h), RED, thickness)

        # FN: GT-only => GREEN
        for (x,y) in fns_by_t.get(fr, []):
            x0 = int(round(x - box_w/2.0))
            y0 = int(round(y - box_h/2.0))
            x0,y0,w,h = _clamp_box(x0,y0,box_w,box_h,W,H)
            cv2.rectangle(frame, (x0,y0), (x0+w, y0+h), GREEN, thickness)

        # TP: matched => YELLOW
        for (x,y) in tps_by_t.get(fr, []):
            x0 = int(round(x - box_w/2.0))
            y0 = int(round(y - box_h/2.0))
            x0,y0,w,h = _clamp_box(x0,y0,box_w,box_h,W,H)
            cv2.rectangle(frame, (x0,y0), (x0+w, y0+h), YELLOW, thickness)

        out.write(frame)
        _progress(fr+1, limit, f"stage10-threshold-overlay:{thr_name}"); fr += 1

    cap.release(); out.release()
    print(f"[stage10] Wrote per-threshold TP/FP/FN video: {out_path}")

# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def overlay_gt_vs_model(
    *,
    orig_video_path: Path,
    pred_csv_path: Path,
    out_video_path: Path,
    gt_norm_csv_path: Optional[Path] = None,   # if None we try to auto-find a *_norm_offset*.csv in pred_csv_path.parent
    thickness: int = 1,
    gt_box_w: int = 10, gt_box_h: int = 10,    # GT boxes are fixed-size around (x,y)
    only_firefly_rows: bool = True,
    max_frames: Optional[int] = None,
    # NEW — where to write per-threshold videos; defaults next to out_video_path in a subfolder
    stage9_dir_hint: Optional[Path] = None,
    render_threshold_overlays: bool = True,
    thr_box_w: Optional[int] = None,
    thr_box_h: Optional[int] = None,
):
    """
    Stage 10 does TWO things now:

    1) ORIGINAL OVERLAY (unchanged):
       Renders a single video drawing:
         - Ground Truth boxes in GREEN (fixed size around (x,y) center from normalized GT CSV)
         - Model output boxes in RED (uses (x,y,w,h) from preds; honors xy_semantics='center')
         - Pixels where the (outline) boxes overlap are drawn in YELLOW.

    2) NEW PER-THRESHOLD OVERLAYS:
       For each Stage 9 distance threshold folder (thr_*.px), renders a separate video
       where boxes are colored by classification:
         - TP = YELLOW
         - FP = RED
         - FN = GREEN
       These use the per-threshold fps.csv / tps.csv / fns.csv saved by Stage 9.
       Box sizes default to gt_box_w/h unless thr_box_w/h are provided.

    This stage expects that Stage 9 has run (for normalized GT and per-threshold CSVs).
    """
    # 1) Original overlay (GT vs Model)
    if gt_norm_csv_path is None:
        gt_norm_csv_path = _find_normalized_gt(pred_csv_path.parent)
        if gt_norm_csv_path is None:
            raise FileNotFoundError("Could not auto-locate normalized GT CSV (expected *_norm_offset*.csv). Run Stage 9 first or pass gt_norm_csv_path.")

    preds_by_frame = _read_preds_by_frame(pred_csv_path, only_firefly=only_firefly_rows, max_frames=max_frames)
    gt_by_frame    = _read_gt_by_frame(gt_norm_csv_path, max_frames=max_frames)

    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {orig_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _ensure_dir(out_video_path.parent)
    out  = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W,H))

    # Colors in BGR
    RED     = (0,0,255)
    GREEN   = (0,255,0)
    YELLOW  = (0,255,255)

    limit = total_frames
    if max_frames is not None:
        limit = min(limit, max_frames)

    fr = 0
    while True:
        if fr >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break

        red_layer   = np.zeros_like(frame)
        green_layer = np.zeros_like(frame)

        # Model output (preds) in RED
        for (x,y,w,h,cls,xy_sem) in preds_by_frame.get(fr, []):
            if isinstance(xy_sem, str) and xy_sem.lower() == 'center':
                x0 = int(round(float(x) - w/2.0))
                y0 = int(round(float(y) - h/2.0))
            else:
                x0 = int(round(float(x)))
                y0 = int(round(float(y)))
            x0,y0,w,h = _clamp_box(x0,y0,w,h,W,H)
            cv2.rectangle(red_layer, (x0,y0), (x0+w, y0+h), RED, thickness)

        # GT as fixed-size GREEN centered on (x,y)
        for (gx,gy) in gt_by_frame.get(fr, []):
            x0 = int(round(gx - gt_box_w/2.0))
            y0 = int(round(gy - gt_box_h/2.0))
            x0,y0,gw,gh = _clamp_box(x0,y0,gt_box_w,gt_box_h,W,H)
            cv2.rectangle(green_layer, (x0,y0), (x0+gw, y0+gh), GREEN, thickness)

        # Pixel-wise overlap between outlines
        red_mask   = np.any(red_layer > 0, axis=2)
        green_mask = np.any(green_layer > 0, axis=2)
        overlap_mask = red_mask & green_mask

        only_red   = red_mask & ~overlap_mask
        frame[only_red] = (0,0,255)
        only_green = green_mask & ~overlap_mask
        frame[only_green] = (0,255,0)
        frame[overlap_mask] = (0,255,255)

        out.write(frame)
        _progress(fr+1, limit, 'stage10-overlay'); fr += 1

    cap.release(); out.release()

    print("\n================ COLOR LEGEND (OVERLAY) ================")
    print("GROUND TRUTH: GREEN")
    print("MODEL OUTPUT: RED")
    print("OVERLAP (TP): YELLOW")
    print("=======================================================\n")

    # 2) Per-threshold TP/FP/FN overlays (auto, unless disabled)
    if render_threshold_overlays:
        base = orig_video_path.stem
        # Try to locate the Stage 9 directory for this video
        stage9_dir = None
        if stage9_dir_hint is not None and stage9_dir_hint.exists():
            # Allow both the parent folder and the per-video child
            if stage9_dir_hint.name == base and stage9_dir_hint.is_dir():
                stage9_dir = stage9_dir_hint
            else:
                cand = stage9_dir_hint / base
                if cand.exists():
                    stage9_dir = cand
        if stage9_dir is None:
            # Heuristic search in common locations
            # - next to the CSV folder
            candidates = []
            csv_parent = pred_csv_path.parent
            candidates.append(csv_parent.parent / 'stage9 validation')
            candidates.append(Path.cwd() / 'stage9 validation')
            stage9_dir = _find_stage9_dir_for_base(base, candidates)

        if stage9_dir is None:
            print("[stage10] WARNING: Could not find Stage 9 per-threshold outputs; skipping TP/FP/FN videos.")
            return

        thr_dirs = _list_threshold_dirs(stage9_dir)
        if not thr_dirs:
            print(f"[stage10] WARNING: No threshold folders found in: {stage9_dir} — skipping TP/FP/FN videos.")
            return

        # Output folder for these videos (next to out_video_path, inside a subfolder)
        per_thr_out_dir = out_video_path.parent / f"{base}_by_threshold"
        _ensure_dir(per_thr_out_dir)

        bw = int(thr_box_w if thr_box_w is not None else gt_box_w)
        bh = int(thr_box_h if thr_box_h is not None else gt_box_h)

        print(f"[stage10] Rendering TP/FP/FN videos for {len(thr_dirs)} thresholds from: {stage9_dir}")
        for thr_name, thr_path in thr_dirs:
            _render_tp_fp_fn_video_for_thr(
                orig_video_path=orig_video_path,
                thr_name=thr_name,
                thr_dir=thr_path,
                out_dir=per_thr_out_dir,
                box_w=bw,
                box_h=bh,
                thickness=thickness,
                max_frames=max_frames,
            )

        print("\n============= COLOR LEGEND (TP/FP/FN VIDEOS) =============")
        print("TRUE POSITIVE (TP): YELLOW")
        print("FALSE POSITIVE (FP): RED")
        print("FALSE NEGATIVE (FN): GREEN")
        print("==========================================================")
