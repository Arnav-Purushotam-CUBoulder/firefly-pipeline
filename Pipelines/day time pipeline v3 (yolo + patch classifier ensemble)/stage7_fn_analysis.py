#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
# Tiny progress bar (consistent with other stages)
_BAR = 50
def _progress(i: int, total: int, tag: str = ''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar  = '=' * int(frac * _BAR) + ' ' * (_BAR - int(frac * _BAR))
    print(f'\r{tag} [{bar}] {int(frac*100):3d}%', end='' if i < total else '\n')

# ──────────────────────────────────────────────────────────────
def _read_points_by_frame(csv_path: Path) -> Dict[int, List[Tuple[float, float]]]:
    """
    Read a tps.csv/fns.csv/fps.csv into: t -> list[(x,y)].
    Expected headers: ['x','y','t', ...]
    """
    by_t: Dict[int, List[Tuple[float, float]]] = {}
    if not csv_path.exists():
        return by_t
    with csv_path.open('r', newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                t = int(float(row.get('t', row.get('frame', '0'))))
                x = float(row['x']); y = float(row['y'])
            except Exception:
                continue
            by_t.setdefault(t, []).append((x, y))
    return by_t

def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)

def _draw_centered_box(img: np.ndarray, cx: float, cy: float, w: int, h: int,
                       color: Tuple[int,int,int], thickness: int = 1):
    H, W = img.shape[:2]
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x1 = x0 + int(w) - 1;        y1 = y0 + int(h) - 1
    x0 = max(0, min(x0, W-1)); y0 = max(0, min(y0, H-1))
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)

# ──────────────────────────────────────────────────────────────
def _analyze_threshold_dir_and_render(
    *,
    thr_dir: Path,
    output_thr_dir: Optional[Path] = None,
    orig_video_path: Path,
    box_w: int,
    box_h: int,
    color: Tuple[int,int,int],
    thickness: int,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    For a single thr_* folder, compute nearest-TP distance for every FN in fns.csv,
    write thr_dir/'fn_nearest_tp.csv', and render a full-frame image per FN with
    both the FN and its nearest TP highlighted in the same color.

    NEW: also render FN vs MODEL overlay frames:
      - All model predictions (TP + FP) in RED
      - The FN (missed GT) in GREEN
      - Overlap pixels (between outlines) in YELLOW
      Saved under thr_dir/'fn_vs_pred_frames/' with a legend and nearest-PRED info.
    """
    fns_csv = thr_dir / 'fns.csv'
    tps_csv = thr_dir / 'tps.csv'
    fps_csv = thr_dir / 'fps.csv'   # NEW: to collect all predictions for overlays
    output_thr_dir = Path(output_thr_dir) if output_thr_dir is not None else thr_dir
    output_thr_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_thr_dir / 'fn_nearest_tp.csv'
    out_img_dir = output_thr_dir / 'fn_pair_frames'
    out_img_dir.mkdir(parents=True, exist_ok=True)

    # NEW: extra folder for FN vs MODEL (TP+FP) overlays
    out_img_dir_fn_vs_pred = output_thr_dir / 'fn_vs_pred_frames'
    out_img_dir_fn_vs_pred.mkdir(parents=True, exist_ok=True)

    fns_by_t = _read_points_by_frame(fns_csv)
    tps_by_t = _read_points_by_frame(tps_csv)
    fps_by_t = _read_points_by_frame(fps_csv) if fps_csv.exists() else {}

    # For overlays: predictions in a frame are TP (pred side) + FP
    preds_by_t: Dict[int, List[Tuple[float, float]]] = {}
    all_ts = set(tps_by_t.keys()) | set(fps_by_t.keys())
    for t in all_ts:
        preds_by_t[t] = (tps_by_t.get(t, []) or []) + (fps_by_t.get(t, []) or [])

    frames = sorted(set(fns_by_t.keys()))
    total_fns = sum(len(v) for v in fns_by_t.values())
    no_tp_cnt = 0

    # Precompute nearest TP for each FN (kept as-is for CSV compatibility)
    per_t_pairs: Dict[int, List[Dict]] = {}
    for t in frames:
        fns = fns_by_t.get(t, [])
        tps = tps_by_t.get(t, [])
        per_t_pairs[t] = []
        for (fx, fy) in fns:
            if not tps:
                per_t_pairs[t].append({
                    't': t, 'fn_x': fx, 'fn_y': fy,
                    'tp_x': None, 'tp_y': None, 'dist': None,
                    'image_path': ''
                })
                no_tp_cnt += 1
            else:
                # Nearest TP by Euclidean distance
                min_d = None
                min_tp = None
                for (tx, ty) in tps:
                    d = _euclid((fx, fy), (tx, ty))
                    if (min_d is None) or (d < min_d):
                        min_d = d
                        min_tp = (tx, ty)
                per_t_pairs[t].append({
                    't': t, 'fn_x': fx, 'fn_y': fy,
                    'tp_x': float(min_tp[0]), 'tp_y': float(min_tp[1]),
                    'dist': float(min_d), 'image_path': ''
                })

    # Render full-frame images and write CSV
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage7] Could not open video: {orig_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    done = 0
    with out_csv.open('w', newline='') as f_out:
        wri = csv.writer(f_out)
        wri.writerow(['t','fn_x','fn_y','nearest_tp_x','nearest_tp_y','distance_px','image_path'])

        fr = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if fr in per_t_pairs and per_t_pairs[fr]:
                for rec in per_t_pairs[fr]:
                    img = frame.copy()
                    # Draw FN
                    _draw_centered_box(img, rec['fn_x'], rec['fn_y'], box_w, box_h, color, thickness)
                    # Draw nearest TP if available
                    if rec['tp_x'] is not None and rec['tp_y'] is not None:
                        _draw_centered_box(img, rec['tp_x'], rec['tp_y'], box_w, box_h, color, thickness)

                    # Save the "pair" image (original behavior)
                    fnx = int(round(rec['fn_x'])); fny = int(round(rec['fn_y']))
                    if rec['tp_x'] is None:
                        img_name = f"t{fr:06d}_fnx{fnx}_fny{fny}_tpNA_dNA.png"
                    else:
                        tpx = int(round(rec['tp_x'])); tpy = int(round(rec['tp_y']))
                        dstr = f"{rec['dist']:.6f}"
                        img_name = f"t{fr:06d}_fnx{fnx}_fny{fny}_tpx{tpx}_tpy{tpy}_d{dstr}.png"

                    out_path = out_img_dir / img_name
                    cv2.imwrite(str(out_path), img)
                    rec['image_path'] = str(out_path)

                    # ── NEW: FN vs MODEL overlay (RED = model preds, GREEN = this FN, YELLOW = overlap)
                    red_layer   = np.zeros_like(frame)
                    green_layer = np.zeros_like(frame)

                    # Draw all predictions (TP + FP) in RED
                    pred_pts = preds_by_t.get(fr, [])
                    for (px, py) in pred_pts:
                        _draw_centered_box(red_layer, px, py, box_w, box_h, (0,0,255), thickness)

                    # Draw this FN in GREEN
                    _draw_centered_box(green_layer, rec['fn_x'], rec['fn_y'], box_w, box_h, (0,255,0), thickness)

                    # Compose outlines → red/green/yellow
                    composed = frame.copy()
                    red_mask   = np.any(red_layer > 0, axis=2)
                    green_mask = np.any(green_layer > 0, axis=2)
                    overlap    = red_mask & green_mask
                    only_red   = red_mask & ~overlap
                    only_green = green_mask & ~overlap
                    composed[only_red]   = (0,0,255)
                    composed[only_green] = (0,255,0)
                    composed[overlap]    = (0,255,255)

                    # Nearest predicted point (from TP+FP set)
                    if pred_pts:
                        min_d_pred = None
                        min_pred   = None
                        for (px, py) in pred_pts:
                            d = _euclid((rec['fn_x'], rec['fn_y']), (px, py))
                            if (min_d_pred is None) or (d < min_d_pred):
                                min_d_pred = d
                                min_pred   = (px, py)
                        npx, npy = int(round(min_pred[0])), int(round(min_pred[1]))
                        dstr_pred = f"{min_d_pred:.6f}"
                        legend = "LEGEND_PRED=RED_FN=GREEN_OVERLAP=YELLOW"
                        img_name_pred = (
                            f"t{fr:06d}_{legend}_"
                            f"FN({fnx},{fny})_nearestPRED({npx},{npy})_d{dstr_pred}.png"
                        )
                    else:
                        legend = "LEGEND_PRED=RED_FN=GREEN_OVERLAP=YELLOW"
                        img_name_pred = (
                            f"t{fr:06d}_{legend}_"
                            f"FN({fnx},{fny})_nearestPRED(NA,NA)_dNA.png"
                        )

                    out_path_pred = out_img_dir_fn_vs_pred / img_name_pred
                    cv2.imwrite(str(out_path_pred), composed)

                    # Write CSV row (unchanged)
                    wri.writerow([
                        rec['t'],
                        f"{rec['fn_x']:.3f}", f"{rec['fn_y']:.3f}",
                        ("" if rec['tp_x'] is None else f"{rec['tp_x']:.3f}"),
                        ("" if rec['tp_y'] is None else f"{rec['tp_y']:.3f}"),
                        ("" if rec['dist'] is None else f"{rec['dist']:.6f}"),
                        rec['image_path'],
                    ])

                    _progress(done + 1, total_fns, f"stage7@{thr_dir.name}")
                    done += 1

            fr += 1
            if fr >= total_frames:
                break

    cap.release()
    if verbose:
        print(f"[stage7] {thr_dir.name}: analyzed {total_fns} FNs  (no-TP frames: {no_tp_cnt})  "
              f"→ {out_csv.name} and {out_img_dir.name}/, plus {out_img_dir_fn_vs_pred.name}/ (FN vs PRED overlays)")

    return total_fns, no_tp_cnt

# ──────────────────────────────────────────────────────────────
def stage7_fn_nearest_tp_analysis(
    *,
    stage5_video_dir: Path,
    output_dir: Optional[Path] = None,
    orig_video_path: Path,
    box_w: int = 10,
    box_h: int = 10,
    color: Tuple[int,int,int] = (0, 255, 255),  # BGR (yellow by default)
    thickness: int = 1,
    verbose: bool = True,
) -> None:
    """
    Iterate all thr_* directories in `stage5_video_dir`, and for each:
      - Load fns.csv and tps.csv
      - Create fn_nearest_tp.csv with nearest-TP distances for every FN
      - Save full-frame images per FN in thr_*/fn_pair_frames/, drawing FN and nearest TP.
      - NEW: Save full-frame images per FN in thr_*/fn_vs_pred_frames/ with
             predictions (TP+FP) in red, FN in green, and yellow where outlines overlap.
    """
    if not stage5_video_dir.exists():
        if verbose:
            print(f"[stage7] Stage-5 dir not found: {stage5_video_dir}")
        return

    thr_dirs = sorted([p for p in stage5_video_dir.iterdir() if p.is_dir() and p.name.startswith('thr_')])
    if not thr_dirs:
        if verbose:
            print(f"[stage7] No thr_* folders in: {stage5_video_dir}")
        return

    if verbose:
        print(f"[stage7] Found {len(thr_dirs)} threshold folders under: {stage5_video_dir}")

    out_root = Path(output_dir) if output_dir is not None else stage5_video_dir
    out_root.mkdir(parents=True, exist_ok=True)

    total_fns_all = 0
    total_no_tp = 0
    for td in thr_dirs:
        fns_csv = td / 'fns.csv'
        tps_csv = td / 'tps.csv'
        if not (fns_csv.exists() and tps_csv.exists()):
            if verbose:
                print(f"[stage7] Skipping {td.name} (missing fns.csv or tps.csv)")
            continue
        total_fns, no_tp = _analyze_threshold_dir_and_render(
            thr_dir=td,
            output_thr_dir=(out_root / td.name),
            orig_video_path=orig_video_path,
            box_w=box_w,
            box_h=box_h,
            color=color,
            thickness=thickness,
            verbose=verbose
        )
        total_fns_all += total_fns
        total_no_tp   += no_tp

    if verbose:
        print(f"[stage7] Done. Total FNs analyzed: {total_fns_all}.  "
              f"FNs without any TP in the same frame: {total_no_tp}.")


# Backwards-compatible alias (older numbering)
stage8_fn_nearest_tp_analysis = stage7_fn_nearest_tp_analysis
