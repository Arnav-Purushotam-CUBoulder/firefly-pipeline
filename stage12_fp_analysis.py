#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
# Tiny progress bar (same style as other stages)
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
    Read a tps.csv/fps.csv into: t -> list[(x,y)].
    Expected headers: ['x','y','t','filepath','confidence']
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
    orig_video_path: Path,
    box_w: int,
    box_h: int,
    color: Tuple[int,int,int],
    thickness: int,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    For a single thr_* folder, compute nearest-TP distance for every FP in fps.csv,
    write thr_dir/'fp_nearest_tp.csv', and render a full-frame image per FP with
    both the FP and its nearest TP highlighted in the same color.
    Returns: (num_fps, num_without_tp_in_frame)
    """
    fps_csv = thr_dir / 'fps.csv'
    tps_csv = thr_dir / 'tps.csv'
    out_csv = thr_dir / 'fp_nearest_tp.csv'
    out_img_dir = thr_dir / 'fp_pair_frames'
    out_img_dir.mkdir(parents=True, exist_ok=True)

    fps_by_t = _read_points_by_frame(fps_csv)
    tps_by_t = _read_points_by_frame(tps_csv)

    frames = sorted(set(fps_by_t.keys()))
    total_fps = sum(len(v) for v in fps_by_t.values())
    no_tp_cnt = 0

    # Precompute nearest TP for each FP
    per_t_pairs: Dict[int, List[Dict]] = {}
    for t in frames:
        fps = fps_by_t.get(t, [])
        tps = tps_by_t.get(t, [])
        per_t_pairs[t] = []
        for (fx, fy) in fps:
            if not tps:
                per_t_pairs[t].append({
                    't': t, 'fp_x': fx, 'fp_y': fy,
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
                    't': t, 'fp_x': fx, 'fp_y': fy,
                    'tp_x': float(min_tp[0]), 'tp_y': float(min_tp[1]),
                    'dist': float(min_d), 'image_path': ''
                })

    # Render full-frame images and write CSV
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage12] Could not open video: {orig_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    done = 0
    with out_csv.open('w', newline='') as f_out:
        wri = csv.writer(f_out)
        wri.writerow(['t','fp_x','fp_y','nearest_tp_x','nearest_tp_y','distance_px','image_path'])

        fr = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if fr in per_t_pairs and per_t_pairs[fr]:
                for rec in per_t_pairs[fr]:
                    img = frame.copy()
                    # Draw FP
                    _draw_centered_box(img, rec['fp_x'], rec['fp_y'], box_w, box_h, color, thickness)
                    # Draw nearest TP if available
                    if rec['tp_x'] is not None and rec['tp_y'] is not None:
                        _draw_centered_box(img, rec['tp_x'], rec['tp_y'], box_w, box_h, color, thickness)

                    # Save image
                    fpx = int(round(rec['fp_x'])); fpy = int(round(rec['fp_y']))
                    if rec['tp_x'] is None:
                        img_name = f"t{fr:06d}_fpx{fpx}_fpy{fpy}_tpNA_dNA.png"
                    else:
                        tpx = int(round(rec['tp_x'])); tpy = int(round(rec['tp_y']))
                        dstr = f"{rec['dist']:.6f}"
                        img_name = f"t{fr:06d}_fpx{fpx}_fpy{fpy}_tpx{tpx}_tpy{tpy}_d{dstr}.png"

                    out_path = out_img_dir / img_name
                    cv2.imwrite(str(out_path), img)
                    rec['image_path'] = str(out_path)

                    # Write CSV row
                    wri.writerow([
                        rec['t'],
                        f"{rec['fp_x']:.3f}", f"{rec['fp_y']:.3f}",
                        ("" if rec['tp_x'] is None else f"{rec['tp_x']:.3f}"),
                        ("" if rec['tp_y'] is None else f"{rec['tp_y']:.3f}"),
                        ("" if rec['dist'] is None else f"{rec['dist']:.6f}"),
                        rec['image_path'],
                    ])

                    done += 1
                    _progress(done, total_fps, f"stage12@{thr_dir.name}")

            fr += 1
            if fr >= total_frames:
                break

    cap.release()
    if verbose:
        print(f"[stage12] {thr_dir.name}: analyzed {total_fps} FPs  (no-TP frames: {no_tp_cnt})  "
              f"→ {out_csv.name} and {out_img_dir.name}/")

    return total_fps, no_tp_cnt

# ──────────────────────────────────────────────────────────────
def stage12_fp_nearest_tp_analysis(
    *,
    stage9_video_dir: Path,
    orig_video_path: Path,
    box_w: int = 10,
    box_h: int = 10,
    color: Tuple[int,int,int] = (255, 0, 255),  # BGR (magenta default)
    thickness: int = 1,
    verbose: bool = True,
) -> None:
    """
    Iterate all thr_* directories in `stage9_video_dir`, and for each:
      - Load fps.csv and tps.csv
      - Create fp_nearest_tp.csv with nearest-TP distances for every FP
      - Save full-frame images per FP in thr_*/fp_pair_frames/, drawing FP and nearest TP.

    Parameters
    ----------
    stage9_video_dir : Path
        Folder with Stage-9 outputs for a single video (contains thr_* subdirs).
    orig_video_path : Path
        Path to the original video to grab frames from.
    box_w, box_h : int
        Size (in pixels) of the boxes to draw, centered at the FP/TP coordinates.
    color : (B, G, R)
        Color used for both FP and nearest TP boxes (same color).
    thickness : int
        Rectangle border thickness.
    """
    if not stage9_video_dir.exists():
        if verbose:
            print(f"[stage12] Stage-9 dir not found: {stage9_video_dir}")
        return

    thr_dirs = sorted([p for p in stage9_video_dir.iterdir() if p.is_dir() and p.name.startswith('thr_')])
    if not thr_dirs:
        if verbose:
            print(f"[stage12] No thr_* folders in: {stage9_video_dir}")
        return

    if verbose:
        print(f"[stage12] Found {len(thr_dirs)} threshold folders under: {stage9_video_dir}")

    total_fps_all = 0
    total_no_tp = 0
    for td in thr_dirs:
        fps_csv = td / 'fps.csv'
        tps_csv = td / 'tps.csv'
        if not (fps_csv.exists() and tps_csv.exists()):
            if verbose:
                print(f"[stage12] Skipping {td.name} (missing fps.csv or tps.csv)")
            continue
        total_fps, no_tp = _analyze_threshold_dir_and_render(
            thr_dir=td,
            orig_video_path=orig_video_path,
            box_w=box_w,
            box_h=box_h,
            color=color,
            thickness=thickness,
            verbose=verbose
        )
        total_fps_all += total_fps
        total_no_tp   += no_tp

    if verbose:
        print(f"[stage12] Done. Total FPs analyzed: {total_fps_all}.  "
              f"FPs without any TP in the same frame: {total_no_tp}.")
