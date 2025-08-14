#!/usr/bin/env python3
"""
metrics_and_crop_fps.py
───────────────────────
Evaluate point detections (x,y,t) against ground truth across multiple distance
thresholds, and save FP, TP, and FN crops per threshold **with model confidence**.

Inputs:
  • Ground truth CSV: x,y,t
  • Predictions CSV:  x,y,t,background_logit,firefly_logit
  • Frames folder: frame_00000.png (or .jpg), frame_00001.png, ...

Matching (independent per threshold):
  • Per frame (t equal)
  • Greedy one-to-one, smallest Euclidean distance first
  • TP if distance <= threshold; unmatched preds → FP; unmatched GT → FN

Outputs (per threshold):
  • Printed metrics: TP, FP, FN, Precision, Recall, F1, MeanErr
  • crops/   + fps.csv:  x,y,t,crop_path,firefly_conf
  • tp_crops/+ tps.csv:  x,y,t,crop_path,firefly_conf
  • fn_crops/+ fns.csv:  x,y,t,crop_path

NOTE: This script assumes frame image filenames are offset by a constant amount
      relative to CSV t. We apply: filename_index = t + FRAME_NAME_OFFSET.
"""

from pathlib import Path
import csv
import math
from collections import defaultdict
import cv2
import os

# ─── GLOBALS ───────────────────────────────────────────────────

GT_CSV       = Path('/Users/arnavps/Desktop/to send/tremulans ground truth/tremulans ground truth centroids.csv')
PRED_CSV     = Path('/Users/arnavps/Desktop/tremulans and forresti seperate models inference data/tremulans inference data/csv files/tremulans_model_xyt_val_4k-5k.csv')

# Frames live here: frame_00000.png, frame_00001.png, ...
FRAMES_DIR   = Path('/Users/arnavps/Desktop/to annotate frames/tremulans')

# Output root: per-threshold subfolders will be created here
OUT_DIR      = Path('forresti, fixing FPs and box overlap/Proof of concept code/test1/bug fixing forresti/bug fixing data/mini val script output on tremulans only model OP')

# Evaluate these distance thresholds (px). Order preserved.
DIST_THRESHOLDS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,8.0,9.0,10.0]

# Crop settings for thumbnails (centered on (x,y))
CROP_W = 10
CROP_H = 10

# Frame filename formatting
FRAME_PREFIX = 'frame_'
FRAME_ZPAD   = 5
FRAME_EXT    = '.png'   # we’ll try this, then fall back to .jpg automatically

# If True, also print per-frame TP/FP/FN for each threshold
SHOW_PER_FRAME = False

# NEW: file-name index offset. Filenames use (t + FRAME_NAME_OFFSET).
#      This implements: "take the frame name number and subtract 3999 to get t".
FRAME_NAME_OFFSET = 1


# ─── helpers ───────────────────────────────────────────────────

def _read_points_gt(path: Path):
    """Return dict: t -> list of (x,y)."""
    by_t = defaultdict(list)
    with path.open('r', newline='') as f:
        r = csv.DictReader(f)
        need = {'x','y','t'}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"{path} must have columns x,y,t")
        for row in r:
            try:
                x = int(round(float(row['x'])))
                y = int(round(float(row['y'])))
                t = int(row['t'])
            except Exception:
                continue
            by_t[t].append((x,y))
    return by_t

def _read_points_pred(path: Path):
    """
    Return dict: t -> list of dicts with keys:
      {'x':int,'y':int,'b':float,'f':float}
    """
    by_t = defaultdict(list)
    with path.open('r', newline='') as f:
        r = csv.DictReader(f)
        need = {'x','y','t','background_logit','firefly_logit'}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"{path} must have columns x,y,t,background_logit,firefly_logit")
        for row in r:
            try:
                x = int(round(float(row['x'])))
                y = int(round(float(row['y'])))
                t = int(row['t'])
                b = float(row['background_logit'])
                f = float(row['firefly_logit'])
            except Exception:
                continue
            by_t[t].append({'x':x,'y':y,'b':b,'f':f})
    return by_t

def _softmax_conf_firefly(b, f):
    # numerically stable softmax for 2 logits
    m = max(b, f)
    eb = math.exp(b - m)
    ef = math.exp(f - m)
    denom = eb + ef
    return 0.0 if denom == 0 else (ef / denom)

def _pairwise_dist2(a, b):
    dx = a[0]-b[0]; dy = a[1]-b[1]
    return dx*dx + dy*dy

def _greedy_match_full(frame_gts, frame_preds_xy, max_dist_px):
    """
    Greedy 1-1 matching in a frame.
    Inputs:
      frame_gts:      list[(x,y)]
      frame_preds_xy: list[(x,y)]
    Returns:
      matches: list of (gi, pi, dist)
      unmatched_pred_indices: list of pi
      unmatched_gt_indices: list of gi
    """
    nG = len(frame_gts); nP = len(frame_preds_xy)
    if nG == 0 and nP == 0:
        return [], [], []

    max_d2 = max_dist_px * max_dist_px
    used_gt = [False]*nG
    used_pr = [False]*nP

    pairs = []
    for gi, g in enumerate(frame_gts):
        for pi, p in enumerate(frame_preds_xy):
            d2 = _pairwise_dist2(g, p)
            if d2 <= max_d2:
                pairs.append((d2, gi, pi))
    pairs.sort(key=lambda x: x[0])

    matches = []
    for d2, gi, pi in pairs:
        if not used_gt[gi] and not used_pr[pi]:
            used_gt[gi] = True
            used_pr[pi] = True
            matches.append((gi, pi, math.sqrt(d2)))

    unmatched_pred_indices = [i for i, u in enumerate(used_pr) if not u]
    unmatched_gt_indices   = [i for i, u in enumerate(used_gt) if not u]
    return matches, unmatched_pred_indices, unmatched_gt_indices

def _frame_path_candidates(t: int):
    """
    Yield candidate image paths for frame index t.
    Apply filename offset: file_index = t + FRAME_NAME_OFFSET.
    """
    file_index = t + FRAME_NAME_OFFSET
    base = f"{FRAME_PREFIX}{file_index:0{FRAME_ZPAD}d}"
    yield FRAMES_DIR / f"{base}{FRAME_EXT}"
    if FRAME_EXT.lower() != '.jpg':
        yield FRAMES_DIR / f"{base}.jpg"

def _load_frame(t: int):
    """Return (img, path_str) or (None, None) if not found."""
    for p in _frame_path_candidates(t):
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img, str(p)
    return None, None

def _safe_crop(img, cx, cy, w, h):
    """
    Center crop around (cx,cy) with size (w,h). Clamps to image bounds.
    Returns cropped image and top-left (x0,y0) used.
    """
    H, W = img.shape[:2]
    x0 = int(round(cx - w/2))
    y0 = int(round(cy - h/2))
    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0
    if x0 + w > W: x0 = max(0, W - w)
    if y0 + h > H: y0 = max(0, H - h)
    x1 = x0 + w
    y1 = y0 + h
    x1 = min(x1, W); y1 = min(y1, H)
    x0 = max(0, x1 - w); y0 = max(0, y1 - h)
    return img[y0:y1, x0:x1], x0, y0

def _thr_folder_name(thr: float) -> str:
    s = f"{thr}".replace('.', 'p')
    return f"thr_{s}px"

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ─── evaluation + crops per threshold ──────────────────────────

def _evaluate_and_crop(gt_by_t, pr_by_t, threshold_px, out_root: Path, show_per_frame=False):
    thr_dir = out_root / _thr_folder_name(threshold_px)
    _ensure_dir(thr_dir)

    # Existing FP outputs (unchanged)
    crops_dir_fp = thr_dir / "crops"
    _ensure_dir(crops_dir_fp)
    csv_fp_path = thr_dir / "fps.csv"

    # NEW: TP + FN outputs
    crops_dir_tp = thr_dir / "tp_crops"
    crops_dir_fn = thr_dir / "fn_crops"
    _ensure_dir(crops_dir_tp)
    _ensure_dir(crops_dir_fn)
    csv_tp_path = thr_dir / "tps.csv"
    csv_fn_path = thr_dir / "fns.csv"

    all_frames = sorted(set(gt_by_t.keys()) | set(pr_by_t.keys()))
    total_tp = total_fp = total_fn = 0
    all_dists = []

    with csv_fp_path.open('w', newline='') as fcsv_fp, \
         csv_tp_path.open('w', newline='') as fcsv_tp, \
         csv_fn_path.open('w', newline='') as fcsv_fn:

        w_fp = csv.writer(fcsv_fp); w_fp.writerow(['x','y','t','crop_path','firefly_conf'])
        w_tp = csv.writer(fcsv_tp); w_tp.writerow(['x','y','t','crop_path','firefly_conf'])
        w_fn = csv.writer(fcsv_fn); w_fn.writerow(['x','y','t','crop_path'])

        for t in all_frames:
            gts = gt_by_t.get(t, [])
            preds = pr_by_t.get(t, [])  # list of dicts {'x','y','b','f'}
            preds_xy = [(p['x'], p['y']) for p in preds]

            # Full greedy matching to get matches + unmatched for both sides
            matches, unmatched_pred_idxs, unmatched_gt_idxs = _greedy_match_full(gts, preds_xy, threshold_px)

            tp = len(matches)
            fp = len(unmatched_pred_idxs)
            fn = len(unmatched_gt_idxs)
            total_tp += tp; total_fp += fp; total_fn += fn
            all_dists.extend([d for (_,_,d) in matches])

            if show_per_frame:
                print(f"[thr={threshold_px:>4.1f}] t={t:6d}  GT={len(gts):3d}  PRED={len(preds):3d}  "
                      f"TP={tp:3d}  FP={fp:3d}  FN={fn:3d}")

            # Load frame once if we need any crops
            need_any = bool(unmatched_pred_idxs or unmatched_gt_idxs or matches)
            if not need_any:
                continue

            img, frame_used_path = _load_frame(t)

            # Save FPs (unchanged; if frame missing, still log row)
            if unmatched_pred_idxs:
                for k, pi in enumerate(unmatched_pred_idxs):
                    p = preds[pi]
                    x, y = p['x'], p['y']
                    conf = _softmax_conf_firefly(p['b'], p['f'])
                    crop_path_str = "MISSING_FRAME"
                    if img is not None:
                        crop, _, _ = _safe_crop(img, x, y, CROP_W, CROP_H)
                        fname = f"t{t:0{FRAME_ZPAD}d}_x{x}_y{y}_thr{str(threshold_px).replace('.','p')}_{k:03d}.png"
                        out_path = crops_dir_fp / fname
                        cv2.imwrite(str(out_path), crop)
                        crop_path_str = str(out_path)
                    w_fp.writerow([x, y, t, crop_path_str, f"{conf:.6f}"])

            # Save TPs (centered on prediction; with confidence)
            if matches:
                for k, (gi, pi, dist) in enumerate(matches):
                    p = preds[pi]
                    x, y = p['x'], p['y']
                    conf = _softmax_conf_firefly(p['b'], p['f'])
                    crop_path_str = "MISSING_FRAME"
                    if img is not None:
                        crop, _, _ = _safe_crop(img, x, y, CROP_W, CROP_H)
                        fname = f"TP_t{t:0{FRAME_ZPAD}d}_x{x}_y{y}_thr{str(threshold_px).replace('.','p')}_{k:03d}.png"
                        out_path = crops_dir_tp / fname
                        cv2.imwrite(str(out_path), crop)
                        crop_path_str = str(out_path)
                    w_tp.writerow([x, y, t, crop_path_str, f"{conf:.6f}"])

            # Save FNs (centered on ground-truth; no confidence)
            if unmatched_gt_idxs:
                for k, gi in enumerate(unmatched_gt_idxs):
                    gx, gy = gts[gi]
                    crop_path_str = "MISSING_FRAME"
                    if img is not None:
                        crop, _, _ = _safe_crop(img, gx, gy, CROP_W, CROP_H)
                        fname = f"FN_t{t:0{FRAME_ZPAD}d}_x{gx}_y{gy}_thr{str(threshold_px).replace('.','p')}_{k:03d}.png"
                        out_path = crops_dir_fn / fname
                        cv2.imwrite(str(out_path), crop)
                        crop_path_str = str(out_path)
                    w_fn.writerow([gx, gy, t, crop_path_str])

    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall    = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2*precision*recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_err  = (sum(all_dists)/len(all_dists)) if all_dists else 0.0

    print(f"{threshold_px:8.1f}  {total_tp:6d} {total_fp:6d} {total_fn:6d}   "
          f"{precision:7.4f} {recall:7.4f} {f1:7.4f}   {mean_err:12.3f}   -> {thr_dir}")


# ─── main ─────────────────────────────────────────────────────-

def main():
    gt_by_t = _read_points_gt(GT_CSV)
    pr_by_t = _read_points_pred(PRED_CSV)

    _ensure_dir(OUT_DIR)

    print("=== Detection Metrics (point, same-frame, distance-sweep) + FP/TP/FN crops ===")
    print(f"GT:   {GT_CSV}")
    print(f"PRED: {PRED_CSV}")
    print(f"Frames dir: {FRAMES_DIR}\n")
    print(f"(Using filename index offset: file_index = t + {FRAME_NAME_OFFSET})\n")

    print(f"{'thr(px)':>8}  {'TP':>6} {'FP':>6} {'FN':>6}   {'Prec':>7} {'Rec':>7} {'F1':>7}   {'MeanErr(px)':>12}   Saved-To")
    print("-"*100)

    for thr in DIST_THRESHOLDS:
        _evaluate_and_crop(gt_by_t, pr_by_t, float(thr), OUT_DIR, show_per_frame=SHOW_PER_FRAME)

if __name__ == '__main__':
    main()
