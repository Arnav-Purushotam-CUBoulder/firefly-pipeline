#!/usr/bin/env python3
import csv, sys, math, os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# Optional: used only for FN confidence inference
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision import models
    from PIL import Image
except Exception:
    torch = None
    nn = None
    T = None
    models = None
    Image = None

# ───────────────────────── helpers ─────────────────────────

_BAR = 50
def _progress(i, total, tag=''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar  = '=' * int(frac * _BAR) + ' ' * (_BAR - int(frac * _BAR))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i >= total: sys.stdout.write('\n')

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _softmax_conf_firefly(b: float, f: float) -> float:
    m = max(b, f)
    eb = math.exp(b - m)
    ef = math.exp(f - m)
    denom = eb + ef
    return ef / denom if denom > 0 else 0.5

def _pairwise_dist2(a, b):
    dx = a[0]-b[0]; dy = a[1]-b[1]
    return dx*dx + dy*dy

def _greedy_match_full(frame_gts, frame_preds_xy, max_dist_px):
    """
    Greedy 1-1 frame matching.
    Returns: matches[(gi,pi,dist)], unmatched_pred_idxs, unmatched_gt_idxs
    """
    nG = len(frame_gts); nP = len(frame_preds_xy)
    if nG == 0 and nP == 0:
        return [], [], []
    max_d2 = max_dist_px * max_dist_px
    used_g = [False]*nG; used_p = [False]*nP
    pairs = []
    for gi, g in enumerate(frame_gts):
        for pi, p in enumerate(frame_preds_xy):
            d2 = _pairwise_dist2(g, p)
            if d2 <= max_d2:
                pairs.append((d2, gi, pi))
    pairs.sort(key=lambda x: x[0])
    matches = []
    for d2, gi, pi in pairs:
        if not used_g[gi] and not used_p[pi]:
            used_g[gi] = True; used_p[pi] = True
            matches.append((gi, pi, math.sqrt(d2)))
    unmatched_pred = [i for i, u in enumerate(used_p) if not u]
    unmatched_gt   = [i for i, u in enumerate(used_g) if not u]
    return matches, unmatched_pred, unmatched_gt

def _center_crop_clamped(img: np.ndarray, cx: float, cy: float, w: int, h: int) -> np.ndarray:
    H, W = img.shape[:2]
    w = max(1, int(round(w))); h = max(1, int(round(h)))
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w)); y0 = max(0, min(y0, H - h))
    return img[y0:y0+h, x0:x0+w].copy()

def sub_abs(n: int) -> str:
    return (f"minus{abs(n)}" if n < 0 else f"plus{abs(n)}")

# ────────────────────── GT & predictions I/O ──────────────────────

def _read_and_normalize_gt(gt_csv: Path, gt_t_offset: int, out_dir: Path, max_frames: Optional[int]):
    """
    Load GT x,y,t; subtract gt_t_offset from t; save normalized GT CSV in out_dir.
    Returns: (gt_by_t dict, normalized_csv_path)
    """
    _ensure_dir(out_dir)
    norm_path = out_dir / f"{gt_csv.stem}_norm_offset{sub_abs(gt_t_offset)}.csv"
    gt_by_t: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    rows_norm = []
    with gt_csv.open('r', newline='') as f:
        r = csv.DictReader(f)
        need = {'x','y','t'}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"[stage9] GT CSV must have columns x,y,t; found: {r.fieldnames}")
        for row in r:
            try:
                x = int(round(float(row['x'])))
                y = int(round(float(row['y'])))
                t = int(round(float(row['t']))) - int(gt_t_offset)
                if t < 0:
                    continue
                if max_frames is not None and t >= max_frames:
                    continue
            except Exception:
                continue
            gt_by_t[t].append((x,y))
            rows_norm.append({'x':x,'y':y,'t':t})
    with norm_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['x','y','t'])
        w.writeheader()
        for r in rows_norm:
            w.writerow(r)
    return gt_by_t, norm_path

def _read_predictions(pred_csv: Path, only_firefly_rows: bool, max_frames: Optional[int]):
    """
    Accepts main pipeline CSV (post–Stage 8) or a slim preds CSV.
    Returns dict: t -> list of pred dicts: {x,y,b,f,conf}
    """
    preds_by_t: Dict[int, List[Dict]] = defaultdict(list)
    with pred_csv.open('r', newline='') as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        has_t = 't' in cols
        has_class = 'class' in cols
        need_logits = {'background_logit','firefly_logit'}.issubset(cols)
        if not need_logits:
            raise ValueError(f"[stage9] Predictions CSV requires logits columns; found: {r.fieldnames}")
        for row in r:
            try:
                cls = (row.get('class','firefly') or 'firefly').strip().lower() if has_class else 'firefly'
                if only_firefly_rows and cls != 'firefly':
                    continue
                t = int(row['t']) if has_t else int(row['frame'])
                if max_frames is not None and t >= max_frames:
                    continue
                x = float(row['x']); y = float(row['y'])
                b = float(row['background_logit']); ffl = float(row['firefly_logit'])
                if 'firefly_confidence' in row and row['firefly_confidence'] != '':
                    conf = float(row['firefly_confidence'])
                else:
                    conf = _softmax_conf_firefly(b, ffl)
            except Exception:
                continue
            preds_by_t[t].append({'x':x,'y':y,'b':b,'f':ffl,'conf':conf})
    return preds_by_t

# ────────────────────── evaluation ──────────────────────

def _evaluate_frames(gt_by_t, preds_by_t, thr_px: float):
    """
    For a single threshold, compute per-frame TP/FP/FN partitions.
    Returns:
      fps_by_t:  t -> list[pred_dict]
      tps_by_t:  t -> list[pred_dict]
      fns_by_t:  t -> list[(x,y)]
      metrics:   (TP, FP, FN, mean_err)
    """
    fps_by_t: Dict[int, List[Dict]] = defaultdict(list)
    tps_by_t: Dict[int, List[Dict]] = defaultdict(list)
    fns_by_t: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    all_dists: List[float] = []

    frames = sorted(set(gt_by_t.keys()) | set(preds_by_t.keys()))
    for t in frames:
        gts = gt_by_t.get(t, [])
        preds = preds_by_t.get(t, [])
        preds_xy = [(p['x'], p['y']) for p in preds]
        matches, unmatched_pred_idxs, unmatched_gt_idxs = _greedy_match_full(gts, preds_xy, thr_px)

        for gi, pi, d in matches:
            tps_by_t[t].append(preds[pi])
            all_dists.append(d)
        for pi in unmatched_pred_idxs:
            fps_by_t[t].append(preds[pi])
        for gi in unmatched_gt_idxs:
            gx, gy = gts[gi]
            fns_by_t[t].append((int(round(gx)), int(round(gy))))

    TP = sum(len(v) for v in tps_by_t.values())
    FP = sum(len(v) for v in fps_by_t.values())
    FN = sum(len(v) for v in fns_by_t.values())
    mean_err = (sum(all_dists)/len(all_dists)) if all_dists else 0.0
    return fps_by_t, tps_by_t, fns_by_t, (TP, FP, FN, mean_err)

def _thr_folder_name(thr: float) -> str:
    return f"thr_{float(thr):.1f}px"

# ─────────────── CNN inference helpers (for FN confidence) ───────────────

def _device():
    if torch is None:
        return None
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        return torch.device('cpu')

def _build_backbone(backbone: str):
    if models is None or nn is None:
        return None
    b = (backbone or 'resnet18').lower()
    fns = {
        'resnet18':  models.resnet18,
        'resnet34':  models.resnet34,
        'resnet50':  models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    fn = fns.get(b, models.resnet18)
    net = fn(weights=None)
    in_f = net.fc.in_features
    net.fc = nn.Linear(in_f, 2)
    return net

def _center_crop_clamped_bgr_to_pil(img_bgr, cx: float, cy: float, w: int, h: int) -> "Image.Image":
    H, W = img_bgr.shape[:2]
    w = max(1, int(round(w))); h = max(1, int(round(h)))
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w)); y0 = max(0, min(y0, H - h))
    crop = img_bgr[y0:y0+h, x0:x0+w]
    if Image is None:
        raise RuntimeError("PIL not available")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _auto_find_weights(search_roots: List[Path]) -> Optional[Path]:
    try:
        cands = []
        for root in search_roots:
            if root and root.exists():
                for p in root.rglob('*.pt'):
                    cands.append(p)
        if not cands:
            return None
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]
    except Exception:
        return None

def _build_infer_fn(model_path: Optional[Path], backbone: str, imagenet_normalize: bool, print_load_status: bool, search_hint: Optional[Path]):
    if torch is None:
        if print_load_status:
            print("[stage9] torch not available; FN confidence will be NaN.")
        return None

    device = _device()
    if device is None:
        if print_load_status:
            print("[stage9] No device; FN confidence will be NaN.")
        return None

    if model_path is None:
        env = os.environ.get('FIREFLY_MODEL_PATH') or os.environ.get('MODEL_PATH')
        if env and Path(env).exists():
            model_path = Path(env)
        else:
            roots = []
            if search_hint:
                roots.extend([search_hint, search_hint.parent, search_hint.parent.parent])
            roots.append(Path.cwd())
            model_path = _auto_find_weights(roots)

    if not (model_path and Path(model_path).exists()):
        if print_load_status:
            print("[stage9] Weights file not found; FN confidence will be NaN. Set $FIREFLY_MODEL_PATH.")
        return None

    try:
        model = _build_backbone(backbone)
        model = model.to(device)
        sd = torch.load(str(model_path), map_location=device)
        for key in ('state_dict','model','net','weights'):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        model.load_state_dict(sd, strict=False)
        model.eval()
        if print_load_status:
            print(f"[stage9] Loaded weights for FN scoring: {model_path}")
    except Exception as e:
        if print_load_status:
            print(f"[stage9] Failed to load weights ({model_path}): {e}")
        return None

    tfms = [T.ToTensor()]
    if imagenet_normalize:
        tfms.append(T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    transform = T.Compose(tfms)

    @torch.no_grad()
    def infer_on_frame(frame_bgr: np.ndarray, cx: float, cy: float, w: int, h: int) -> float:
        try:
            pil = _center_crop_clamped_bgr_to_pil(frame_bgr, cx, cy, w, h)
            ten = transform(pil).unsqueeze(0).to(device)
            logits = model(ten)[0].detach().cpu().numpy().tolist()
            bg, ff = float(logits[0]), float(logits[1])
            m = max(bg, ff)
            eb = math.exp(bg - m); ef = math.exp(ff - m)
            denom = eb + ef
            return (ef / denom) if denom > 0 else 0.5
        except Exception:
            return float('nan')

    return infer_on_frame

# ────────────────────── output writers ──────────────────────

def _write_crops_and_csvs_for_threshold(
    video_path: Path,
    out_root: Path,
    thr: float,
    fps_by_t, tps_by_t, fns_by_t,
    crop_w: int, crop_h: int,
    max_frames: Optional[int],
    fn_conf_getter = None   # callable(frame_bgr, cx, cy, w, h) -> conf float
):
    thr_dir = out_root / _thr_folder_name(thr)
    crops_dir_fp = thr_dir / "crops"
    crops_dir_tp = thr_dir / "tp_crops"
    crops_dir_fn = thr_dir / "fn_crops"
    for d in (thr_dir, crops_dir_fp, crops_dir_tp, crops_dir_fn):
        _ensure_dir(d)

    csv_fp = thr_dir / "fps.csv"
    csv_tp = thr_dir / "tps.csv"
    csv_fn = thr_dir / "fns.csv"

    with csv_fp.open('w', newline='') as f_fp, \
         csv_tp.open('w', newline='') as f_tp, \
         csv_fn.open('w', newline='') as f_fn:

        w_fp = csv.writer(f_fp); w_fp.writerow(['x','y','t','filepath','confidence'])
        w_tp = csv.writer(f_tp); w_tp.writerow(['x','y','t','filepath','confidence'])
        w_fn = csv.writer(f_fn); w_fn.writerow(['x','y','t','filepath','confidence'])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"[stage9] Could not open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        limit = min(max_frames, total_frames) if max_frames is not None else total_frames

        fr = 0
        while True:
            if limit is not None and fr >= limit:
                break
            ok, frame = cap.read()
            if not ok:
                break

            # FPs: use conf from preds
            for p in fps_by_t.get(fr, []):
                x = float(p['x']); y = float(p['y']); conf = float(p['conf'])
                crop = _center_crop_clamped(frame, x, y, crop_w, crop_h)
                fname = f"FP_t{fr:06d}_x{int(round(x))}_y{int(round(y))}_conf{conf:.4f}.png"
                outp = crops_dir_fp / fname
                cv2.imwrite(str(outp), crop)
                w_fp.writerow([int(round(x)), int(round(y)), fr, str(outp), f"{conf:.6f}"])

            # TPs: use conf from preds
            for p in tps_by_t.get(fr, []):
                x = float(p['x']); y = float(p['y']); conf = float(p['conf'])
                crop = _center_crop_clamped(frame, x, y, crop_w, crop_h)
                fname = f"TP_t{fr:06d}_x{int(round(x))}_y{int(round(y))}_conf{conf:.4f}.png"
                outp = crops_dir_tp / fname
                cv2.imwrite(str(outp), crop)
                w_tp.writerow([int(round(x)), int(round(y)), fr, str(outp), f"{conf:.6f}"])

            # FNs: compute conf via model if available; else NaN
            for (gx, gy) in fns_by_t.get(fr, []):
                conf = float('nan')
                if fn_conf_getter is not None:
                    try:
                        conf = float(fn_conf_getter(frame, float(gx), float(gy), crop_w, crop_h))
                    except Exception:
                        conf = float('nan')
                crop = _center_crop_clamped(frame, gx, gy, crop_w, crop_h)
                # use 0.0 in filename if NaN (to avoid "confnan")
                conf_for_name = 0.0 if conf != conf else conf
                fname = f"FN_t{fr:06d}_x{gx}_y{gy}_conf{conf_for_name:.4f}.png"
                outp = crops_dir_fn / fname
                cv2.imwrite(str(outp), crop)
                w_fn.writerow([gx, gy, fr, str(outp), ("" if conf!=conf else f"{conf:.6f}")])

            _progress(fr+1, limit, f"stage9-crops@{_thr_folder_name(thr)}"); fr += 1

        cap.release()

# ────────────────────── entry point ──────────────────────

def stage9_validate_against_gt(
    *,
    orig_video_path: Path,
    pred_csv_path: Path,
    gt_csv_path: Path,
    out_dir: Path,
    dist_thresholds: List[float],
    crop_w: int = 10,
    crop_h: int = 10,
    gt_t_offset: int = 9000,
    max_frames: Optional[int] = None,
    only_firefly_rows: bool = True,
    show_per_frame: bool = False,
    # NEW: for FN confidence scoring
    model_path: Optional[Path] = None,
    backbone: str = 'resnet18',
    imagenet_normalize: bool = False,
    print_load_status: bool = True,
):
    """
    Validate predictions against ground truth across a sweep of distance thresholds.

    Notes:
    - Assumes pipeline is zero-based. Ground truth t will be normalized by subtracting gt_t_offset.
    - Saves normalized GT CSV into out_dir, and also copies it into the main CSV folder.
    - Predictions CSV can be the main pipeline CSV (post–Stage 8).
    - FP/TP/FN crop filenames now include the firefly confidence (from preds for FP/TP; from model for FN).
    """
    _ensure_dir(out_dir)

    # Normalize GT (subtract offset) and write a normalized copy
    gt_by_t, norm_gt_csv = _read_and_normalize_gt(gt_csv_path, gt_t_offset, out_dir, max_frames)
    print(f"[stage9] Normalized GT saved to: {norm_gt_csv}")

    # Also copy normalized GT to the pipeline's CSV directory (same folder as predictions CSV)
    try:
        csv_dir = pred_csv_path.parent
        if csv_dir and csv_dir.exists():
            norm_copy = csv_dir / norm_gt_csv.name
            if str(norm_copy.resolve()) != str(norm_gt_csv.resolve()):
                import shutil as _shutil
                _shutil.copyfile(norm_gt_csv, norm_copy)
                print(f"[stage9] Normalized GT also saved to: {norm_copy}")
    except Exception as e:
        print(f"[stage9] Warning: failed to copy normalized GT to CSV folder: {e}")

    # Read predictions (use only class=='firefly' rows by default)
    preds_by_t = _read_predictions(pred_csv_path, only_firefly_rows, max_frames)

    # Build FN confidence getter (tries env/auto if model_path is None)
    fn_conf_getter = _build_infer_fn(model_path, backbone, imagenet_normalize, print_load_status, search_hint=pred_csv_path.parent)

    # Overview
    print("\n=== Detection Metrics (point, same-frame, dist sweep) ===")
    print(f"Video: {orig_video_path}")
    print(f"Pred CSV: {pred_csv_path}")
    print(f"GT CSV (raw): {gt_csv_path}")
    print(f"GT offset: {gt_t_offset} (t' = t - offset)")
    if max_frames is not None:
        print(f"Max frames considered: {max_frames}")
    print(f"Distance thresholds (px): {', '.join(str(t) for t in dist_thresholds)}")
    print()

    # Sweep thresholds
    for thr in dist_thresholds:
        fps_by_t, tps_by_t, fns_by_t, (TP, FP, FN, mean_err) = _evaluate_frames(gt_by_t, preds_by_t, thr)

        if show_per_frame:
            frames = sorted(set(gt_by_t.keys()) | set(preds_by_t.keys()))
            for t in frames:
                gs = len(gt_by_t.get(t, []))
                ps = len(preds_by_t.get(t, []))
                ts = len(tps_by_t.get(t, []))
                fs = len(fps_by_t.get(t, []))
                ns = len(fns_by_t.get(t, []))
                print(f"t={t:06d} | GT:{gs}  Pred:{ps}  TP:{ts}  FP:{fs}  FN:{ns}")

        prec = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        rec  = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        f1   = (2*prec*rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # Save crops + per-threshold CSVs
        _write_crops_and_csvs_for_threshold(
            orig_video_path, out_dir, float(thr),
            fps_by_t, tps_by_t, fns_by_t,
            crop_w, crop_h, max_frames,
            fn_conf_getter=fn_conf_getter
        )

        # Clean, labeled summary block
        print(f"Threshold: {thr:.1f}px")
        print(f"  TP: {TP}   FP: {FP}   FN: {FN}")
        print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}")
        print(f"  Mean error (px): {mean_err:.3f}\n")
