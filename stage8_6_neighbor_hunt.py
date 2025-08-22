#!/usr/bin/env python3
"""
Stage 8.6 — neighbour hunting (post 8.5) with:
 • CNN scoring
 • Stage-8-style Gaussian recenter (local 10×10)
 • Full-frame debug overlays (white = search window, gray = blackout, red = final box+pixel)
 • Dedupe against ALL EXISTING **FIREFLY** rows in the frame (from the saved CSV)

Behavior
--------
For every firefly row in the main CSV:
  1) Hunt a bright neighbour around its center (central 10×10 blacked out in the search patch).
  2) Peak-brightness prefilter.
  3) CNN on a 10×10 at the peak; keep only if firefly_conf ≥ threshold.
  4) Recenter inside that 10×10 using Stage-8’s local Gaussian/intensity centroid.
  5) **Dedupe**: if the refined center is within min_separation_px of *any existing firefly* in that frame
     (from the CSV prior to 8.6), discard. Also dedupe against neighbours accepted earlier in this same frame.
  6) Append accepted neighbours to the main CSV (w=h=10, class/logits/conf, xy_semantics='center').
  7) Append to `<stem>_fireflies_logits.csv`.
  8) Save crop (red pixel at centroid) + full frame (white search, gray blackout, red final box).

"""

import csv, math, os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Progress bar
_BAR = 50
def _progress(i, total, tag=''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = 0 if total == 0 else min(1.0, max(0.0, i / total))
    bar  = '=' * int(frac * _BAR) + ' ' * (_BAR - int(frac * _BAR))
    try:
        import sys
        sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
        sys.stdout.flush()
        if i == total:
            sys.stdout.write('\n')
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# CSV utils
def _read_csv_rows(p: Path) -> List[dict]:
    try:
        with p.open('r', newline='') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []

def _write_csv_rows(p: Path, rows: List[dict], orig_cols: List[str]):
    fieldnames = orig_cols[:]
    for extra in ['class','background_logit','firefly_logit','firefly_confidence','xy_semantics']:
        if extra not in fieldnames:
            fieldnames.append(extra)
    with p.open('w', newline='') as f:
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        for r in rows:
            wri.writerow({k: r.get(k, '') for k in fieldnames})

# ──────────────────────────────────────────────────────────────
# Geometry + math
def _euclid2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    dx = float(a[0]) - float(b[0]); dy = float(a[1]) - float(b[1])
    return dx*dx + dy*dy

def _frame_firefly_centers(rows: List[dict], frame_idx: int, has_class: bool) -> List[Tuple[float,float]]:
    """Centers of EXISTING FIREFLY rows (or all rows if no 'class' column)."""
    out = []
    for r in rows:
        try:
            if int(r['frame']) != frame_idx: 
                continue
            if has_class and r.get('class') != 'firefly':
                continue
            w = int(round(float(r.get('w', 10))))
            h = int(round(float(r.get('h', 10))))
            if str(r.get('xy_semantics','')).lower() == 'center':
                cx = float(r['x']); cy = float(r['y'])
            else:
                cx = float(r['x']) + w/2.0; cy = float(r['y']) + h/2.0
            out.append((cx, cy))
        except Exception:
            continue
    return out

def _clamped_center_box(cx: float, cy: float, w: int, h: int, W: int, H: int) -> Tuple[int,int,int,int]:
    w = max(1, int(round(w))); h = max(1, int(round(h)))
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w)); y0 = max(0, min(y0, H - h))
    return x0, y0, x0 + w, y0 + h

# ── Stage-8-style Gaussian centroid on LOCAL 10×10 crop ───────
def _gaussian_kernel(w: int, h: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        k = np.ones((h, w), dtype=np.float32)
        return k / float(h * w)
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0
    y  = np.arange(h, dtype=np.float32)[:, None]
    x  = np.arange(w, dtype=np.float32)[None, :]
    g  = np.exp(-((x - xc)**2 + (y - yc)**2) / (2.0 * sigma**2))
    s  = g.sum()
    if s > 0:
        g /= s
    return g.astype(np.float32)

def _intensity_centroid_local(img_gray: np.ndarray, gaussian_sigma: float = 0.0) -> Tuple[float,float]:
    """
    Stage-8 equivalent: return centroid (cx, cy) IN LOCAL CROP COORDS.
    """
    img = img_gray.astype(np.float32)
    if gaussian_sigma and gaussian_sigma > 0:
        gh, gw = img.shape[:2]
        G = _gaussian_kernel(gw, gh, gaussian_sigma)
        img = img * G
    total = float(img.sum())
    if total <= 1e-6:
        H, W = img.shape[:2]
        return (W/2.0, H/2.0)
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    cx = float((xs * img).sum() / total)
    cy = float((ys * img).sum() / total)
    return cx, cy

# ──────────────────────────────────────────────────────────────
# Model helpers
def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _build_backbone(backbone: str) -> nn.Module:
    b = (backbone or 'resnet18').lower()
    fns = {
        'resnet18':  models.resnet18,
        'resnet34':  models.resnet34,
        'resnet50':  models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    net = fns.get(b, models.resnet18)(weights=None)
    in_f = net.fc.in_features
    net.fc = nn.Linear(in_f, 2)
    return net

def _softmax_fire_prob(bg_logit: float, ff_logit: float) -> float:
    m = max(bg_logit, ff_logit)
    eb = math.exp(bg_logit - m); ef = math.exp(ff_logit - m)
    denom = eb + ef
    return ef / denom if denom > 0 else 0.5

def _center_crop_bgr(frame_bgr: np.ndarray, cx: float, cy: float, w: int, h: int) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    H, W = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _clamped_center_box(cx, cy, w, h, W, H)
    return frame_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

@torch.no_grad()
def stage8_6_neighbor_hunt(
    *,
    orig_video_path: Path,
    csv_path: Path,
    neighbour_radius_px: int,
    blackout_w: int = 10,
    blackout_h: int = 10,
    min_brightness_for_neighbour: int = 50,
    min_separation_px: int = 2,
    # model settings
    model_path: Optional[Path] = None,
    backbone: str = 'resnet18',
    imagenet_normalize: bool = False,
    firefly_conf_thresh: float = 0.5,
    fail_if_weights_missing: bool = True,
    # Gaussian centroid
    gaussian_sigma: float = 1.0,
    # debug saves
    save_debug_artifacts: bool = True,
    # misc
    max_frames: Optional[int] = None,
    verbose: bool = True,
    **extra_kwargs,
) -> int:
    rows = _read_csv_rows(csv_path)
    if not rows:
        if verbose: print("[stage8.6] No rows; nothing to do.")
        return 0

    orig_cols = list(rows[0].keys())
    has_class = ('class' in rows[0].keys())

    # Build per-frame indexes
    # • by_frame: indexes FIRELY source rows we iterate over (same as before)
    # • centers_ff: EXISTING **firefly** centers (from the CSV prior to 8.6) we must dedupe against
    by_frame: Dict[int, List[int]] = {}
    centers_ff: Dict[int, List[Tuple[float,float]]] = {}  # prior fireflies in saved CSV
    frames_seen = set()
    for i, r in enumerate(rows):
        try:
            f = int(r['frame'])
            frames_seen.add(f)
            if max_frames is not None and f >= max_frames:
                continue
            if (not has_class) or (r.get('class') == 'firefly'):
                by_frame.setdefault(f, []).append(i)
        except Exception:
            continue
    for f in frames_seen:
        centers_ff[f] = _frame_firefly_centers(rows, f, has_class)

    # Video
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage8.6] Could not open video: {orig_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = min(max_frames, total_frames) if max_frames is not None else total_frames
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    # Output dirs (next to CSV)
    video_stem = Path(orig_video_path).stem
    base_dir   = csv_path.parent
    crops_dir  = base_dir / "stage8_6_neighbor_crops" / video_stem
    frames_dir = base_dir / "stage8_6_neighbor_frames" / video_stem
    if save_debug_artifacts:
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

    # Model
    dev = _device()
    net = _build_backbone(backbone).to(dev).eval()

    ok_weights = False
    if model_path is not None and Path(model_path).exists():
        try:
            blob = torch.load(str(model_path), map_location=dev)
            sd = None
            if isinstance(blob, dict):
                for k in ['state_dict','model_state_dict','net','model','weights']:
                    if k in blob and isinstance(blob[k], dict):
                        sd = blob[k]; break
                if sd is None and all(isinstance(v, torch.Tensor) for v in blob.values()):
                    sd = blob
            elif isinstance(blob, nn.Module):
                net.load_state_dict(blob.state_dict(), strict=False)
                ok_weights = True
            if sd is not None:
                net.load_state_dict(sd, strict=False)
                ok_weights = True
        except Exception as e:
            if verbose: print(f"[stage8.6] Warning: failed to load weights: {e}")
            ok_weights = False
    if not ok_weights:
        msg = f"[stage8.6] Model weights missing at {model_path}."
        if fail_if_weights_missing: raise RuntimeError(msg)
        if verbose: print(msg + " Proceeding with random weights (NOT recommended).")

    # Transforms
    tfms = [T.ToTensor()]
    if imagenet_normalize:
        tfms.append(T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    transform = T.Compose(tfms)

    # Idempotence caches:
    #  • existing fireflies in saved CSV (centers_ff)
    #  • neighbours we accept during this run (accepted_ff)
    accepted_ff: Dict[int, List[Tuple[float,float]]] = {f: [] for f in frames_seen}

    new_rows: List[dict] = []
    new_ff_logits_rows: List[dict] = []

    fr = 0
    while True:
        if limit is not None and fr >= limit: break
        ok, frame_bgr = cap.read()
        if not ok: break

        idxs = by_frame.get(fr, [])
        if not idxs:
            _progress(fr+1, limit, 'stage8.6'); fr += 1
            continue

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Centers to dedupe against in this frame = prior fireflies + those we accepted earlier in this frame
        def too_close_to_any(xy: Tuple[float,float]) -> bool:
            px, py = xy
            thr2 = float(min_separation_px**2)
            # prior fireflies
            for c in centers_ff.get(fr, []):
                if _euclid2((px,py), c) <= thr2: 
                    return True
            # accepted in this run (same frame)
            for c in accepted_ff.get(fr, []):
                if _euclid2((px,py), c) <= thr2:
                    return True
            return False

        for i in idxs:
            r = rows[i]
            try:
                w = int(round(float(r.get('w', 10))))
                h = int(round(float(r.get('h', 10))))
                if str(r.get('xy_semantics','')).lower() == 'center':
                    cx = float(r['x']); cy = float(r['y'])
                else:
                    cx = float(r['x']) + w/2.0; cy = float(r['y']) + h/2.0

                # Search window (includes blackout padding)
                pad_x = neighbour_radius_px + blackout_w//2
                pad_y = neighbour_radius_px + blackout_h//2
                sx0 = max(0, int(round(cx - pad_x)))
                sy0 = max(0, int(round(cy - pad_y)))
                sx1 = min(W, int(round(cx + pad_x)))
                sy1 = min(H, int(round(cy + pad_y)))
                if sx1 <= sx0 or sy1 <= sy0: 
                    continue

                patch = gray[sy0:sy1, sx0:sx1].copy()

                # Black out central 10×10 inside the search patch
                bx0 = int(round(cx - blackout_w/2.0)) - sx0
                by0 = int(round(cy - blackout_h/2.0)) - sy0
                bx1 = bx0 + blackout_w
                by1 = by0 + blackout_h
                bx0_c = max(0, min(bx0, patch.shape[1]))
                by0_c = max(0, min(by0, patch.shape[0]))
                bx1_c = max(0, min(bx1, patch.shape[1]))
                by1_c = max(0, min(by1, patch.shape[0]))
                if bx1_c > bx0_c and by1_c > by0_c:
                    patch[by0_c:by1_c, bx0_c:bx1_c] = 0

                # Peak search
                _, max_val, _, max_loc = cv2.minMaxLoc(patch)
                if max_val < float(min_brightness_for_neighbour):
                    continue

                # Candidate center (global)
                nx = float(sx0 + max_loc[0])
                ny = float(sy0 + max_loc[1])

                # Early dedupe vs existing & already-accepted fireflies
                if too_close_to_any((nx, ny)):
                    continue

                # CNN on 10×10 at (nx,ny)
                crop10_bgr, (cx0, cy0, cx1, cy1) = _center_crop_bgr(frame_bgr, nx, ny, blackout_w, blackout_h)
                x = transform(_bgr_to_pil(crop10_bgr)).unsqueeze(0).to(_device())
                logits = net(x)[0]
                bg_logit = float(logits[0].item())
                ff_logit = float(logits[1].item())
                ff_conf  = _softmax_fire_prob(bg_logit, ff_logit)
                if ff_conf < float(firefly_conf_thresh):
                    continue

                # Recenter using Stage-8 local-crop Gaussian centroid
                crop_gray = cv2.cvtColor(crop10_bgr, cv2.COLOR_BGR2GRAY)
                ccx, ccy  = _intensity_centroid_local(crop_gray, gaussian_sigma)
                gx        = float(cx0 + ccx)
                gy        = float(cy0 + ccy)

                # Final dedupe right before saving (AGAINST EXISTING CSV FIRELIES + accepted in-run)
                if too_close_to_any((gx, gy)):
                    continue

                # Final crop at refined center
                final_crop_bgr, (fx0, fy0, fx1, fy1) = _center_crop_bgr(frame_bgr, gx, gy, blackout_w, blackout_h)

                # Mark centroid as saturated RED pixel
                cx_local = int(round(gx - fx0))
                cy_local = int(round(gy - fy0))
                if 0 <= cy_local < final_crop_bgr.shape[0] and 0 <= cx_local < final_crop_bgr.shape[1]:
                    final_crop_bgr[cy_local, cx_local] = (0, 0, 255)

                # Save artifacts
                if save_debug_artifacts:
                    # 1) Crop
                    crop_name  = f"f{fr:05d}_x{int(round(gx))}_y{int(round(gy))}_conf{ff_conf:.3f}.png"
                    cv2.imwrite(str(crops_dir / crop_name),  final_crop_bgr)

                    # 2) Full frame overlays
                    frame_dbg = frame_bgr.copy()
                    # Search window (white)
                    cv2.rectangle(frame_dbg, (sx0, sy0), (sx1-1, sy1-1), (255,255,255), 1)
                    # Blacked-out region at ORIGINAL seed center (gray fill)
                    bx0_abs, by0_abs, bx1_abs, by1_abs = _clamped_center_box(cx, cy, blackout_w, blackout_h, W, H)
                    cv2.rectangle(frame_dbg, (bx0_abs, by0_abs), (bx1_abs-1, by1_abs-1), (128,128,128), thickness=-1)
                    # Final 10×10 neighbour bbox (red) + center pixel
                    rx0, ry0, rx1, ry1 = _clamped_center_box(gx, gy, blackout_w, blackout_h, W, H)
                    cv2.rectangle(frame_dbg, (rx0, ry0), (rx1-1, ry1-1), (0,0,255), 1)
                    if 0 <= int(round(gy)) < H and 0 <= int(round(gx)) < W:
                        frame_dbg[int(round(gy)), int(round(gx))] = (0,0,255)
                    frame_name = f"f{fr:05d}_x{int(round(gx))}_y{int(round(gy))}_conf{ff_conf:.3f}.jpg"
                    cv2.imwrite(str(frames_dir / frame_name), frame_dbg)

                # Append to main CSV
                new_r = {k: '' for k in orig_cols}
                new_r['frame'] = str(fr)
                new_r['x'] = f"{gx:.3f}"
                new_r['y'] = f"{gy:.3f}"
                new_r['w'] = str(int(blackout_w))
                new_r['h'] = str(int(blackout_h))
                if has_class:
                    new_r['class'] = 'firefly'
                new_r['background_logit']   = f"{bg_logit:.6f}"
                new_r['firefly_logit']      = f"{ff_logit:.6f}"
                new_r['firefly_confidence'] = f"{ff_conf:.6f}"
                new_r['xy_semantics']       = 'center'
                new_rows.append(new_r)

                # Append to logits CSV
                new_ff_logits_rows.append({
                    'x': str(int(round(gx))),
                    'y': str(int(round(gy))),
                    't': str(int(fr)),
                    'background_logit': f"{bg_logit:.6f}",
                    'firefly_logit':    f"{ff_logit:.6f}",
                })

                # Register as accepted so subsequent neighbours in this frame dedupe against it
                accepted_ff.setdefault(fr, []).append((gx, gy))

            except Exception:
                continue

        _progress(fr+1, limit, 'stage8.6'); fr += 1

    cap.release()

    if not new_rows:
        if verbose: print("[stage8.6] No neighbours accepted by CNN/dedupe; CSV unchanged.")
        return 0

    # Rewrite main CSV
    all_rows = rows + new_rows
    _write_csv_rows(csv_path, all_rows, list(orig_cols))

    # Update <stem>_fireflies_logits.csv
    ff_csv = csv_path.with_name(csv_path.stem + '_fireflies_logits.csv')
    ff_cols = ['x','y','t','background_logit','firefly_logit']
    existing = []
    if ff_csv.exists():
        try:
            with ff_csv.open('r', newline='') as f:
                rd = csv.DictReader(f)
                if rd.fieldnames: ff_cols = rd.fieldnames
                existing = list(rd)
        except Exception:
            pass

    with ff_csv.open('w', newline='') as f:
        wri = csv.DictWriter(f, fieldnames=ff_cols)
        wri.writeheader()
        for r in existing:
            wri.writerow({k: r.get(k, '') for k in ff_cols})
        for r in new_ff_logits_rows:
            wri.writerow({k: r.get(k, '') for k in ff_cols})

    if verbose:
        print(f"[stage8.6] Added {len(new_rows)} neighbours ≥ conf {firefly_conf_thresh} (σ={gaussian_sigma}) → {csv_path.name}")
        print(f"[stage8.6] Updated logits CSV with {len(new_ff_logits_rows)} rows → {ff_csv.name}")
        if save_debug_artifacts:
            print(f"[stage8.6] Crops → {crops_dir}")
            print(f"[stage8.6] Frames → {frames_dir}")

    return len(new_rows)
