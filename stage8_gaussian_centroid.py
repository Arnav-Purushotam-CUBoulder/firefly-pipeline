#!/usr/bin/env python3
import csv
import sys
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
# Tiny progress bar
BAR_LEN = 50
def _progress(i, total, tag=''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac  = 0 if total == 0 else min(1.0, max(0.0, i / float(total)))
    bar   = '=' * int(frac * BAR_LEN) + ' ' * (BAR_LEN - int(frac * BAR_LEN))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i >= total:
        sys.stdout.write('\n')

# ──────────────────────────────────────────────────────────────
# Helpers

def _center_crop_clamped(img, cx: float, cy: float, w: int, h: int):
    H, W = img.shape[:2]
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    x0 = int(round(cx - w/2))
    y0 = int(round(cy - h/2))
    x0 = max(0, min(x0, W - w))
    y0 = max(0, min(y0, H - h))
    return img[y0:y0+h, x0:x0+w], x0, y0

def _gaussian_kernel(w: int, h: int, sigma: float):
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

def _intensity_centroid(img_gray: np.ndarray, gaussian_sigma: float = 0.0):
    """Return centroid (cx, cy) within the small crop."""
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

def _parse_float_safe(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default

def _infer_firefly_conf(row: dict) -> float:
    """Prefer 'firefly_confidence'. If missing, compute from logits if available."""
    if 'firefly_confidence' in row and row['firefly_confidence'] != '':
        return _parse_float_safe(row['firefly_confidence'])
    bg = _parse_float_safe(row.get('background_logit', np.nan))
    ff = _parse_float_safe(row.get('firefly_logit', np.nan))
    if not (math.isnan(bg) or math.isnan(ff)):
        # two-class softmax probability for firefly
        # to improve numerical stability:
        m = max(bg, ff)
        ebg = math.exp(bg - m)
        eff = math.exp(ff - m)
        denom = ebg + eff
        if denom > 0:
            return eff / denom
    return float('nan')

# ──────────────────────────────────────────────────────────────
# Stage-8 main entry point

def recenter_gaussian_centroid(
    orig_video_path: Path,
    csv_path: Path,
    *,
    # canonical params
    centroid_patch_w: int = 10,      # patch used to COMPUTE the centroid
    centroid_patch_h: int = 10,
    gaussian_sigma: float = 0.0,     # 0 = plain intensity; >0 = Gaussian-weighted
    max_frames: Optional[int] = None,
    verbose: bool = True,
    crop_dir: Optional[Path] = None, # where to save crops

    # backward-compat aliases
    patch_w: Optional[int] = None,
    patch_h: Optional[int] = None,
):
    """
    Reads csv_path (expects at least frame,x,y,w,h [,+class,+logits]),
    recomputes a refined centroid inside a (centroid_patch_w×centroid_patch_h) window
    using an optional Gaussian-weighted intensity moment.

    Then REWRITES the same CSV so that:
      • x,y = the refined Gaussian centroid (center semantics)
      • w,h = fixed patch size (10×10 by default) for rendering/cropping
      • All extra columns (class, logits, confidence, etc.) are preserved
      • Adds xy_semantics='center'

    Also writes a fireflies-only CSV (zero-based frames):
      x,y,t,background_logit,firefly_logit

    Finally, it RELOADS the rewritten CSV and saves crops for rows with class == 'firefly',
    using (x,y) as CENTER and (w,h) as size, paints the center pixel red,
    and appends the firefly confidence to each filename.
    """
    # map aliases if provided
    if patch_w is not None:
        centroid_patch_w = int(patch_w)
    if patch_h is not None:
        centroid_patch_h = int(patch_h)

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return

    # Outputs
    if crop_dir is None:
        crop_dir = csv_path.parent / 'stage8_crops'
    crop_dir.mkdir(parents=True, exist_ok=True)
    fireflies_csv = csv_path.with_name(csv_path.stem + '_fireflies_logits.csv')

    # Group detections by (zero-based) frame
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        try:
            f = int(r['frame'])  # zero-based
            if max_frames is not None and f >= max_frames:
                continue
            by_frame[f].append(r)
        except Exception:
            continue

    # Pass 1: refine centroids and overwrite main CSV
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {orig_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_rows: List[dict] = []
    firefly_rows: List[Tuple[float,float,int,float,float]] = []

    fr = 0  # ZERO-BASED
    while True:
        if max_frames is not None and fr >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        dets = by_frame.get(fr, [])
        if not dets:
            _progress(fr+1, total_frames, 'stage8'); fr += 1
            continue

        for r in dets:
            # current center from row's box (may still be top-left semantics from earlier stages)
            x, y, w, h = int(float(r['x'])), int(float(r['y'])), int(float(r['w'])), int(float(r['h']))
            cx = x + w/2.0
            cy = y + h/2.0

            # small crop to compute centroid
            crop, x0, y0 = _center_crop_clamped(frame, cx, cy, centroid_patch_w, centroid_patch_h)
            if crop.size == 0:
                continue
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ccx, ccy = _intensity_centroid(crop_gray, gaussian_sigma)

            # translate to image coords (centroid)
            new_cx = x0 + ccx
            new_cy = y0 + ccy

            # Set center semantics with a fixed 10×10 box for downstream
            FIX_W = 10; FIX_H = 10
            out = dict(r)
            out['x'] = float(new_cx)
            out['y'] = float(new_cy)
            out['w'] = FIX_W
            out['h'] = FIX_H
            out['xy_semantics'] = 'center'
            out_rows.append(out)

            # Collect for fireflies-only CSV (t = zero-based frame)
            if out.get('class') == 'firefly' and ('background_logit' in out or 'firefly_logit' in out):
                bg = _parse_float_safe(out.get('background_logit', 0.0), 0.0)
                ff = _parse_float_safe(out.get('firefly_logit', 0.0), 0.0)
                firefly_rows.append((float(new_cx), float(new_cy), fr, bg, ff))

        _progress(fr+1, total_frames, 'stage8'); fr += 1

    cap.release()

    # Overwrite main CSV (preserve original column order + xy_semantics at end)
    orig_fields = list(rows[0].keys())
    base = ['frame','x','y','w','h']
    ordered = [c for c in base if c in orig_fields] + [c for c in orig_fields if c not in base]
    with csv_path.open('w', newline='') as f:
        fieldnames = ordered[:]
        if 'xy_semantics' not in fieldnames:
            fieldnames.append('xy_semantics')
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        for r in out_rows:
            wri.writerow({k: r.get(k, '') for k in fieldnames})

    # Fireflies-only logits CSV (x,y as ints like your example; t is zero-based)
    with fireflies_csv.open('w', newline='') as f:
        wri = csv.writer(f)
        wri.writerow(['x','y','t','background_logit','firefly_logit'])
        for x_c, y_c, t, bg, ff in firefly_rows:
            wri.writerow([int(round(x_c)), int(round(y_c)), t, bg, ff])

    # Pass 2: reload rewritten CSV and save crops for fireflies (center semantics)
    det_rows = list(csv.DictReader(csv_path.open()))
    if det_rows:
        has_class = 'class' in det_rows[0].keys()
        has_xy_sem = 'xy_semantics' in det_rows[0].keys()
        has_conf  = 'firefly_confidence' in det_rows[0].keys()
        has_bglog = 'background_logit' in det_rows[0].keys()
        has_fflog = 'firefly_logit' in det_rows[0].keys()

        by_frame_firefly: Dict[int, List[dict]] = defaultdict(list)
        for r in det_rows:
            try:
                if has_class and r.get('class') != 'firefly':
                    continue
                f = int(r['frame'])
                if max_frames is not None and f >= max_frames:
                    continue
                by_frame_firefly[f].append(r)
            except Exception:
                continue

        cap2 = cv2.VideoCapture(str(orig_video_path))
        if not cap2.isOpened():
            raise RuntimeError(f"Could not open video: {orig_video_path}")

        total_video_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        saved = 0
        fr = 0  # ZERO-BASED
        while True:
            if max_frames is not None and fr >= max_frames:
                break
            ok, frame = cap2.read()
            if not ok:
                break

            rows_f = by_frame_firefly.get(fr, [])
            for r in rows_f:
                try:
                    w = int(float(r['w'])); h = int(float(r['h']))
                    if has_xy_sem and str(r.get('xy_semantics','')).lower() == 'center':
                        cx = float(r['x']); cy = float(r['y'])
                    else:
                        cx = float(r['x']) + w/2.0
                        cy = float(r['y']) + h/2.0

                    crop, x0, y0 = _center_crop_clamped(frame, cx, cy, w, h)

                    # Paint center pixel in red (BGR)
                    px = int(round(cx - x0))
                    py = int(round(cy - y0))
                    if 0 <= py < crop.shape[0] and 0 <= px < crop.shape[1]:
                        crop[py, px] = (0, 0, 255)

                    # Determine firefly confidence for filename
                    conf = _infer_firefly_conf(r)
                    # Format to 4 decimals; if NaN, write 'nan'
                    conf_str = f"{conf:.4f}" if conf == conf else "nan"

                    if crop.shape[0] == h and crop.shape[1] == w:
                        out_name = (
                            f"stage8_frame_{fr:06d}_x{int(round(cx))}_y{int(round(cy))}"
                            f"_{w}x{h}_conf{conf_str}.png"
                        )
                        cv2.imwrite(str(crop_dir / out_name), crop)
                        saved += 1
                except Exception:
                    continue

            _progress(fr+1, total_video_frames, 'stage8-crops'); fr += 1

        cap2.release()
        if verbose:
            print(f"Stage-8: saved {saved} crops to {crop_dir}")

    if verbose:
        print(f"Stage-8: wrote firefly logits CSV to {fireflies_csv}")
        print("Done.")
