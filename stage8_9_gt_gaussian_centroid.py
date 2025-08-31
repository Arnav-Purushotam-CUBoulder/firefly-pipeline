#!/usr/bin/env python3
"""
Stage 8.9 — Recenter GT via Gaussian centroid and write x,y,t (t = RAW frame)
──────────────────────────────────────────────────────────────────────────────

Reads a GT CSV with columns x,y,w,h,frame (x,y are centers), computes a
Gaussian-weighted centroid in a crop_w×crop_h window for the corresponding
video frame, and OVERWRITES gt_csv_path with x,y,t where **t is the RAW frame
number from the filename (offset NOT subtracted)**.

Saves crops (if out_crop_dir):
  • A *clean* GT crop (no overlay) with a “__clean” suffix.
  • A debug crop with the centroid pixel colored red (BGR=(0,0,255)).

Returns: (num_processed_points, num_saved_files)
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
_BAR = 50
def _progress(i: int, total: int, tag: str = '') -> None:
    total = max(1, int(total or 1))
    i = min(i, total)
    done = int(_BAR * i / total)
    sys.stdout.write(f"\r[{tag:<14}] |{'#'*done}{'.'*(max(0,_BAR-done))}| {i:>5}/{total:<5}")
    sys.stdout.flush()
    if i >= total:
        sys.stdout.write("\n")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
def _center_crop_clamped(img: np.ndarray, cx: float, cy: float, w: int, h: int) -> Tuple[np.ndarray, int, int]:
    H, W = img.shape[:2]
    w = max(1, int(round(w))); h = max(1, int(round(h)))
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w)); y0 = max(0, min(y0, H - h))
    return img[y0:y0+h, x0:x0+w].copy(), x0, y0

def _gaussian_kernel(w: int, h: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.full((h, w), 1.0/(h*w), dtype=np.float32)
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    g = np.exp(-((x - xc)**2 + (y - yc)**2) / (2.0 * sigma**2))
    s = g.sum()
    if s > 0:
        g /= s
    return g.astype(np.float32)

def _intensity_centroid(gray: np.ndarray, gaussian_sigma: float = 0.0) -> Tuple[float, float]:
    if gray.size == 0:
        return 0.0, 0.0
    img = gray.astype(np.float32)
    if gaussian_sigma and gaussian_sigma > 0:
        ker = _gaussian_kernel(img.shape[1], img.shape[0], gaussian_sigma)
        wimg = img * ker
        total = wimg.sum()
        if total <= 1e-6:
            wimg = img
            total = float(wimg.sum()) if wimg.size else 1.0
    else:
        wimg = img
        total = float(wimg.sum()) if wimg.size else 1.0

    h, w = wimg.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    cx = float((wimg * xs).sum() / (total if total != 0 else 1.0))
    cy = float((wimg * ys).sum() / (total if total != 0 else 1.0))
    return cx, cy

# ──────────────────────────────────────────────────────────────
def _read_gt_xywh_frame(gt_csv: Path) -> List[dict]:
    with gt_csv.open('r', newline='') as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        if {'x','y','w','h','frame'} <= cols:
            return [dict(row) for row in r]
        elif {'x','y','t'} <= cols:
            print(f"[stage8.9] Detected x,y,t schema in {gt_csv.name}; skipping rewrite.")
            return []
        else:
            raise ValueError(f"[stage8.9] Expected GT columns x,y,w,h,frame; found: {sorted(cols)}")

def _atomic_write_xy_t(out_csv: Path, rows_xy_t: List[Tuple[int,int,int]]) -> None:
    tmp = out_csv.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        wri = csv.DictWriter(f, fieldnames=['x','y','t'])
        wri.writeheader()
        for (x,y,t) in rows_xy_t:
            wri.writerow({'x':x,'y':y,'t':t})
    tmp.replace(out_csv)

def _parse_frame_index(frame_name: str) -> Optional[int]:
    digs = re.findall(r'\d+', frame_name)
    return int(digs[-1]) if digs else None

# ──────────────────────────────────────────────────────────────
def stage8_9_recenter_gt_gaussian_centroid(
    *,
    orig_video_path: Path,
    gt_csv_path: Path,
    crop_w: int = 10,
    crop_h: int = 10,
    gaussian_sigma: float = 1.0,
    gt_t_offset: int = 9000,                 # used ONLY to locate the frame in the video
    max_frames: Optional[int] = None,
    out_crop_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[int,int]:
    """
    Reads GT CSV with columns x,y,w,h,frame (x,y are centers), computes a
    Gaussian-weighted centroid in a crop_w×crop_h window for the corresponding
    video frame, and OVERWRITES gt_csv_path with x,y,t where **t is the RAW
    frame number from the filename (offset NOT subtracted)**.

    Also saves:
      • Clean GT crop (no overlay) with a “__clean” suffix,
      • Debug crop with the centroid pixel colored red.
    Returns: (num_processed_points, num_saved_files)
    """
    rows = _read_gt_xywh_frame(gt_csv_path)
    if not rows:
        return 0, 0

    # Map from normalized frame index (for video access) -> entries
    # Each entry keeps the raw frame index for CSV output.
    by_norm_t: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        try:
            x = float(r['x']); y = float(r['y'])
            w = int(round(float(r.get('w', crop_w))))
            h = int(round(float(r.get('h', crop_h))))
            fstr = str(r['frame'])
            raw_idx = _parse_frame_index(fstr)  # e.g., 9019 from frame_009019.png
            if raw_idx is None:
                continue
            norm_t = raw_idx - int(gt_t_offset)  # used only to index into the video
            if norm_t < 0:
                # Can't index video with negative; skip this row
                continue
            by_norm_t[norm_t].append({
                'x': x, 'y': y, 'w': w, 'h': h, 'raw_idx': raw_idx, 'frame_name': fstr
            })
        except Exception:
            continue

    if not by_norm_t:
        if verbose:
            print("[stage8.9] No valid GT rows after parsing/offset for video access.")
        return 0, 0

    if out_crop_dir is not None:
        _ensure_dir(out_crop_dir)

    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage8.9] Could not open video: {orig_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = min(total_frames, int(max_frames)) if max_frames is not None else total_frames

    out_rows: List[Tuple[int,int,int]] = []
    saved = 0
    processed = 0

    fr = 0  # normalized (0-based) frame index for the video
    while True:
        if fr >= limit:
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break

        entries = by_norm_t.get(fr, [])
        if entries:
            for e in entries:
                try:
                    cx0 = float(e['x']); cy0 = float(e['y'])
                    crop, x0, y0 = _center_crop_clamped(frame_bgr, cx0, cy0, crop_w, crop_h)
                    if crop.size == 0:
                        continue
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    ccx, ccy = _intensity_centroid(gray, gaussian_sigma)

                    new_cx = x0 + ccx
                    new_cy = y0 + ccy

                    x_int = int(round(new_cx))
                    y_int = int(round(new_cy))

                    # IMPORTANT: write RAW frame index (offset NOT subtracted)
                    out_rows.append((x_int, y_int, int(e['raw_idx'])))
                    processed += 1

                    if out_crop_dir is not None:
                        # 1) Save a clean GT crop (no red overlay)
                        clean = crop.copy()

                        # 2) Create the debug crop (with red dot at centroid)
                        dbg = clean.copy()
                        px = int(round(ccx)); py = int(round(ccy))
                        if 0 <= py < dbg.shape[0] and 0 <= px < dbg.shape[1]:
                            dbg[py, px] = (0, 0, 255)  # BGR red

                        base_name = (
                            f"tRaw{int(e['raw_idx']):06d}_norm{fr:06d}"
                            f"__orig({int(round(cx0))},{int(round(cy0))})"
                            f"__cent({x_int},{y_int})"
                        )

                        # Clean crop filename
                        clean_name = f"{base_name}__clean.png"
                        cv2.imwrite(str(out_crop_dir / clean_name), clean)

                        # Debug crop filename (keeps original naming)
                        dbg_name = f"{base_name}.png"
                        cv2.imwrite(str(out_crop_dir / dbg_name), dbg)

                        saved += 2
                except Exception:
                    continue

        _progress(fr+1, limit, 'stage8.9-gt'); fr += 1

    cap.release()

    _atomic_write_xy_t(gt_csv_path, out_rows)
    if verbose:
        print(f"[stage8.9] Processed {processed} GT points; wrote x,y,t (t=RAW frame) to {gt_csv_path}")
        if out_crop_dir is not None:
            print(f"[stage8.9] Saved {saved} crops (clean+debug) to {out_crop_dir}")

    return processed, saved

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Stage 8.9 — Recenter GT via Gaussian centroid and write x,y,t with t=RAW frame.")
    ap.add_argument("--video", required=True, help="Path to original video.")
    ap.add_argument("--gt", required=True, help="Path to input GT CSV (x,y,w,h,frame). Will be OVERWRITTEN to x,y,t (t=RAW frame).")
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma (0 => plain intensity centroid).")
    ap.add_argument("--crop_w", type=int, default=10)
    ap.add_argument("--crop_h", type=int, default=10)
    ap.add_argument("--gt_offset", type=int, default=9000, help="Offset used ONLY to index into the video; not applied to CSV output.")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--out_crops", type=str, default=None, help="Folder to save debug + clean crops (optional).")
    args = ap.parse_args()

    v = Path(args.video)
    g = Path(args.gt)
    out_dir = Path(args.out_crops) if args.out_crops else None

    stage8_9_recenter_gt_gaussian_centroid(
        orig_video_path=v,
        gt_csv_path=g,
        crop_w=int(args.crop_w),
        crop_h=int(args.crop_h),
        gaussian_sigma=float(args.sigma),
        gt_t_offset=int(args.gt_offset),
        max_frames=int(args.max_frames) if args.max_frames is not None else None,
        out_crop_dir=out_dir,
        verbose=True,
    )
