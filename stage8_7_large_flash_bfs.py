#!/usr/bin/env python3
"""
Stage 8.7 — Large-flash fixer via BFS + centroided square replacement.

Per frame:
  • For each firefly row (center semantics), run a BFS from the saved centroid
    over neighbors with gray >= INTENSITY_THR, not limited to 10×10.
  • Compute a Gaussian/intensity centroid over the grown region.
  • Build the MINIMUM square *centered at that centroid* that covers the region.
  • If area > 100 px, mark as a candidate replacement (carry over logits/class).
  • Run Stage-7-style dedupe (centroid distance threshold) among candidates,
    keep the heaviest by RGB-sum.
  • Delete all originals that contributed; insert the kept replacements.
  • Save side-by-side crops (old 10×10 vs new square) with detailed names.
  • Keep CSV column order; preserve extra columns (class/logits/confidence).
  • Update the *_fireflies_logits.csv (x,y,t,bg_logit,firefly_logit).

Outputs live under: ROOT / "stage8.7" / <video_stem> / replacements/
"""

from __future__ import annotations
import csv, sys, math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import cv2
import numpy as np
from audit_trail import AuditTrail

# ──────────────────────────────────────────────────────────────
# Tiny progress bar (same style as your other stages)
_BAR = 50
def _progress(i: int, total: int, tag: str = ''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar  = '=' * int(frac * _BAR) + ' ' * (_BAR - int(frac * _BAR))
    print(f'\r{tag} [{bar}] {int(frac*100):3d}%', end='' if i < total else '\n')

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Orchestrator accessor so we can reuse global paths/params
def _orc():
    import __main__ as ORC
    return ORC

# ──────────────────────────────────────────────────────────────
def _read_csv_rows(p: Path) -> List[dict]:
    try:
        with p.open("r", newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []

def _write_csv_rows(p: Path, rows: List[dict], field_order: Sequence[str]):
    # keep original columns + xy_semantics at end if present
    fieldnames = list(field_order)
    if 'xy_semantics' not in fieldnames:
        fieldnames.append('xy_semantics')
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def _rgb_sum_center_square(frame: np.ndarray, cx: float, cy: float, side: int) -> float:
    H, W = frame.shape[:2]
    s = max(1, int(side))
    x0 = int(round(cx - s/2.0))
    y0 = int(round(cy - s/2.0))
    x0 = max(0, min(x0, W - s))
    y0 = max(0, min(y0, H - s))
    crop = frame[y0:y0+s, x0:x0+s]
    return float(crop.sum()) if crop.size else -1.0

# ──────────────────────────────────────────────────────────────
# Masked intensity centroid (optionally Gaussian-weighted)
def _gaussian_kernel(w: int, h: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.ones((h, w), dtype=np.float32)
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0
    y  = np.arange(h, dtype=np.float32)[:, None]
    x  = np.arange(w, dtype=np.float32)[None, :]
    g  = np.exp(-((x - xc)**2 + (y - yc)**2) / (2.0 * sigma**2))
    s  = g.sum()
    return (g / s).astype(np.float32) if s > 0 else np.ones((h, w), dtype=np.float32)

def _masked_centroid(gray: np.ndarray, mask: np.ndarray, sigma: float) -> Tuple[float,float]:
    # gray, mask are for the SAME ROI
    img = gray.astype(np.float32)
    msk = (mask.astype(np.uint8) > 0).astype(np.float32)
    if img.size == 0 or msk.sum() <= 0:
        h, w = gray.shape[:2]
        return (w/2.0, h/2.0)
    if sigma > 0:
        G = _gaussian_kernel(gray.shape[1], gray.shape[0], sigma)
        img = img * G
    wimg = img * msk
    tot  = float(wimg.sum())
    if tot <= 1e-6:
        h, w = gray.shape[:2]
        return (w/2.0, h/2.0)
    ys, xs = np.mgrid[0:gray.shape[0], 0:gray.shape[1]].astype(np.float32)
    cx = float((xs * wimg).sum() / tot)
    cy = float((ys * wimg).sum() / tot)
    return cx, cy

# ──────────────────────────────────────────────────────────────
def _bfs_region(gray: np.ndarray, seed_x: int, seed_y: int, thr: int) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    Returns (mask, (minx, miny, maxx, maxy)) where mask is uint8 ROI-sized binary
    for the grown region (1s inside the bounding rectangle), or empty if seed < thr.
    """
    H, W = gray.shape[:2]
    if not (0 <= seed_x < W and 0 <= seed_y < H):
        return np.zeros((0,0), np.uint8), (0,0,-1,-1)
    if int(gray[seed_y, seed_x]) < int(thr):
        return np.zeros((0,0), np.uint8), (0,0,-1,-1)

    visited = np.zeros_like(gray, dtype=np.uint8)
    q = deque()
    q.append((seed_x, seed_y))
    visited[seed_y, seed_x] = 1
    minx = maxx = seed_x
    miny = maxy = seed_y

    while q:
        x, y = q.popleft()
        # expand bounds
        if x < minx: minx = x
        if x > maxx: maxx = x
        if y < miny: miny = y
        if y > maxy: maxy = y
        # 8-neighbors
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and not visited[ny, nx]:
                    if int(gray[ny, nx]) >= int(thr):
                        visited[ny, nx] = 1
                        q.append((nx, ny))
                    else:
                        visited[ny, nx] = 2  # seen but below thr (avoid requeue)

    if maxx < minx or maxy < miny:
        return np.zeros((0,0), np.uint8), (0,0,-1,-1)

    # ROI + mask inside ROI
    roi = gray[miny:maxy+1, minx:maxx+1]
    roi_mask = (visited[miny:maxy+1, minx:maxx+1] == 1).astype(np.uint8)
    return roi_mask, (minx, miny, maxx, maxy)

def _centered_min_square_for_region(cx: float, cy: float, minx: int, miny: int, maxx: int, maxy: int) -> Tuple[int,int,int]:
    """
    Returns (square_side, x0, y0) for the minimal square centered at (cx,cy) that
    fully contains [minx..maxx] × [miny..maxy] (inclusive).
    """
    # max distance from centroid to the rectangle edges along each axis
    dx = max(abs(cx - minx), abs(maxx - cx))
    dy = max(abs(cy - miny), abs(maxy - cy))
    # side must cover 2*dx and 2*dy; +1 to be inclusive on pixels
    side = int(math.ceil(2*max(dx, dy) + 1))
    x0 = int(round(cx - side/2.0))
    y0 = int(round(cy - side/2.0))
    return side, x0, y0

def _euclid(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.hypot(dx, dy)

# ──────────────────────────────────────────────────────────────
def stage8_7_expand_large_fireflies(
    *,
    orig_video_path: Path,
    main_csv_path: Path,
    neighbor_intensity_threshold: int = 60,
    dedupe_dist_px: float = 8.0,
    min_square_area_px: int = 101,             # must be > 100 to exceed 10×10
    gaussian_sigma: float = 0.0,               # reuse orchestrator’s Stage-8 sigma
    max_frames: Optional[int] = None,
    verbose: bool = True,
    audit: AuditTrail | None = None,
) -> Dict[str, int]:
    """
    Returns metrics dict:
      {'frames_seen', 'candidates', 'replaced', 'dedup_groups', 'old_rows_removed',
       'new_rows_inserted', 'skipped_dim_seed'}
    """
    ORC = _orc()
    ROOT = ORC.ROOT

    rows = _read_csv_rows(main_csv_path)
    if not rows:
        print("[stage8.7] Main CSV empty — nothing to process.")
        return {}

    # field order for writing back
    orig_fields = list(rows[0].keys())

    # group rows by frame
    by_t: Dict[int, List[dict]] = defaultdict(list)
    has_class  = ('class' in rows[0].keys())
    has_xy_sem = ('xy_semantics' in rows[0].keys())
    for r in rows:
        try:
            t = int(r.get('frame', r.get('t', '0')))
            if max_frames is not None and t >= max_frames:
                continue
            # We ONLY consider firefly rows
            if has_class and r.get('class', 'firefly') != 'firefly':
                by_t[t].append(r)    # keep but don’t process
            else:
                by_t[t].append(r)
        except Exception:
            continue

    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage8.7] Could not open video: {orig_video_path}")
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = min(total_video_frames, max(by_t.keys(), default=0) + 1)
    if max_frames is not None:
        limit = min(limit, max_frames)

    video_stem = Path(orig_video_path).stem
    out_root = ROOT / "stage8.7" / video_stem / "replacements"
    _ensure_dir(out_root)

    # logits CSV path (made by Stage-8)
    main_ff_logits_path = main_csv_path.with_name(main_csv_path.stem + "_fireflies_logits.csv")
    existing_ff_rows: List[dict] = []
    ff_fieldnames = ['x','y','t','background_logit','firefly_logit']
    if main_ff_logits_path.exists():
        with main_ff_logits_path.open("r", newline="") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames:
                ff_fieldnames = rd.fieldnames
            existing_ff_rows = list(rd)

    metrics = dict(frames_seen=0, candidates=0, replaced=0, dedup_groups=0,
                   old_rows_removed=0, new_rows_inserted=0, skipped_dim_seed=0)

    # We'll rewrite the WHOLE CSV at the end; for now, accumulate per frame.
    new_all_rows: List[dict] = []

    # audit collectors
    pairs_log: List[dict] = []
    dim_seeds_log: List[dict] = []

    fr = 0
    while True:
        if fr >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break

        rows_f = by_t.get(fr, [])
        if not rows_f:
            _progress(fr+1, limit, 'stage8.7'); fr += 1; continue

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Build original-list and candidate replacements for THIS frame
        # Each cand: dict with keys (new_cx, new_cy, side, weight, src_idx, src_row)
        cands: List[dict] = []
        cand_src_idxs = set()
        for idx, r in enumerate(rows_f):
            # Only firefly rows can seed candidates
            if has_class and r.get('class', 'firefly') != 'firefly':
                continue

            # Derive center from semantics (expect center after Stage-8)
            try:
                w = int(round(float(r.get('w', 10)))); h = int(round(float(r.get('h', 10))))
                if has_xy_sem and str(r.get('xy_semantics','')).lower() == 'center':
                    cx = float(r['x']); cy = float(r['y'])
                else:
                    cx = float(r['x']) + w/2.0; cy = float(r['y']) + h/2.0
            except Exception:
                continue

            sx, sy = int(round(cx)), int(round(cy))
            if not (0 <= sx < W and 0 <= sy < H):
                continue
            if int(gray[sy, sx]) < int(neighbor_intensity_threshold):
                metrics['skipped_dim_seed'] += 1
                if audit is not None:
                    dim_seeds_log.append({
                        'video': str(orig_video_path),
                        'frame': fr,
                        'x': sx, 'y': sy,
                        'seed_intensity': int(gray[sy, sx]),
                        'min_thr': int(neighbor_intensity_threshold),
                    })
                continue

            mask_roi, (minx, miny, maxx, maxy) = _bfs_region(gray, sx, sy, int(neighbor_intensity_threshold))
            if mask_roi.size == 0:
                continue

            # Compute Gaussian/intensity centroid within grown region
            roi = gray[miny:maxy+1, minx:maxx+1]
            ccx_roi, ccy_roi = _masked_centroid(roi, mask_roi, float(gaussian_sigma))
            new_cx = float(minx) + float(ccx_roi)
            new_cy = float(miny) + float(ccy_roi)

            # Minimum square centered at (new_cx,new_cy) that contains region
            side, x0, y0 = _centered_min_square_for_region(new_cx, new_cy, minx, miny, maxx, maxy)
            side = int(max(1, min(side, min(W, H))))  # clamp side to frame

            if side * side <= int(min_square_area_px):  # must EXCEED 10×10 area
                continue

            wt = _rgb_sum_center_square(frame, new_cx, new_cy, side)

            cands.append({
                'new_cx': new_cx, 'new_cy': new_cy, 'side': side, 'weight': wt,
                'src_idx': idx, 'src_row': r
            })
            cand_src_idxs.add(idx)

        metrics['frames_seen'] += 1
        metrics['candidates']  += len(cands)

        # ──────────────────────────────────────────────────────
        # Global union-find across BOTH candidates and leftover originals
        # Node list = candidates + original firefly rows (that are not their sources)
        nodes: List[dict] = []
        # candidates
        for ci, c in enumerate(cands):
            nodes.append({
                'kind': 'cand', 'cand_idx': ci, 'row_idx': c['src_idx'],
                'cx': c['new_cx'], 'cy': c['new_cy'], 'side': c['side'], 'weight': c['weight']
            })
        # originals (include all firefly rows; dedupe will resolve with candidates)
        for i, r in enumerate(rows_f):
            if has_class and r.get('class','firefly') != 'firefly':
                continue
            # compute center for this row
            try:
                w = int(round(float(r.get('w', 10))))
                h = int(round(float(r.get('h', 10))))
                if has_xy_sem and str(r.get('xy_semantics','')).lower() == 'center':
                    rcx = float(r['x']); rcy = float(r['y'])
                else:
                    rcx = float(r['x']) + w/2.0
                    rcy = float(r['y']) + h/2.0
            except Exception:
                continue
            # weight for tie-breaks among originals if ever needed
            wt = _rgb_sum_center_square(frame, rcx, rcy, max(w, h))
            nodes.append({
                'kind': 'orig', 'row_idx': i,
                'cx': rcx, 'cy': rcy, 'side': max(w, h), 'weight': wt
            })

        N = len(nodes)
        kept_cands: List[dict] = []
        to_remove = set()

        if N:
            parent = list(range(N))
            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            thr = float(dedupe_dist_px)
            # union by centroid distance
            for i in range(N):
                ci = (nodes[i]['cx'], nodes[i]['cy'])
                for j in range(i+1, N):
                    cj = (nodes[j]['cx'], nodes[j]['cy'])
                    if _euclid(ci, cj) <= thr:
                        union(i, j)

            # bucket groups
            groups: Dict[int, List[int]] = defaultdict(list)
            for i in range(N):
                groups[find(i)].append(i)

            metrics['dedup_groups'] += len(groups)

            # For each group: if any candidate exists, keep the heaviest candidate; remove all others (orig+cand).
            for root, gidxs in groups.items():
                cand_nodes = [k for k in gidxs if nodes[k]['kind'] == 'cand']
                orig_nodes = [k for k in gidxs if nodes[k]['kind'] == 'orig']

                if not cand_nodes:
                    # no candidate in this group → keep originals as-is
                    continue

                # choose best candidate by weight
                best_k = max(cand_nodes, key=lambda k: nodes[k]['weight'])
                best_ci = nodes[best_k]['cand_idx']
                best = cands[best_ci]
                # record replacement
                # gather all original rows this group replaces (both cand sources and original nodes)
                group_src_rows = []
                seen_src = set()

                # add all candidate sources in this group
                for k in cand_nodes:
                    ci2 = nodes[k]['cand_idx']
                    src_idx2 = cands[ci2]['src_idx']
                    if src_idx2 not in seen_src:
                        group_src_rows.append(rows_f[src_idx2])
                        seen_src.add(src_idx2)
                    to_remove.add(src_idx2)

                # add all original nodes in this group
                for k in orig_nodes:
                    ri = nodes[k]['row_idx']
                    if ri not in seen_src:
                        group_src_rows.append(rows_f[ri])
                        seen_src.add(ri)
                    to_remove.add(ri)

                # also ensure the winning candidate's own source is removed
                to_remove.add(best['src_idx'])

                # store kept candidate along with all contributing old rows
                kept_cands.append({
                    'new_cx': best['new_cx'], 'new_cy': best['new_cy'],
                    'side': best['side'], 'weight': best['weight'],
                    'src_idx': best['src_idx'], 'src_row': best['src_row'],
                    'group_src_rows': group_src_rows,
                    'group_id': int(root),
                })

        metrics['old_rows_removed'] += len(to_remove)

        # Build frame rows: keep everything not removed, then append replacements
        frame_out_rows: List[dict] = []
        for i, r in enumerate(rows_f):
            if i in to_remove:
                continue
            frame_out_rows.append(r)

        for g in kept_cands:
            src = dict(g['src_row'])
            src['x']  = float(g['new_cx'])
            src['y']  = float(g['new_cy'])
            src['w']  = int(g['side'])
            src['h']  = int(g['side'])
            src['xy_semantics'] = 'center'
            frame_out_rows.append(src)

        # Save crops for each kept replacement — each in its own folder
        for g in kept_cands:
            src = g['src_row']
            # collect old crop metadata from all contributing rows
            old_crops_data = []
            for orow in g['group_src_rows']:
                try:
                    ow = int(round(float(orow.get('w', 10)))); oh = int(round(float(orow.get('h', 10))))
                    if has_xy_sem and str(orow.get('xy_semantics','')).lower() == 'center':
                        ocx = float(orow['x']); ocy = float(orow['y'])
                    else:
                        ocx = float(orow['x']) + ow/2.0; ocy = float(orow['y']) + oh/2.0
                except Exception:
                    continue
                ox0 = int(round(ocx - ow/2.0)); oy0 = int(round(ocy - oh/2.0))
                ox0 = max(0, min(ox0, W - ow)); oy0 = max(0, min(oy0, H - oh))
                old_crop = frame[oy0:oy0+oh, ox0:ox0+ow]
                old_crops_data.append({
                    'crop': old_crop, 'x0': ox0, 'y0': oy0, 'w': ow, 'h': oh,
                    'cx': ocx, 'cy': ocy
                })

            # new crop (center semantics)
            ns = int(g['side'])
            ncx, ncy = float(g['new_cx']), float(g['new_cy'])
            nx0 = int(round(ncx - ns/2.0)); ny0 = int(round(ncy - ns/2.0))
            nx0 = max(0, min(nx0, W - ns)); ny0 = max(0, min(ny0, H - ns))
            new_crop = frame[ny0:ny0+ns, nx0:nx0+ns]

            # build folder and filenames with details
            bg = src.get('background_logit', '')
            ff = src.get('firefly_logit', '')
            conf = src.get('firefly_confidence', '')
            ncxi, ncyi = int(round(ncx)), int(round(ncy))
            base = f"t{fr:06d}_new_x{ncxi}_y{ncyi}_{ns}x{ns}_thr{int(neighbor_intensity_threshold)}"
            if ff != '' or bg != '':
                base += f"_ff{ff}_bg{bg}"
            if conf != '':
                base += f"_conf{conf}"

            out_folder = out_root / base
            _ensure_dir(out_folder)

            # save all old crops (each file carries its own details)
            for k, d in enumerate(old_crops_data):
                ocxi, ocyi = int(round(d['cx'])), int(round(d['cy']))
                ow, oh = d['w'], d['h']
                old_path = out_folder / f"OLD_{k:02d}_x{ocxi}_y{ocyi}_{ow}x{oh}.png"
                cv2.imwrite(str(old_path), d['crop'])

            # save new crop
            new_path = out_folder / f"NEW_x{ncxi}_y{ncyi}_{ns}x{ns}.png"
            cv2.imwrite(str(new_path), new_crop)

            # 100x100 annotated context crop (new=green, old=red)
            annot = frame.copy()
            cv2.rectangle(annot, (nx0, ny0), (nx0+ns-1, ny0+ns-1), (0,255,0), 2)
            for d in old_crops_data:
                ox0, oy0, ow, oh = d['x0'], d['y0'], d['w'], d['h']
                cv2.rectangle(annot, (ox0, oy0), (ox0+ow-1, oy0+oh-1), (0,0,255), 2)

            xs = [nx0, nx0+ns-1] + [d['x0'] for d in old_crops_data] + [d['x0']+d['w']-1 for d in old_crops_data]
            ys = [ny0, ny0+ns-1] + [d['y0'] for d in old_crops_data] + [d['y0']+d['h']-1 for d in old_crops_data]
            if xs and ys:
                bx0, bx1 = max(0, min(xs)), min(W-1, max(xs))
                by0, by1 = max(0, min(ys)), min(H-1, max(ys))
                bc_x = (bx0 + bx1) / 2.0
                bc_y = (by0 + by1) / 2.0
            else:
                bc_x, bc_y = ncx, ncy

            ctx_side = 100
            ctx_s = min(ctx_side, W, H)
            cx0 = int(round(bc_x - ctx_s/2.0))
            cy0 = int(round(bc_y - ctx_s/2.0))
            cx0 = max(0, min(cx0, W - ctx_s))
            cy0 = max(0, min(cy0, H - ctx_s))
            context = annot[cy0:cy0+ctx_s, cx0:cx0+ctx_s]

            ctx_path = out_folder / ("CONTEXT_100.png" if ctx_s == 100 else f"CONTEXT_{ctx_s}.png")
            cv2.imwrite(str(ctx_path), context)

            # audit pair logs for each contributing original → kept replacement
            if audit is not None:
                for orow in g['group_src_rows']:
                    try:
                        ow = int(round(float(orow.get('w', 10)))); oh = int(round(float(orow.get('h', 10))))
                        if has_xy_sem and str(orow.get('xy_semantics','')).lower() == 'center':
                            ocx = float(orow['x']); ocy = float(orow['y'])
                        else:
                            ocx = float(orow['x']) + ow/2.0; ocy = float(orow['y']) + oh/2.0
                    except Exception:
                        continue
                    pairs_log.append({
                        'frame': fr,
                        'group_id': int(g.get('group_id', -1)),
                        'new_x': int(round(ncx)), 'new_y': int(round(ncy)),
                        'new_side': int(ns), 'new_weight': float(g.get('weight', -1.0)),
                        'old_x': int(round(ocx)), 'old_y': int(round(ocy)),
                        'old_w': int(ow), 'old_h': int(oh),
                        'reason': 'large_flash_replace',
                        'intensity_thr': int(neighbor_intensity_threshold),
                        'gaussian_sigma': float(gaussian_sigma),
                    })

        metrics['replaced']          += len(kept_cands)
        metrics['new_rows_inserted'] += len(kept_cands)

        # append to global list
        new_all_rows.extend(frame_out_rows)

        _progress(fr+1, limit, 'stage8.7'); fr += 1

    cap.release()

    # Rewrite MAIN CSV
    _write_csv_rows(main_csv_path, new_all_rows, orig_fields)

    # Rewrite fireflies_logits.csv (remove old coords, add new)
    if existing_ff_rows or any(('background_logit' in r or 'firefly_logit' in r) for r in new_all_rows):
        # Build an index of (t, x, y) we should have for fireflies (after 8.7).
        ff_map: Dict[Tuple[int,int,int], Tuple[str,str]] = {}
        for r in new_all_rows:
            try:
                if has_class and r.get('class','firefly') != 'firefly':
                    continue
                t = int(r.get('frame', r.get('t', '0')))
                x = int(round(float(r['x']))); y = int(round(float(r['y'])))
                bg = str(r.get('background_logit', ''))
                ff = str(r.get('firefly_logit', ''))
                if bg != '' or ff != '':
                    ff_map[(t, x, y)] = (bg, ff)
            except Exception:
                continue

        with main_ff_logits_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ff_fieldnames)
            w.writeheader()
            # prefer fresh rows derived from new_all_rows; fall back to existing rows for others
            written = set()
            for (t,x,y), (bg,ff) in ff_map.items():
                row = {'t':t, 'x':x, 'y':y, 'background_logit':bg, 'firefly_logit':ff}
                w.writerow({k: row.get(k, "") for k in ff_fieldnames})
                written.add((t,x,y))
            # keep any legacy rows not overwritten
            for r in existing_ff_rows:
                try:
                    t = int(r.get('t', 0)); x = int(r.get('x', 0)); y = int(r.get('y', 0))
                    if (t,x,y) in written: 
                        continue
                    w.writerow({k: r.get(k, "") for k in ff_fieldnames})
                except Exception:
                    continue

    # audit sidecars
    if audit is not None:
        if pairs_log:
            audit.log_pairs('08_7_large_flash_bfs', pairs_log, filename='replacements.csv')
        if dim_seeds_log:
            audit.log_removed('08_7_large_flash_bfs', 'dim_seed', dim_seeds_log,
                              extra_cols=['seed_intensity','min_thr'])

    # Summary
    print("\nStage 8.7 summary")
    print(f"  Frames processed       : {metrics['frames_seen']}")
    print(f"  Candidate replacements : {metrics['candidates']}")
    print(f"  Dedup groups           : {metrics['dedup_groups']}")
    print(f"  Replacements kept      : {metrics['replaced']}")
    print(f"  Old rows removed       : {metrics['old_rows_removed']}")
    print(f"  New rows inserted      : {metrics['new_rows_inserted']}")
    print(f"  Seeds below threshold  : {metrics['skipped_dim_seed']}")
    print(f"  Crops folder           : {out_root}")

    return metrics
