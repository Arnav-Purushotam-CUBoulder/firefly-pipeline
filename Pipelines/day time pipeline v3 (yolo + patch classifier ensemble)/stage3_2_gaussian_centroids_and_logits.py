#!/usr/bin/env python3
from __future__ import annotations

"""
Stage 3.2: For Stage3.1-selected detections, compute Gaussian/intensity centroid
inside each crop and write a compact CSV for downstream analysis.

Inputs:
  - Stage3.1 selection CSV (preferred):
      STAGE3_DIR/<stem>/<stem>_patches_motion.csv
    (falls back to <stem>_patches_motion_all.csv filtered by traj_is_selected==1)
  - Stage3 crops:
      STAGE3_DIR/<stem>/crops/positives/*.png

Outputs (under STAGE3_DIR/<stem>/<STAGE3_2_DIRNAME>/):
  - <stem>_stage3_2_firefly_background_logits.csv with columns:
      x,y,t,firefly_logit,background_logit
    where x,y are the refined Gaussian centroid (full-frame coords), and
    logits are log-probabilities derived from Stage3 confidence.
  - params.STAGE3_2_XYT_EXPORT_DIR/<stem>.csv with columns:
      x,y,t
    exported for downstream 3D reconstruction.
  - (optional) annotated crops with a red pixel at the computed centroid.
"""

import csv
import math
import re
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

import params


_CROP_RE = re.compile(
    r"^f_(?P<t>\d+)_x(?P<x>-?\d+)_y(?P<y>-?\d+)_w(?P<w>\d+)_h(?P<h>\d+)_p(?P<p>\d+(?:\.\d+)?)\.png$"
)


def _gaussian_kernel(w: int, h: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        k = np.ones((h, w), dtype=np.float32)
        return k / float(h * w)
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    g = np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    s = float(g.sum())
    if s > 0:
        g /= s
    return g


def _intensity_centroid(gray: np.ndarray, sigma: float) -> Tuple[float, float]:
    img = gray.astype(np.float32)
    if sigma and sigma > 0:
        gh, gw = img.shape[:2]
        img = img * _gaussian_kernel(gw, gh, float(sigma))
    total = float(img.sum())
    if total <= 1e-6:
        H, W = img.shape[:2]
        return (W / 2.0, H / 2.0)
    ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(np.float32)
    cx = float((xs * img).sum() / total)
    cy = float((ys * img).sum() / total)
    return cx, cy


def _index_stage3_crops(pos_dir: Path) -> Dict[tuple[int, int, int, int, int], Path]:
    """Map (t,x,y,w,h) -> crop path (choosing highest p if duplicates)."""
    best: Dict[tuple[int, int, int, int, int], tuple[float, Path]] = {}
    if not pos_dir.exists():
        return {}
    for p in pos_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        m = _CROP_RE.match(p.name)
        if not m:
            continue
        try:
            key = (
                int(m.group("t")),
                int(m.group("x")),
                int(m.group("y")),
                int(m.group("w")),
                int(m.group("h")),
            )
            prob = float(m.group("p"))
        except Exception:
            continue
        prev = best.get(key)
        if prev is None or prob > prev[0]:
            best[key] = (prob, p)
    return {k: v for k, (_, v) in best.items()}


def _log_prob(p: float, eps: float = 1e-8) -> float:
    p = float(p)
    p = min(1.0 - eps, max(eps, p))
    return float(math.log(p))


def run_for_video(video_path: Path) -> Path:
    stem = video_path.stem
    s3_dir = params.STAGE3_DIR / stem

    selected_csv = s3_dir / f"{stem}_patches_motion.csv"
    all_csv = s3_dir / f"{stem}_patches_motion_all.csv"
    if selected_csv.exists():
        in_csv = selected_csv
        filter_selected = False
    elif all_csv.exists():
        in_csv = all_csv
        filter_selected = True
    else:
        raise FileNotFoundError(f"Missing Stage3.1 CSVs for {stem}: {selected_csv} / {all_csv}")

    out_root = s3_dir / str(getattr(params, "STAGE3_2_DIRNAME", "stage3_2"))
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"{stem}_stage3_2_firefly_background_logits.csv"
    out_xyt_dir = Path(getattr(params, "STAGE3_2_XYT_EXPORT_DIR", params.ROOT / "stage3_2 xyt for 3d reconstruction"))
    out_xyt_dir.mkdir(parents=True, exist_ok=True)
    out_xyt_csv = out_xyt_dir / f"{stem}.csv"

    pos_dir = s3_dir / "crops" / "positives"
    crop_index = _index_stage3_crops(pos_dir)

    sigma = float(getattr(params, "STAGE3_2_GAUSSIAN_SIGMA", 1.0))
    save_crops = bool(getattr(params, "STAGE3_2_SAVE_ANNOTATED_CROPS", True))
    mark_red = bool(getattr(params, "STAGE3_2_MARK_CENTROID_RED_PIXEL", True))
    crops_dir = out_root / "crops"
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_crops = 0
    bad_images = 0

    with (
        in_csv.open("r", newline="") as f_in,
        out_csv.open("w", newline="") as f_out,
        out_xyt_csv.open("w", newline="") as f_xyt,
    ):
        r = csv.DictReader(f_in)
        w = csv.writer(f_out)
        w_xyt = csv.writer(f_xyt)
        w.writerow(["x", "y", "t", "firefly_logit", "background_logit"])
        w_xyt.writerow(["x", "y", "t"])

        for row in r:
            if filter_selected:
                sel_raw = row.get("traj_is_selected")
                if sel_raw is not None and str(sel_raw).strip() in {"", "0", "False", "false"}:
                    continue
            try:
                t = int(row.get("frame_idx") or row.get("frame_number") or 0)
                x0 = int(float(row["x"]))
                y0 = int(float(row["y"]))
                ww = int(float(row.get("w", getattr(params, "PATCH_SIZE_PX", 10))))
                hh = int(float(row.get("h", getattr(params, "PATCH_SIZE_PX", 10))))
                conf = float(row.get("conf", 0.0) or 0.0)
                traj_id = int(float(row.get("traj_id", -1) or -1))
            except Exception:
                continue

            src = crop_index.get((t, x0, y0, ww, hh))
            if src is None or not src.exists():
                missing_crops += 1
                continue

            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                bad_images += 1
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ccx, ccy = _intensity_centroid(gray, sigma)
            gx = float(x0) + float(ccx)
            gy = float(y0) + float(ccy)

            firefly_logit = _log_prob(conf)
            background_logit = _log_prob(1.0 - conf)
            w.writerow([gx, gy, int(t), firefly_logit, background_logit])
            w_xyt.writerow([gx, gy, int(t)])
            written += 1

            if save_crops:
                out_img = img.copy()
                if mark_red:
                    px = int(round(ccx))
                    py = int(round(ccy))
                    if 0 <= py < out_img.shape[0] and 0 <= px < out_img.shape[1]:
                        out_img[py, px] = (0, 0, 255)
                out_name = (
                    f"traj_{traj_id:05d}_t{t:06d}_x{x0}_y{y0}_"
                    f"gcx{gx:.2f}_gcy{gy:.2f}_ff{conf:.3f}.png"
                )
                cv2.imwrite(str(crops_dir / out_name), out_img)

    print(f"Stage3.2 Wrote CSV → {out_csv} (rows={written})")
    print(f"Stage3.2 Wrote 3D reconstruction CSV → {out_xyt_csv} (rows={written})")
    if missing_crops:
        print(f"Stage3.2 NOTE: missing_crops={missing_crops} (expected Stage3 crops in {pos_dir})")
    if bad_images:
        print(f"Stage3.2 NOTE: bad_images={bad_images}")
    if save_crops:
        print(f"Stage3.2 Wrote annotated crops → {crops_dir}")
    return out_csv


__all__ = ["run_for_video"]
