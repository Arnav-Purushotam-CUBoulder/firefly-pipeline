#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

import params
from utils import open_video, center_crop_with_pad, progress


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


def _intensity_centroid(img_gray: np.ndarray, sigma: float) -> Tuple[float, float]:
    img = img_gray.astype(np.float32)
    if sigma and sigma > 0:
        gh, gw = img.shape[:2]
        G = _gaussian_kernel(gw, gh, sigma)
        img = img * G
    total = float(img.sum())
    if total <= 1e-6:
        H, W = img.shape[:2]
        return (W / 2.0, H / 2.0)
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    cx = float((xs * img).sum() / total)
    cy = float((ys * img).sum() / total)
    return cx, cy


def run_for_video(video_path: Path) -> Path:
    """Recenter selected patches using Gaussian-weighted intensity centroid.

    Reads Stage 2 CSV (x,y as centers), refines center, saves rows and
    crops with a red dot at the refined center.
    """
    stem = video_path.stem
    s2_csv = (params.STAGE2_DIR / stem) / f"{stem}_patches.csv"
    assert s2_csv.exists(), f"Missing Stage2 CSV for {stem}: {s2_csv}"

    out_root = params.STAGE3_DIR / stem
    crops_dir = out_root / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_root / f"{stem}_gauss.csv"

    cap, W, H, fps, total = open_video(video_path)

    rows = []
    with s2_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    with out_csv.open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["x", "y", "w", "h", "frame_number", "global_trajectory_id"])  # centers

        # Group rows by frame for efficient seeking
        by_frame: dict[int, list[dict]] = {}
        for row in rows:
            t = int(row["frame_number"])
            by_frame.setdefault(t, []).append(row)

        frames_sorted = sorted(by_frame.keys())
        if params.MAX_FRAMES is not None:
            max_idx = max(0, int(params.MAX_FRAMES) - 1)
            frames_sorted = [t for t in frames_sorted if t <= max_idx]
        for idx, t in enumerate(frames_sorted):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(t))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for row in by_frame[t]:
                cx = float(row["x"])
                cy = float(row["y"])
                # Use global patch size everywhere (ignore CSV w,h)
                w = int(params.PATCH_SIZE_PX)
                h = int(params.PATCH_SIZE_PX)
                gid = int(row["global_trajectory_id"]) if "global_trajectory_id" in row else -1

                crop_g, x0, y0 = center_crop_with_pad(gray, cx, cy, w, h)
                ccx, ccy = _intensity_centroid(crop_g, float(params.GAUSS_SIGMA))
                new_cx = x0 + ccx
                new_cy = y0 + ccy

                # Save marked crop (red dot at refined center)
                crop_c, x0c, y0c = center_crop_with_pad(frame, new_cx, new_cy, w, h)
                px = int(round(new_cx - x0c))
                py = int(round(new_cy - y0c))
                if 0 <= py < crop_c.shape[0] and 0 <= px < crop_c.shape[1]:
                    crop_c[py, px] = (0, 0, 255)
                out_name = f"gid_{gid:06d}_t{t:06d}_x{int(round(new_cx))}_y{int(round(new_cy))}_{w}x{h}.png"
                cv2.imwrite(str(crops_dir / out_name), crop_c)

                wcsv.writerow([float(new_cx), float(new_cy), int(w), int(h), int(t), int(gid)])
            if idx % 20 == 0:
                progress(idx + 1, max(1, len(frames_sorted)), "Stage3 refine")
        progress(len(frames_sorted), max(1, len(frames_sorted)), "Stage3 refine done")

    cap.release()
    print(f"Stage3  Wrote refined CSV → {out_csv}")
    return out_csv


__all__ = ["run_for_video"]
