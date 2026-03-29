#!/usr/bin/env python3
"""
nolan_mp4_to_predcsv.py
----------------------
Legacy lab baseline detector (Nolan R. Bonnie style).

This is adapted from the historical `mp4_to_xyt.py` script:
  1) Read video, keep green channel
  2) Rolling-mean background (window in seconds)
  3) Background subtraction + Gaussian blur + global threshold
  4) Connected components
  5) Intensity-weighted centroid per blob
  6) Write a Stage-5-compatible predictions CSV:
       x,y,w,h,t,class,xy_semantics,firefly_logit,background_logit,firefly_confidence

Note: This script is meant for automated baseline comparisons, not for production use.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _log_prob(p: float, eps: float = 1e-8) -> float:
    p = float(p)
    p = min(1.0 - eps, max(eps, p))
    return float(math.log(p))


def _measure(mask: np.ndarray, gray: np.ndarray) -> List[Dict[str, float]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return []

    dets: List[Dict[str, float]] = []
    for lab in range(1, int(num)):
        pix = labels == lab
        ys, xs = np.nonzero(pix)
        if xs.size == 0:
            continue
        intens = gray[pix].astype(np.float32, copy=False)
        weight = float(intens.sum())
        if weight <= 0.0:
            continue
        cx = float((xs.astype(np.float32) * intens).sum() / weight)
        cy = float((ys.astype(np.float32) * intens).sum() / weight)

        max_int = float(intens.max()) if intens.size else 0.0
        conf = max(0.0, min(1.0, max_int / 255.0))
        dets.append({"x": cx, "y": cy, "conf": conf})
    return dets


def run(
    *,
    video_path: Path,
    out_csv: Path,
    threshold: float,
    blur_sigma: float,
    bkgr_window_sec: float,
    max_frames: int | None,
    box_w: int,
    box_h: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = total_frames
    if max_frames is not None:
        limit = min(limit, int(max_frames))

    win_len = max(1, int(round(float(fps) * float(bkgr_window_sec))))
    buffer: deque[np.ndarray] = deque(maxlen=win_len)

    # Bootstrap background
    for _ in range(win_len):
        if len(buffer) >= limit:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        buffer.append(frame[:, :, 1].astype(np.float32))

    if not buffer:
        cap.release()
        raise RuntimeError("Video is empty (no frames read).")

    bkgr = np.mean(np.stack(list(buffer), axis=0), axis=0).astype(np.float32, copy=False)

    thr_val = int(round(float(threshold) * 255.0)) if float(threshold) <= 1.0 else int(round(float(threshold)))
    thr_val = max(0, min(255, thr_val))

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "x",
        "y",
        "w",
        "h",
        "t",
        "class",
        "xy_semantics",
        "firefly_logit",
        "background_logit",
        "firefly_confidence",
    ]

    fid = len(buffer)  # zero-based index of next frame to process

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        while fid < limit:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            g = frame[:, :, 1].astype(np.float32, copy=False)

            # Update rolling-mean background
            oldest = buffer[0]
            bkgr = bkgr + (g - oldest) / float(win_len)
            buffer.append(g)

            # Foreground
            fg = np.clip(g - bkgr, 0, 255).astype(np.uint8)
            if float(blur_sigma) > 0.2:
                fg = cv2.GaussianBlur(fg, (0, 0), float(blur_sigma))
            _, bw = cv2.threshold(fg, thr_val, 255, cv2.THRESH_BINARY)

            dets = _measure(bw, g)
            for d in dets:
                conf = float(d["conf"])
                w.writerow(
                    {
                        "x": float(d["x"]),
                        "y": float(d["y"]),
                        "w": int(box_w),
                        "h": int(box_h),
                        "t": int(fid),
                        "class": "firefly",
                        "xy_semantics": "center",
                        "firefly_logit": _log_prob(conf),
                        "background_logit": _log_prob(1.0 - conf),
                        "firefly_confidence": conf,
                    }
                )

            fid += 1

    cap.release()
    print(f"[lab-baseline] Wrote predictions CSV → {out_csv}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Legacy lab baseline: mp4 -> Stage-5 predictions CSV.")
    p.add_argument("--video", type=str, required=True, help="Input video path.")
    p.add_argument("--out-csv", type=str, required=True, help="Output predictions CSV.")
    p.add_argument("--threshold", type=float, default=0.12, help="Binary threshold (0..1). Default 0.12.")
    p.add_argument("--blur-sigma", type=float, default=1.0, help="Gaussian blur sigma (px).")
    p.add_argument("--bkgr-window-sec", type=float, default=2.0, help="Rolling background window (sec).")
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames processed.")
    p.add_argument("--box-w", type=int, default=10, help="Box width in CSV (px).")
    p.add_argument("--box-h", type=int, default=10, help="Box height in CSV (px).")
    return p.parse_args()


def main() -> int:
    a = _parse_args()
    video = Path(a.video).expanduser().resolve()
    out_csv = Path(a.out_csv).expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    run(
        video_path=video,
        out_csv=out_csv,
        threshold=float(a.threshold),
        blur_sigma=float(a.blur_sigma),
        bkgr_window_sec=float(a.bkgr_window_sec),
        max_frames=int(a.max_frames) if a.max_frames is not None else None,
        box_w=int(a.box_w),
        box_h=int(a.box_h),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

