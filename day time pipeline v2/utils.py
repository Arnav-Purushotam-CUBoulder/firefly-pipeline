#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def progress(i: int, total: int, tag: str = "") -> None:
    total = max(int(total or 1), 1)
    i = min(int(i), total)
    bar_w = 36
    frac = i / total
    fill = int(frac * bar_w)
    bar = "█" * fill + "·" * (bar_w - fill)
    sys.stdout.write(f"\r[{bar}] {i}/{total} {tag}")
    if i == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def purge_dir(p: Path) -> None:
    import shutil
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass
    p.mkdir(parents=True, exist_ok=True)


def open_video(path: Path) -> Tuple[cv2.VideoCapture, int, int, float, int]:
    assert path.exists(), f"Input not found: {path}"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, w, h, fps, count


def make_writer(path: Path, w: int, h: int, fps: float, codec: str = "mp4v", is_color: bool = True) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(w), int(h)), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


def center_crop_with_pad(img: np.ndarray, cx: float, cy: float, size_w: int, size_h: int) -> tuple[np.ndarray, int, int]:
    H, W = img.shape[:2]
    w = max(1, int(size_w))
    h = max(1, int(size_h))
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x1 = x0 + w
    y1 = y0 + h

    vx0 = max(0, x0)
    vy0 = max(0, y0)
    vx1 = min(W, x1)
    vy1 = min(H, y1)

    px0 = vx0 - x0
    py0 = vy0 - y0

    if img.ndim == 3:
        patch = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
        patch[py0:py0 + (vy1 - vy0), px0:px0 + (vx1 - vx0), :] = img[vy0:vy1, vx0:vx1, :]
    else:
        patch = np.zeros((h, w), dtype=img.dtype)
        patch[py0:py0 + (vy1 - vy0), px0:px0 + (vx1 - vx0)] = img[vy0:vy1, vx0:vx1]
    return patch, x0, y0
