#!/usr/bin/env python3
"""
Create a long-exposure image from a video using one of three modes:
- "lighten": channel-wise maximum (light trails)
- "average": temporal average (motion blur look)
- "trails": background-subtracted foreground OR (moving objects only)

Edit the globals below (MODE, INPUT_VIDEO_PATH, OUTPUT_DIR, etc.) and run.
You can also set INTERVAL_FRAMES to produce one long-exposure image for
every N-frame chunk of the video (e.g., 100 → one image per 100 frames).
Outputs are written under OUTPUT_DIR/<video_stem>/ as PNG files named:
  <interval>_<video_stem>_<mode>_<start>-<end>.png
Dependencies: pip install opencv-python numpy
"""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

# ========= Global configuration =========

# Choose one: "lighten", "average", "trails"
MODE = "lighten"

# Input/output paths
INPUT_VIDEO_PATH = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/raw input videos/20240606_cam1_GS010064.mp4')
OUTPUT_DIR = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/long exposure shots')

# Processing caps and progress
MAX_FRAMES: int | None = None     # None = full video
INTERVAL_FRAMES: int | None = 100  # None/<=0 → single image; else chunk size
PROGRESS_EVERY = 50

# Average-mode options
AVERAGE_USE_GAMMA = False         # if True, average in linear space (gamma ~2.2)
AVERAGE_GAMMA = 2.2

# Trails-mode options
BGS_DETECT_SHADOWS = True
BGS_HISTORY = 1000
BGS_LEARNING_RATE = -1.0          # -1 => OpenCV decides
FG_MASK_THRESHOLD = 200            # ignore shadows=127
TRAILS_DILATE_ITERS = 1            # 0=off
TRAILS_DILATE_KERNEL = 3           # odd size
TRAILS_BLUR_KSIZE = 0              # 0=off; else odd size
TRAILS_OVERLAY = False             # overlay trails on first frame
TRAILS_OVERLAY_ALPHA = 0.70


# ========= Helpers =========

def _open_video(path: Path):
    assert path.exists(), f"Input not found: {path}"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, w, h, fps, count


def _progress(i: int, total: int, tag: str = ""):
    total = max(int(total or 1), 1)
    i = min(i, total)
    bar_w = 36
    frac = i / total
    fill = int(frac * bar_w)
    bar = "█" * fill + "·" * (bar_w - fill)
    print(f"\r[{bar}] {i}/{total} {tag}", end="")
    if i == total:
        print("")


# ========= Methods =========

def long_exposure_lighten(video_path: Path, max_frames: int | None = None) -> np.ndarray:
    cap, W, H, fps, total_est = _open_video(video_path)
    ok, accum = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read first frame")

    total = total_est
    seen = 1
    try:
        while True:
            if max_frames is not None and seen >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            accum = np.maximum(accum, frame)
            seen += 1
            if seen % max(1, PROGRESS_EVERY) == 0:
                _progress(seen, total or seen, "lighten")
        _progress(seen, seen, "lighten done")
    finally:
        cap.release()
    return accum


def long_exposure_average(video_path: Path, max_frames: int | None = None,
                          use_gamma: bool = False, gamma: float = 2.2) -> np.ndarray:
    cap, W, H, fps, total_est = _open_video(video_path)
    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read first frame")

    if use_gamma:
        inv_gamma = 1.0 / max(1e-6, float(gamma))
        to_lin = lambda im: np.power(im.astype(np.float32) / 255.0, float(gamma))
        to_srgb = lambda im: (np.power(np.clip(im, 0, 1), inv_gamma) * 255.0).astype(np.uint8)
        accum = to_lin(frame0)
    else:
        accum = frame0.astype(np.float32)

    total = total_est
    seen = 1
    try:
        while True:
            if max_frames is not None and seen >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if use_gamma:
                accum += to_lin(frame)
            else:
                accum += frame.astype(np.float32)
            seen += 1
            if seen % max(1, PROGRESS_EVERY) == 0:
                _progress(seen, total or seen, "average")
        _progress(seen, seen, "average done")
    finally:
        cap.release()

    avg = accum / max(1, seen)
    if use_gamma:
        return to_srgb(avg)
    return np.clip(avg, 0, 255).astype(np.uint8)


def long_exposure_trails(video_path: Path, max_frames: int | None = None,
                         detect_shadows: bool = True, history: int = 1000,
                         learning_rate: float = -1.0, fg_thresh: int = 200,
                         blur_ksize: int = 0, dilate_iters: int = 1, dilate_kernel: int = 3,
                         overlay: bool = False, overlay_alpha: float = 0.7) -> np.ndarray:
    cap, W, H, fps, total_est = _open_video(video_path)
    bgs = cv2.createBackgroundSubtractorMOG2(detectShadows=bool(detect_shadows), history=int(history))

    ok, base = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read first frame")

    trails = np.zeros((H, W), dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total = total_est
    seen = 0

    kernel = None
    if dilate_iters and dilate_iters > 0:
        k = int(dilate_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    try:
        while True:
            if max_frames is not None and seen >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            fg = bgs.apply(frame, learningRate=learning_rate)
            if blur_ksize and blur_ksize > 1:
                k = int(blur_ksize)
                if k % 2 == 0:
                    k += 1
                fg = cv2.GaussianBlur(fg, (k, k), 0)
            mask = (fg >= int(fg_thresh)).astype(np.uint8) * 255
            if kernel is not None and dilate_iters > 0:
                mask = cv2.dilate(mask, kernel, iterations=int(dilate_iters))
            trails = cv2.bitwise_or(trails, mask)
            seen += 1
            if seen % max(1, PROGRESS_EVERY) == 0:
                _progress(seen, total or seen, "trails")
        _progress(seen, seen, "trails done")
    finally:
        cap.release()

    if overlay:
        trails_rgb = cv2.merge([trails, trails, trails])
        out = cv2.addWeighted(base, 1.0, trails_rgb, float(overlay_alpha), 0.0)
        return out
    else:
        return cv2.merge([trails, trails, trails])


# ========= Chunked variants =========

def _lighten_range(video_path: Path, start: int, count: int) -> np.ndarray | None:
    cap, W, H, fps, total_est = _open_video(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    ok, accum = cap.read()
    if not ok:
        cap.release()
        return None
    seen = 1
    try:
        while seen < count:
            ok, frame = cap.read()
            if not ok:
                break
            accum = np.maximum(accum, frame)
            seen += 1
            if seen % max(1, PROGRESS_EVERY) == 0:
                _progress(seen, count or seen, f"lighten {start}")
        _progress(seen, max(1, count), f"lighten {start} done")
    finally:
        cap.release()
    return accum


def _average_range(video_path: Path, start: int, count: int, use_gamma: bool, gamma: float) -> np.ndarray | None:
    cap, W, H, fps, total_est = _open_video(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        return None
    if use_gamma:
        inv_gamma = 1.0 / max(1e-6, float(gamma))
        to_lin = lambda im: np.power(im.astype(np.float32) / 255.0, float(gamma))
        to_srgb = lambda im: (np.power(np.clip(im, 0, 1), inv_gamma) * 255.0).astype(np.uint8)
        accum = to_lin(frame0)
    else:
        accum = frame0.astype(np.float32)
    seen = 1
    try:
        while seen < count:
            ok, frame = cap.read()
            if not ok:
                break
            if use_gamma:
                accum += to_lin(frame)
            else:
                accum += frame.astype(np.float32)
            seen += 1
            if seen % max(1, PROGRESS_EVERY) == 0:
                _progress(seen, count or seen, f"average {start}")
        _progress(seen, max(1, count), f"average {start} done")
    finally:
        cap.release()
    avg = accum / max(1, seen)
    if use_gamma:
        return to_srgb(avg)
    return np.clip(avg, 0, 255).astype(np.uint8)


def _trails_range(video_path: Path, start: int, count: int,
                  detect_shadows: bool, history: int, learning_rate: float,
                  fg_thresh: int, blur_ksize: int, dilate_iters: int, dilate_kernel: int,
                  overlay: bool, overlay_alpha: float) -> np.ndarray | None:
    cap, W, H, fps, total_est = _open_video(video_path)
    bgs = cv2.createBackgroundSubtractorMOG2(detectShadows=bool(detect_shadows), history=int(history))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    ok, base = cap.read()
    if not ok:
        cap.release()
        return None
    trails = np.zeros((H, W), dtype=np.uint8)
    kernel = None
    if dilate_iters and dilate_iters > 0:
        k = int(dilate_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    seen = 0
    try:
        while seen < count:
            ok, frame = cap.read()
            if not ok:
                break
            fg = bgs.apply(frame, learningRate=learning_rate)
            if blur_ksize and blur_ksize > 1:
                k = int(blur_ksize)
                if k % 2 == 0:
                    k += 1
                fg = cv2.GaussianBlur(fg, (k, k), 0)
            mask = (fg >= int(fg_thresh)).astype(np.uint8) * 255
            if kernel is not None and dilate_iters > 0:
                mask = cv2.dilate(mask, kernel, iterations=int(dilate_iters))
            trails = cv2.bitwise_or(trails, mask)
            seen += 1
            if seen % max(1, PROGRESS_EVERY) == 0:
                _progress(seen, count or seen, f"trails {start}")
        _progress(seen, max(1, count), f"trails {start} done")
    finally:
        cap.release()
    if overlay:
        trails_rgb = cv2.merge([trails, trails, trails])
        out = cv2.addWeighted(base, 1.0, trails_rgb, float(overlay_alpha), 0.0)
        return out
    else:
        return cv2.merge([trails, trails, trails])


# ========= Main =========

def main():
    mode = str(MODE).lower().strip()
    in_path = INPUT_VIDEO_PATH
    out_root = OUTPUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    video_stem = in_path.stem
    out_video_dir = out_root / video_stem
    out_video_dir.mkdir(parents=True, exist_ok=True)

    # Single full-image mode (no interval)
    if not INTERVAL_FRAMES or INTERVAL_FRAMES <= 0:
        # Determine frame limit for naming
        cap, W, H, fps, total_est = _open_video(in_path)
        cap.release()
        limit = int(total_est)
        if MAX_FRAMES is not None:
            limit = min(limit, int(MAX_FRAMES))

        if mode == "lighten":
            img = long_exposure_lighten(in_path, MAX_FRAMES)
        elif mode == "average":
            img = long_exposure_average(in_path, MAX_FRAMES, use_gamma=AVERAGE_USE_GAMMA, gamma=AVERAGE_GAMMA)
        elif mode == "trails":
            img = long_exposure_trails(
                in_path,
                MAX_FRAMES,
                detect_shadows=BGS_DETECT_SHADOWS,
                history=BGS_HISTORY,
                learning_rate=BGS_LEARNING_RATE,
                fg_thresh=FG_MASK_THRESHOLD,
                blur_ksize=TRAILS_BLUR_KSIZE,
                dilate_iters=TRAILS_DILATE_ITERS,
                dilate_kernel=TRAILS_DILATE_KERNEL,
                overlay=TRAILS_OVERLAY,
                overlay_alpha=TRAILS_OVERLAY_ALPHA,
            )
        else:
            raise ValueError("MODE must be one of: 'lighten', 'average', 'trails'")
        # Build output filename: <interval>_<video_stem>_<mode>_<start>-<end>.png
        interval_size = limit  # for full-range single image
        start, end = 0, max(0, limit - 1)
        out_path = out_video_dir / f"{interval_size}_{video_stem}_{mode}_{start:06d}-{end:06d}.png"
        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            raise RuntimeError(f"Failed to write output image: {out_path}")
        print(f"Saved long-exposure image → {out_path}")
        return

    # Chunked mode
    cap, W, H, fps, total_est = _open_video(in_path)
    cap.release()
    limit = int(total_est)
    if MAX_FRAMES is not None:
        limit = min(limit, int(MAX_FRAMES))
    interval = int(max(1, INTERVAL_FRAMES))

    out_images = []
    start = 0
    while start < limit:
        count = min(interval, limit - start)
        end = start + count
        # Create filename: <interval>_<video_stem>_<mode>_<start>-<end-1>.png
        out_path = out_video_dir / f"{interval}_{video_stem}_{mode}_{start:06d}-{end-1:06d}.png"

        if mode == "lighten":
            img = _lighten_range(in_path, start, count)
        elif mode == "average":
            img = _average_range(in_path, start, count, use_gamma=AVERAGE_USE_GAMMA, gamma=AVERAGE_GAMMA)
        elif mode == "trails":
            img = _trails_range(
                in_path, start, count,
                detect_shadows=BGS_DETECT_SHADOWS,
                history=BGS_HISTORY,
                learning_rate=BGS_LEARNING_RATE,
                fg_thresh=FG_MASK_THRESHOLD,
                blur_ksize=TRAILS_BLUR_KSIZE,
                dilate_iters=TRAILS_DILATE_ITERS,
                dilate_kernel=TRAILS_DILATE_KERNEL,
                overlay=TRAILS_OVERLAY,
                overlay_alpha=TRAILS_OVERLAY_ALPHA,
            )
        else:
            raise ValueError("MODE must be one of: 'lighten', 'average', 'trails'")

        if img is None:
            print(f"Skipping chunk {start}-{end-1}: failed to read frames")
        else:
            ok = cv2.imwrite(str(out_path), img)
            if not ok:
                raise RuntimeError(f"Failed to write output image: {out_path}")
            print(f"Saved chunk image → {out_path}")
            out_images.append(out_path)

        start = end

    print(f"Done. Saved {len(out_images)} images from {in_path.name} with interval={interval} frames.")


if __name__ == "__main__":
    main()
