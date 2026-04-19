#!/usr/bin/env python3
"""
Stage 1: create long-exposure images from raw videos.

For each input video under params.ORIGINAL_VIDEOS_DIR, this stage writes one or
more long-exposure PNGs under:

  params.STAGE1_DIR / <video_stem> / <interval>_<video_stem>_<mode>_<start>-<end>.png

The naming convention matches the standalone long_exposure_from_raw_video.py
script so that frame ranges can be parsed later by Stage 2.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import params


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
            if seen % max(1, int(getattr(params, "PROGRESS_EVERY", 50))) == 0:
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

        def to_lin(im: np.ndarray) -> np.ndarray:
            return np.power(im.astype(np.float32) / 255.0, float(gamma))

        def to_srgb(im: np.ndarray) -> np.ndarray:
            return (np.power(np.clip(im, 0, 1), inv_gamma) * 255.0).astype(np.uint8)

        accum = to_lin(frame0)
    else:
        to_srgb = None  # type: ignore[assignment]
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
            if seen % max(1, int(getattr(params, "PROGRESS_EVERY", 50))) == 0:
                _progress(seen, count or seen, f"average {start}")
        _progress(seen, max(1, count), f"average {start} done")
    finally:
        cap.release()

    avg = accum / max(1, seen)
    if use_gamma:
        return to_srgb(avg)  # type: ignore[arg-type]
    return np.clip(avg, 0, 255).astype(np.uint8)


def _trails_range(
    video_path: Path,
    start: int,
    count: int,
    detect_shadows: bool,
    history: int,
    learning_rate: float,
    fg_thresh: int,
    blur_ksize: int,
    dilate_iters: int,
    dilate_kernel: int,
    overlay: bool,
    overlay_alpha: float,
) -> np.ndarray | None:
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
            if seen % max(1, int(getattr(params, "PROGRESS_EVERY", 50))) == 0:
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


def run_for_video(video_path: Path) -> list[Path]:
    """Render long-exposure image(s) for a single video.

    Returns a list of output image paths under STAGE1_DIR/<video_stem>/.
    """
    assert video_path.exists(), f"Video not found: {video_path}"

    mode = str(getattr(params, "LONG_EXPOSURE_MODE", "lighten")).lower().strip()
    if mode not in {"lighten", "average", "trails"}:
        raise ValueError("LONG_EXPOSURE_MODE must be one of: 'lighten', 'average', 'trails'")

    max_frames_cfg = getattr(params, "MAX_FRAMES", None)
    max_frames = int(max_frames_cfg) if max_frames_cfg is not None else None
    interval_cfg = getattr(params, "INTERVAL_FRAMES", None)

    video_stem = video_path.stem
    out_video_dir = params.STAGE1_DIR / video_stem
    out_video_dir.mkdir(parents=True, exist_ok=True)

    # Determine frame limit and interval
    cap, W, H, fps, total_est = _open_video(video_path)
    cap.release()
    limit = int(total_est)
    if max_frames is not None:
        limit = min(limit, int(max_frames))
    if limit <= 0:
        print(f"[stage1_long_exposure] No frames to process for {video_path.name}")
        return []

    out_images: list[Path] = []

    def _render_chunk(start: int, count: int) -> np.ndarray | None:
        if mode == "lighten":
            return _lighten_range(video_path, start, count)
        if mode == "average":
            return _average_range(
                video_path,
                start,
                count,
                use_gamma=bool(getattr(params, "AVERAGE_USE_GAMMA", False)),
                gamma=float(getattr(params, "AVERAGE_GAMMA", 2.2)),
            )
        if mode == "trails":
            return _trails_range(
                video_path,
                start,
                count,
                detect_shadows=bool(getattr(params, "BGS_DETECT_SHADOWS", True)),
                history=int(getattr(params, "BGS_HISTORY", 1000)),
                learning_rate=float(getattr(params, "BGS_LEARNING_RATE", -1.0)),
                fg_thresh=int(getattr(params, "FG_MASK_THRESHOLD", 200)),
                blur_ksize=int(getattr(params, "TRAILS_BLUR_KSIZE", 0)),
                dilate_iters=int(getattr(params, "TRAILS_DILATE_ITERS", 1)),
                dilate_kernel=int(getattr(params, "TRAILS_DILATE_KERNEL", 3)),
                overlay=bool(getattr(params, "TRAILS_OVERLAY", False)),
                overlay_alpha=float(getattr(params, "TRAILS_OVERLAY_ALPHA", 0.7)),
            )
        raise AssertionError("unreachable")

    interval = int(interval_cfg) if interval_cfg is not None else 0

    # Single full-image mode (no interval)
    if interval <= 0:
        start = 0
        count = limit
        end = start + count
        interval_size = count
        out_path = out_video_dir / f"{interval_size}_{video_stem}_{mode}_{start:06d}-{end-1:06d}.png"
        img = _render_chunk(start, count)
        if img is None:
            print(f"[stage1_long_exposure] Skipping {video_path.name}: failed to read frames")
        else:
            ok = cv2.imwrite(str(out_path), img)
            if not ok:
                raise RuntimeError(f"Failed to write output image: {out_path}")
            print(f"[stage1_long_exposure] Saved long-exposure image → {out_path}")
            out_images.append(out_path)
    else:
        interval = max(1, interval)
        start = 0
        while start < limit:
            count = min(interval, limit - start)
            end = start + count
            out_path = out_video_dir / f"{interval}_{video_stem}_{mode}_{start:06d}-{end-1:06d}.png"
            img = _render_chunk(start, count)
            if img is None:
                print(f"[stage1_long_exposure] Skipping chunk {start}-{end-1}: failed to read frames")
            else:
                ok = cv2.imwrite(str(out_path), img)
                if not ok:
                    raise RuntimeError(f"Failed to write output image: {out_path}")
                print(f"[stage1_long_exposure] Saved chunk image → {out_path}")
                out_images.append(out_path)
            start = end

    print(f"[stage1_long_exposure] Done. Saved {len(out_images)} image(s) for {video_path.name}")
    return out_images


__all__ = ["run_for_video"]

