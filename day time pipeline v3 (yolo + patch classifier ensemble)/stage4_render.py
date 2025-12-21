#!/usr/bin/env python3
from __future__ import annotations

"""
Stage 4: render video with patch-classifier boxes overlaid.

Uses the Stage 3 CSV (per-frame positive crops) for a video and writes a
rendered MP4 with all boxes drawn on the original frames.
"""

import csv
from collections import defaultdict
from pathlib import Path

import cv2

import params


def _progress(i: int, total: int, tag: str = "") -> None:
    total = max(int(total or 1), 1)
    i = min(int(i), total)
    bar_w = 36
    frac = i / total
    fill = int(frac * bar_w)
    bar = "█" * fill + "·" * (bar_w - fill)
    print(f"\r[{bar}] {i}/{total} {tag}", end="")
    if i == total:
        print("")


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


def _make_writer(path: Path, w: int, h: int, fps: float, codec: str = "mp4v", is_color: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(w), int(h)), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


def run_for_video(video_path: Path) -> Path:
    """Render final video with boxes drawn from Stage 3 CSV."""
    stem = video_path.stem
    s3_dir = params.STAGE3_DIR / stem
    # Prefer Stage3.1 curvy-filtered CSV if present; else fall back to raw Stage3.
    preferred = s3_dir / f"{stem}_patches_curvy.csv"
    fallback = s3_dir / f"{stem}_patches.csv"
    s3_csv = preferred if preferred.exists() else fallback
    assert s3_csv.exists(), f"Missing Stage3 CSV for {stem}: {s3_csv}"

    out_root = params.STAGE4_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{stem}_patches.mp4"

    # Read boxes by frame (top-left x,y,w,h)
    boxes_by_t: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    with s3_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = int(row.get("frame_idx") or row.get("frame_number"))
                x = int(float(row["x"]))
                y = int(float(row["y"]))
                w = int(float(row.get("w", params.PATCH_SIZE_PX)))
                h = int(float(row.get("h", params.PATCH_SIZE_PX)))
            except Exception:
                continue
            boxes_by_t[t].append((x, y, w, h))

    cap, W, H, fps_src, total = _open_video(video_path)
    max_frames = int(params.MAX_FRAMES) if (params.MAX_FRAMES is not None) else total
    fps = float(params.RENDER_FPS_HINT or fps_src)
    writer = _make_writer(out_path, W, H, fps, codec=params.RENDER_CODEC, is_color=True)

    t = 0
    try:
        while t < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if t in boxes_by_t:
                for (x, y, w, h) in boxes_by_t[t]:
                    x0 = int(x)
                    y0 = int(y)
                    x1 = x0 + int(w)
                    y1 = y0 + int(h)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            t += 1
            if t % 50 == 0:
                _progress(t, max_frames or t, "Stage5 render")
        _progress(t, max_frames or t or 1, "Stage5 render done")
    finally:
        cap.release()
        writer.release()

    # Stats summary
    frames_in_csv = len(boxes_by_t)
    total_boxes = sum(len(v) for v in boxes_by_t.values())
    avg_boxes_per_frame = (total_boxes / max(1, max_frames)) if max_frames else 0.0
    from os import stat as _stat

    try:
        size_mb = _stat(out_path).st_size / (1024 * 1024)
    except Exception:
        size_mb = None

    print(f"Stage4  Frames rendered: {t} of {max_frames}; frames_with_boxes: {frames_in_csv}")
    print(f"Stage4  Boxes drawn: total={total_boxes}; avg/frame={avg_boxes_per_frame:.2f}")
    if total_boxes:
        top = sorted(
            ((ti, len(lst)) for ti, lst in boxes_by_t.items()),
            key=lambda x: (-x[1], x[0]),
        )[:5]
        top_s = ", ".join([f"t={ti}:{c}" for (ti, c) in top])
        print(f"Stage4  Top frames by boxes: {top_s}")
    if size_mb is not None:
        print(f"Stage4  Output size: {size_mb:.2f} MB")
    print(f"Stage4  Wrote rendered video → {out_path}")
    return out_path


__all__ = ["run_for_video"]
