#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2

import params
from utils import open_video, make_writer, progress


def run_for_video(video_path: Path) -> Path:
    """Render the final video with boxes drawn from Stage 3 CSV."""
    stem = video_path.stem
    s3_csv = (params.STAGE3_DIR / stem) / f"{stem}_gauss.csv"
    assert s3_csv.exists(), f"Missing Stage3 CSV for {stem}: {s3_csv}"

    out_root = params.STAGE4_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{stem}_detections.mp4"

    # Read boxes by frame
    boxes_by_t: dict[int, list[tuple[float, float, int, int]]] = defaultdict(list)
    with s3_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cx = float(row["x"])
            cy = float(row["y"])
            # Use global patch size everywhere (ignore CSV w,h)
            w = int(params.PATCH_SIZE_PX)
            h = int(params.PATCH_SIZE_PX)
            t = int(row["frame_number"])  # 0-based
            boxes_by_t[t].append((cx, cy, w, h))

    cap, W, H, fps_src, total = open_video(video_path)
    max_frames = int(params.MAX_FRAMES) if (params.MAX_FRAMES is not None) else total
    fps = float(params.RENDER_FPS_HINT or fps_src)
    writer = make_writer(out_path, W, H, fps, codec=params.RENDER_CODEC, is_color=True)

    t = 0
    try:
        while t < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if t in boxes_by_t:
                for cx, cy, w, h in boxes_by_t[t]:
                    x0 = int(round(cx - w / 2.0))
                    y0 = int(round(cy - h / 2.0))
                    x1 = x0 + int(w)
                    y1 = y0 + int(h)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            t += 1
            if t % 50 == 0:
                progress(t, max_frames or t, "Stage4 render")
        progress(t, max_frames or t or 1, "Stage4 render done")
    finally:
        cap.release()
        writer.release()

    # --- Stats summary ---
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
        # top frames by boxes
        top = sorted(((ti, len(lst)) for ti, lst in boxes_by_t.items()), key=lambda x: (-x[1], x[0]))[:5]
        top_s = ", ".join([f"t={ti}:{c}" for (ti, c) in top])
        print(f"Stage4  Top frames by boxes: {top_s}")
    if size_mb is not None:
        print(f"Stage4  Output size: {size_mb:.2f} MB")
    print(f"Stage4  Wrote rendered video → {out_path}")
    return out_path


__all__ = ["run_for_video"]
