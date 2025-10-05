#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2

import params
from utils import open_video, make_writer, progress


def run_for_video(video_path: Path) -> Path:
    """Render the final video with boxes drawn from Stage 4 CSV."""
    stem = video_path.stem
    s4_csv = (params.STAGE4_DIR / stem) / f"{stem}_gauss.csv"
    assert s4_csv.exists(), f"Missing Stage4 CSV for {stem}: {s4_csv}"

    out_root = params.STAGE5_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{stem}_detections.mp4"

    # Read boxes by frame
    boxes_by_t: dict[int, list[tuple[float, float, int, int]]] = defaultdict(list)
    with s4_csv.open("r", newline="") as f:
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
                    cv2.rectangle(
                        frame,
                        (x0, y0),
                        (x1, y1),
                        (0, 0, 255),
                        int(getattr(params, 'RENDER_BOX_THICKNESS', 1)),
                        cv2.LINE_AA,
                    )
            writer.write(frame)
            t += 1
            if t % 50 == 0:
                progress(t, max_frames or t, "Stage5 render")
        progress(t, max_frames or t or 1, "Stage5 render done")
    finally:
        cap.release()
        writer.release()

    print(f"Stage5  Wrote rendered video → {out_path}")
    return out_path


__all__ = ["run_for_video"]

