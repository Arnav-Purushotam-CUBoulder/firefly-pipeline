#!/usr/bin/env python3
"""
Render a Stage-5-style baseline predictions CSV back onto the source video.

This is shared by the legacy Lab and Raphael baselines so both methods can
produce a watchable annotated output video from the same CSV schema.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _read_predictions(pred_csv: Path) -> Dict[int, List[dict]]:
    rows_by_t: Dict[int, List[dict]] = defaultdict(list)
    if not pred_csv.exists():
        raise FileNotFoundError(pred_csv)

    with pred_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t_raw = row.get("t", row.get("frame_idx", row.get("frame_number", "")))
                t = int(float(t_raw))
                x = float(row.get("x", ""))
                y = float(row.get("y", ""))
            except Exception:
                continue

            try:
                w = int(round(float(row.get("w", "10") or 10)))
            except Exception:
                w = 10
            try:
                h = int(round(float(row.get("h", "10") or 10)))
            except Exception:
                h = 10

            conf = None
            try:
                conf_raw = row.get("firefly_confidence", "")
                conf = float(conf_raw) if str(conf_raw).strip() else None
            except Exception:
                conf = None

            rows_by_t[t].append(
                {
                    "x": float(x),
                    "y": float(y),
                    "w": int(max(1, w)),
                    "h": int(max(1, h)),
                    "xy_semantics": str(row.get("xy_semantics", "center") or "center").strip().lower(),
                    "conf": conf,
                }
            )

    return rows_by_t


def _open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, width, height, fps, frame_count


def _make_writer(out_path: Path, width: int, height: int, fps: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")
    return writer


def _to_box(det: dict) -> tuple[int, int, int, int]:
    x = float(det["x"])
    y = float(det["y"])
    w = int(det["w"])
    h = int(det["h"])
    semantics = str(det.get("xy_semantics", "center") or "center").strip().lower()
    if semantics == "center":
        x0 = int(round(x - (w / 2.0)))
        y0 = int(round(y - (h / 2.0)))
    else:
        x0 = int(round(x))
        y0 = int(round(y))
    return x0, y0, x0 + w, y0 + h


def render_predictions(
    *,
    video_path: Path,
    pred_csv: Path,
    out_video: Path,
    label: str,
    max_frames: int | None,
) -> Path:
    rows_by_t = _read_predictions(pred_csv)
    cap, width, height, fps, frame_count = _open_video(video_path)
    limit = frame_count
    if max_frames is not None:
        limit = min(limit, int(max_frames))
    if limit <= 0:
        limit = frame_count

    writer = _make_writer(out_video, width, height, fps)
    frame_idx = 0

    try:
        while True:
            if max_frames is not None and frame_idx >= int(max_frames):
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            dets = rows_by_t.get(frame_idx) or []
            for det in dets:
                x0, y0, x1, y1 = _to_box(det)
                x0 = max(0, min(x0, width - 1))
                y0 = max(0, min(y0, height - 1))
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
                conf = det.get("conf")
                if conf is not None:
                    text = f"{float(conf):.2f}"
                    text_y = max(14, y0 - 4)
                    cv2.putText(frame, text, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            header = f"{label} | frame={frame_idx} | detections={len(dets)}"
            cv2.putText(frame, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"[baseline-render] Video: {video_path}")
    print(f"[baseline-render] Predictions CSV: {pred_csv}")
    print(f"[baseline-render] Frames rendered: {frame_idx}")
    print(f"[baseline-render] Output video: {out_video}")
    return out_video


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render baseline predictions CSV onto the source video.")
    p.add_argument("--video", type=str, required=True, help="Input video path.")
    p.add_argument("--pred-csv", type=str, required=True, help="Stage-5-style predictions CSV.")
    p.add_argument("--out-video", type=str, required=True, help="Output rendered video path.")
    p.add_argument("--label", type=str, default="baseline", help="Short overlay label for the video header.")
    p.add_argument("--max-frames", type=int, default=None, help="Optional frame cap.")
    return p.parse_args()


def main() -> int:
    a = _parse_args()
    video = Path(a.video).expanduser().resolve()
    pred_csv = Path(a.pred_csv).expanduser().resolve()
    out_video = Path(a.out_video).expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    if video.suffix.lower() not in VIDEO_EXTS:
        raise SystemExit(f"Unsupported video extension: {video.suffix}")
    if not pred_csv.exists():
        raise SystemExit(f"Predictions CSV not found: {pred_csv}")

    render_predictions(
        video_path=video,
        pred_csv=pred_csv,
        out_video=out_video,
        label=str(a.label),
        max_frames=int(a.max_frames) if a.max_frames is not None else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
