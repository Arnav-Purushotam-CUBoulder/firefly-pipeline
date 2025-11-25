#!/usr/bin/env python3
"""
Stage 2: YOLO detection on long-exposure images.

For each long-exposure image produced in Stage 1, run a YOLO model to detect
streaks and aggregate all detections into a single CSV per source video with
schema:

  x, y, w, h, frame_range, video_name

where frame_range is taken from the long-exposure filename and video_name is
the original input video filename.
"""
from __future__ import annotations

from pathlib import Path
import csv

import cv2
import numpy as np

import params

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Stage 2 YOLO detection requires ultralytics. Install with: pip install ultralytics\n"
        f"Import error: {e}"
    ) from e


def _draw_boxes(image: np.ndarray, boxes_xyxy: np.ndarray, color=(0, 0, 255), thickness: int = 1) -> np.ndarray:
    """Return a copy of image with axis-aligned boxes drawn."""
    out = image.copy()
    for x1, y1, x2, y2 in boxes_xyxy.astype(int):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, int(thickness), cv2.LINE_AA)
    return out


def _select_device(user_choice: str | int | None):
    """Resolve device choice, mirroring infer_yolo_firefly.py behaviour."""
    import torch

    if isinstance(user_choice, int):
        return user_choice
    if isinstance(user_choice, str) and user_choice.lower() not in {"auto", "", "none"}:
        return user_choice
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return 0
    return "cpu"


def _parse_meta_from_image_path(img_path: Path) -> dict:
    """Parse interval, video_stem, mode, start, end from parent folder or filename.

    Expects pattern:
      <interval>_<video_stem>_<mode>_<start>-<end>
    where mode is one of lighten|average|trails.
    """
    import re

    name_candidates = [img_path.parent.name, img_path.stem]
    pat = re.compile(
        r"^(?P<interval>\d+?)_(?P<video>.+?)_(?P<mode>lighten|average|trails)_(?P<start>\d+)-(?P<end>\d+)$"
    )
    for s in name_candidates:
        m = pat.match(s)
        if m:
            d = m.groupdict()
            return {
                "interval": int(d["interval"]),
                "video_stem": d["video"],
                "mode": d["mode"],
                "start": int(d["start"]),
                "end": int(d["end"]),
            }
    raise ValueError(
        f"Could not parse start/end from '{img_path}'. Expected name like '<interval>_<video>_<mode>_<start>-<end>'."
    )


def run_for_video(video_path: Path) -> Path:
    """Run YOLO on all Stage 1 long-exposure images for a single video.

    Returns path to the aggregated CSV under STAGE2_DIR/<video_stem>/<video_stem>.csv
    """
    assert video_path.exists(), f"Video not found: {video_path}"
    video_stem = video_path.stem

    stage1_dir = params.STAGE1_DIR / video_stem
    if not stage1_dir.exists():
        raise FileNotFoundError(f"Stage 1 output directory not found: {stage1_dir}")

    out_root = params.STAGE2_DIR / video_stem
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect long-exposure images (PNG/JPEG)
    images = [
        p
        for p in sorted(stage1_dir.iterdir())
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not images:
        print(f"[stage2_yolo_detect] No long-exposure images found in {stage1_dir}")
        csv_path = out_root / f"{video_stem}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "w", "h", "frame_range", "video_name"])
        return csv_path

    weights = Path(params.YOLO_MODEL_WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(f"YOLO_MODEL_WEIGHTS not found: {weights}")

    model = YOLO(str(weights))

    # Decide image size once based on the first image if not provided
    img_size_cfg = getattr(params, "YOLO_IMG_SIZE", None)
    if img_size_cfg is None:
        img0 = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if img0 is None:
            raise RuntimeError(f"Failed to read image: {images[0]}")
        h0, w0 = img0.shape[:2]
        imgsz = max(h0, w0)
        print(f"[stage2_yolo_detect] Auto image size from {images[0].name}: {w0}x{h0} → imgsz={imgsz}")
    else:
        imgsz = int(img_size_cfg)

    rows: list[dict] = []
    device = _select_device(getattr(params, "YOLO_DEVICE", "cpu"))
    conf_thres = float(getattr(params, "YOLO_CONF_THRES", 0.1))
    iou_thres = float(getattr(params, "YOLO_IOU_THRES", 0.15))

    annotated_dir = out_root / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        meta = _parse_meta_from_image_path(img_path)
        frame_range = f"{meta['start']}-{meta['end']}"

        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            verbose=False,
            stream=False,
        )
        if not results:
            continue
        r = results[0]

        boxes_xyxy = (
            r.boxes.xyxy.cpu().numpy()
            if hasattr(r, "boxes") and getattr(r, "boxes", None) is not None
            else np.zeros((0, 4), dtype=np.float32)
        )

        # Save annotated long-exposure image with boxes drawn
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            annotated = _draw_boxes(img, boxes_xyxy)
            out_img = annotated_dir / f"{img_path.stem}_pred.png"
            cv2.imwrite(str(out_img), annotated)

        for i in range(boxes_xyxy.shape[0]):
            x1, y1, x2, y2 = boxes_xyxy[i]
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            rows.append(
                {
                    "x": int(round(x1)),
                    "y": int(round(y1)),
                    "w": int(round(w)),
                    "h": int(round(h)),
                    "frame_range": frame_range,
                    "video_name": video_path.name,
                }
            )

    csv_path = out_root / f"{video_stem}.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["x", "y", "w", "h", "frame_range", "video_name"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[stage2_yolo_detect] Saved {len(rows)} detection(s) for {video_path.name} → {csv_path}"
    )
    return csv_path


__all__ = ["run_for_video"]
