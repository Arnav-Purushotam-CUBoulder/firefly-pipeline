#!/usr/bin/env python3
"""
Run inference with a trained YOLO model on a long‑exposure image.
Saves an annotated image (red, 1px boxes) and a CSV with detections.

Edit the globals below, then run this file.

Requires: pip install ultralytics>=8.0.0 opencv-python
"""

from __future__ import annotations
from pathlib import Path
import csv
import sys

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Ultralytics is required. Install with: pip install ultralytics\n"
        f"Import error: {e}"
    )


# ===================== Globals (edit these) =====================

# Path to the trained weights (.pt)
MODEL_WEIGHTS = Path("/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/test1/pyrallis_exp/raw long exposure exp/yolo train output data/runs_firefly/train_20251030_2129592/weights/best.pt")

# Input long‑exposure image path
INPUT_IMAGE = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/500 frame each long exposure shots/20240606_cam1_GS010064/500_20240606_cam1_GS010064_lighten_000000-000499.png')

# Output directory (annotated image + CSV go here)
OUTPUT_DIR = Path("/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/test1/pyrallis_exp/raw long exposure exp/yolo infer output data")

# Inference params
IMG_SIZE: int | None = None  # None → auto from input image size (max(H,W))
CONF_THRES = 0.1
IOU_THRES = 0.15  # NMS
DEVICE: str | int | None = "cpu"  # 'auto'|'cpu'|'mps'|CUDA index

# Drawing params
BOX_COLOR = (0, 0, 255)  # BGR red
BOX_THICKNESS = 1


# ===================== Impl =====================

def _select_device(user_choice: str | int | None):
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

def _ensure_paths():
    if not Path(MODEL_WEIGHTS).exists():
        raise FileNotFoundError(f"MODEL_WEIGHTS not found: {MODEL_WEIGHTS}")
    if not Path(INPUT_IMAGE).exists():
        raise FileNotFoundError(f"INPUT_IMAGE not found: {INPUT_IMAGE}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _draw_boxes(image: np.ndarray, boxes_xyxy: np.ndarray, color=(0, 0, 255), thickness=1) -> np.ndarray:
    out = image.copy()
    for x1, y1, x2, y2 in boxes_xyxy.astype(int):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, int(thickness))
    return out


def main():
    _ensure_paths()

    model = YOLO(str(MODEL_WEIGHTS))
    # Ultralytics returns a list of Results; use stream=False
    # Determine image size if None
    img = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {INPUT_IMAGE}")
    if IMG_SIZE is None:
        h, w = img.shape[:2]
        imgsz = max(h, w)
        print(f"Auto image size: {w}x{h} → imgsz={imgsz}")
    else:
        imgsz = int(IMG_SIZE)
    results = model.predict(
        source=str(INPUT_IMAGE),
        imgsz=imgsz,
        conf=float(CONF_THRES),
        iou=float(IOU_THRES),
        device=_select_device(DEVICE),
        verbose=False,
        stream=False,
    )

    if not results:
        raise RuntimeError("No results returned by model.predict")
    r = results[0]

    # Boxes to numpy (x1,y1,x2,y2), scores, class ids
    boxes_xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r, "boxes") else np.zeros((0, 4), dtype=np.float32)
    scores = r.boxes.conf.cpu().numpy() if hasattr(r, "boxes") else np.zeros((0,), dtype=np.float32)
    cls_ids = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r, "boxes") else np.zeros((0,), dtype=int)

    annotated = _draw_boxes(img, boxes_xyxy, color=BOX_COLOR, thickness=BOX_THICKNESS)

    # Save annotated image
    out_img = Path(OUTPUT_DIR) / f"{Path(INPUT_IMAGE).stem}_pred.png"
    cv2.imwrite(str(out_img), annotated)

    # Save CSV in x,y,w,h,image_name format (top-left x,y)
    rows = []
    for i in range(boxes_xyxy.shape[0]):
        x1, y1, x2, y2 = boxes_xyxy[i]
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        rows.append({
            "x": int(round(x1)),
            "y": int(round(y1)),
            "w": int(round(w)),
            "h": int(round(h)),
            "image_name": Path(INPUT_IMAGE).name,
        })

    out_csv = Path(OUTPUT_DIR) / f"{Path(INPUT_IMAGE).stem}_pred.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "w", "h", "image_name"])
        writer.writeheader()
        writer.writerows(rows)

    # Console summary
    print(f"Saved annotated image → {out_img}")
    print(f"Saved CSV → {out_csv}  (detections: {len(rows)})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
