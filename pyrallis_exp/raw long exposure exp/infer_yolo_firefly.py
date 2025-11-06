#!/usr/bin/env python3
"""
Run inference with a trained YOLO model on a long‑exposure image.
Saves an annotated image (red, 1px boxes) and a CSV with detections.

Optionally, for each YOLO bbox, scan the underlying source video over the
frame range encoded in the long‑exposure image name, find the brightest pixel
inside that bbox per frame, center a 10×10 crop there, classify with a binary
model, export a per‑frame CSV of positives, render an overlay video, and save
all positive 10×10 crops into a folder under OUTPUT_DIR for quick inspection.

Edit the globals below, then run this file.

Requires: pip install ultralytics>=8.0.0 opencv-python tqdm torch torchvision
"""

from __future__ import annotations
from pathlib import Path
import csv
import sys
from collections import defaultdict, Counter

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
INPUT_IMAGE = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/long exposure shots/20240606_cam1_GS010064/100_20240606_cam1_GS010064_lighten_000100-000199.png')

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


# ------------- Optional per-frame 10×10 filtering settings -------------
# The raw source video used to form the long-exposure image
SOURCE_VIDEO_PATH: Path | None = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/raw input videos/20240606_cam1_GS010064.mp4')

# Resize video frames to match long-exposure resolution before processing
RESIZE_TO_LONG = True

# Binary classifier (2-class) for 10×10 crops
CLASSIFIER_MODEL_PATH: Path | None = "/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/models and other data/pyrallis gopro models resnet18/resnet18_pyrallis_gopro_best_model v2.pt"  # e.g. Path("/path/to/resnet18_best.pt")
POSITIVE_CLASS_INDEX = 1
POSITIVE_THRESHOLD = 0.95
CLASSIFIER_BACKBONE = 'resnet18'  # one of: resnet18|resnet34|resnet50|resnet101|resnet152
CLASSIFIER_IMAGENET_NORM = False

# Output CSV path for per-frame positives (defaults to OUTPUT_DIR if None)
CSV_OUTPUT_PATH: Path | None = None

# Output overlay video path (defaults to OUTPUT_DIR if None)
VIDEO_OUTPUT_PATH: Path | None = None  # e.g. Path("./overlay.mp4")

# Also save each final positive 10×10 crop to a folder under OUTPUT_DIR
SAVE_POSITIVE_CROPS = True
POSITIVE_CROPS_DIR: Path | None = None  # None → OUTPUT_DIR/perframe_crops/<image_stem>/positives


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


def _parse_meta_from_image_path(img_path: Path) -> dict:
    """Parse interval, video_stem, mode, start, end from parent folder or filename.

    Expects pattern:
      <interval>_<video_stem>_<mode>_<start>-<end>
    where mode is one of lighten|average|trails.
    """
    import re
    name_candidates = [img_path.parent.name, img_path.stem]
    pat = re.compile(r"^(?P<interval>\d+?)_(?P<video>.+?)_(?P<mode>lighten|average|trails)_(?P<start>\d+)-(?P<end>\d+)$")
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


def _open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, W, H, count


def _centered_box(cx: int, cy: int, bw: int, bh: int, W: int, H: int) -> tuple[int, int, int, int]:
    """Return top-left x,y and w,h for a box of size (bw,bh) centered at (cx,cy) clipped inside WxH."""
    bw = int(bw); bh = int(bh)
    half_w = bw // 2
    half_h = bh // 2
    x = int(cx) - half_w
    y = int(cy) - half_h
    if x < 0: x = 0
    if y < 0: y = 0
    if x + bw > W: x = max(0, W - bw)
    if y + bh > H: y = max(0, H - bh)
    if W < bw: x = 0; bw = W
    if H < bh: y = 0; bh = H
    return int(x), int(y), int(bw), int(bh)


def _load_classifier(model_path: Path | None):
    """Load 2-class classifier and its transform. Returns (model, device, transform) or (None,None,None)."""
    if model_path is None:
        return None, None, None
    import torch
    import torch.nn as nn
    from torchvision import models, transforms as T

    dev = (
        torch.device("cuda") if torch.cuda.is_available()
        else (torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else torch.device("cpu"))
    )

    backbone_map = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    if CLASSIFIER_BACKBONE not in backbone_map:
        raise ValueError(f"Unknown CLASSIFIER_BACKBONE: {CLASSIFIER_BACKBONE}")
    model = backbone_map[CLASSIFIER_BACKBONE](weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(str(model_path), map_location=dev)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif hasattr(ckpt, 'state_dict'):
        state = ckpt.state_dict()
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith('module.') else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)
    model.eval().to(dev)

    tfms = [
        T.ToPILImage(),
        T.Resize((10, 10), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ]
    if CLASSIFIER_IMAGENET_NORM:
        tfms.append(T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform = T.Compose(tfms)
    return model, dev, transform


def _predict_patch(model, device, transform, patch_bgr: np.ndarray) -> float:
    import torch
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    x = transform(patch_rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return float(prob[int(POSITIVE_CLASS_INDEX)])


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

    # ================= Optional per-frame 10×10 filtering and overlay =================
    if SOURCE_VIDEO_PATH is not None and CLASSIFIER_MODEL_PATH is not None and len(rows) > 0:
        try:
            meta = _parse_meta_from_image_path(INPUT_IMAGE)
        except Exception as e:
            raise SystemExit(f"Failed to parse frame range from long-exposure name: {e}")
        start, end = int(meta['start']), int(meta['end'])
        n_frames = (end - start + 1)

        # Read YOLO boxes from the just-saved CSV
        yolo_boxes = []  # list of (x,y,w,h)
        with open(out_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                x = int(float(r['x'])); y = int(float(r['y']))
                w = int(float(r['w'])); h = int(float(r['h']))
                if w > 0 and h > 0:
                    yolo_boxes.append((x, y, w, h))

        if not yolo_boxes:
            print("No YOLO boxes to process for per-frame filtering.")
            return

        model_f, dev, transform = _load_classifier(CLASSIFIER_MODEL_PATH)
        if model_f is None:
            print("Model filtering skipped: CLASSIFIER_MODEL_PATH is not set.")
            return

        try:
            from tqdm import tqdm
        except Exception:
            def tqdm(x, **kwargs):
                return x

        cap, VW, VH, VCOUNT = _open_video(SOURCE_VIDEO_PATH)
        try:
            if VCOUNT > 0:
                end = min(end, VCOUNT - 1)
                n_frames = (end - start + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            # Load long-exposure to know target W,H (we already have img)
            H, W = img.shape[:2]

            rows_per_frame: list[dict] = []
            frame_boxes: dict[int, list[tuple[int,int,int,int]]] = defaultdict(list)
            pos_per_box = Counter()
            pos_per_frame = Counter()
            # Prepare directory for saving positive 10x10 crops
            saved_crops = 0
            if SAVE_POSITIVE_CROPS:
                pos_dir = POSITIVE_CROPS_DIR if POSITIVE_CROPS_DIR is not None else (Path(OUTPUT_DIR) / "perframe_crops" / Path(INPUT_IMAGE).stem / "positives")
                Path(pos_dir).mkdir(parents=True, exist_ok=True)

            print("Per-frame filtering: scanning frames and classifying 10x10 crops…")
            for idx in tqdm(range(start, end + 1), total=n_frames, ncols=100):
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != (W, H):
                    if not RESIZE_TO_LONG:
                        raise ValueError(
                            f"Video frame size {frame.shape[1]}x{frame.shape[0]} != long-exposure {W}x{H}. Enable RESIZE_TO_LONG."
                        )
                    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

                # For each YOLO box
                for bi, (x, y, w, h) in enumerate(yolo_boxes):
                    # clip box to image bounds to avoid indexing errors
                    x0 = max(0, int(x)); y0 = max(0, int(y))
                    x1 = min(W, int(x + w)); y1 = min(H, int(y + h))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    roi = frame[y0:y1, x0:x1]
                    if roi.size == 0:
                        continue
                    mean_roi = roi.mean(axis=2)
                    flat_idx = int(np.argmax(mean_roi))
                    by = int(flat_idx // (x1 - x0))
                    bx = int(flat_idx % (x1 - x0))
                    cx = x0 + bx
                    cy = y0 + by
                    rx, ry, rw, rh = _centered_box(cx, cy, 10, 10, W, H)
                    patch = frame[ry:ry + rh, rx:rx + rw]
                    if patch.shape[0] != 10 or patch.shape[1] != 10:
                        patch = cv2.resize(patch, (10, 10), interpolation=cv2.INTER_LINEAR)

                    p_pos = _predict_patch(model_f, dev, transform, patch)
                    if p_pos >= float(POSITIVE_THRESHOLD):
                        rec = {
                            'frame_idx': idx,
                            'video_name': Path(SOURCE_VIDEO_PATH).name,
                            'x': int(rx), 'y': int(ry), 'w': int(rw), 'h': int(rh),
                            'conf': float(p_pos),
                            'yolo_box_id': int(bi),
                            'long_exposure': Path(INPUT_IMAGE).name,
                        }
                        rows_per_frame.append(rec)
                        frame_boxes[idx].append((int(rx), int(ry), int(rw), int(rh)))
                        pos_per_box[bi] += 1
                        pos_per_frame[idx] += 1
                        if SAVE_POSITIVE_CROPS:
                            # Ensure crop is 10x10, then save it
                            crop10 = patch
                            if crop10.shape[:2] != (10, 10):
                                crop10 = cv2.resize(crop10, (10, 10), interpolation=cv2.INTER_LINEAR)
                            fname = f"f_{idx:06d}_x{rx}_y{ry}_w{rw}_h{rh}_p{p_pos:.3f}.png"
                            cv2.imwrite(str(pos_dir / fname), crop10)
                            saved_crops += 1

            # Save per-frame CSV
            csv2_path = CSV_OUTPUT_PATH if CSV_OUTPUT_PATH is not None else (Path(OUTPUT_DIR) / f"{Path(INPUT_IMAGE).stem}_perframe_candidates_thr{int(POSITIVE_THRESHOLD*100):02d}.csv")
            with open(csv2_path, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['frame_idx','video_name','x','y','w','h','conf','yolo_box_id','long_exposure']
                )
                writer.writeheader(); writer.writerows(rows_per_frame)
            print(f"Saved per-frame CSV → {csv2_path}  (positives: {len(rows_per_frame)})")

            # Rebuild frame_boxes from CSV (use as source-of-truth)
            frame_boxes = defaultdict(list)
            with open(csv2_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    idxr = int(r['frame_idx'])
                    xr = int(float(r['x'])); yr = int(float(r['y']))
                    wr = int(float(r['w'])); hr = int(float(r['h']))
                    frame_boxes[idxr].append((xr, yr, wr, hr))

            # Render overlay video
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vid_out_path = VIDEO_OUTPUT_PATH if VIDEO_OUTPUT_PATH is not None else (Path(OUTPUT_DIR) / f"{Path(INPUT_IMAGE).stem}_overlay.mp4")
            writer = cv2.VideoWriter(str(vid_out_path), fourcc, float(fps), (W, H))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            print("Rendering video with overlays…")
            for idx in tqdm(range(start, end + 1), total=n_frames, ncols=100):
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != (W, H):
                    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                for (rx, ry, rw, rh) in frame_boxes.get(idx, []):
                    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), BOX_COLOR, int(BOX_THICKNESS))
                writer.write(frame)
            writer.release()
            print(f"Saved overlay video → {vid_out_path}")

            # Stats
            total_crops = len(yolo_boxes) * n_frames
            print("=== Per-frame Filtering Summary ===")
            print(f"Frames:        {n_frames} ({start}..{end})  fps={fps:.2f}")
            print(f"YOLO boxes:    {len(yolo_boxes)}")
            print(f"Crops tested:  {total_crops}")
            print(f"Positives:     {len(rows_per_frame)}  ({(len(rows_per_frame)/max(1,total_crops))*100.0:.2f}%)")
            if pos_per_frame:
                mean_pf = np.mean(list(pos_per_frame.values()))
                med_pf = np.median(list(pos_per_frame.values()))
                print(f"Positives/frame: mean={mean_pf:.2f} median={med_pf:.2f}")
            if pos_per_box:
                mean_pb = np.mean(list(pos_per_box.values()))
                med_pb = np.median(list(pos_per_box.values()))
                print(f"Positives/box: mean={mean_pb:.2f} median={med_pb:.2f}")
            if SAVE_POSITIVE_CROPS:
                print(f"Saved positive crops: {saved_crops} → {(POSITIVE_CROPS_DIR if POSITIVE_CROPS_DIR is not None else (Path(OUTPUT_DIR) / 'perframe_crops' / Path(INPUT_IMAGE).stem / 'positives'))}")
        finally:
            cap.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
