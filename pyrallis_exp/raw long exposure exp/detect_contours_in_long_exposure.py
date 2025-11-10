#!/usr/bin/env python3
"""
Detect bright contours in a long‑exposure image and save an annotated
image with bounding boxes. Optionally, scan the source video frames for
each kept contour to extract 10×10 crops centered on the brightest pixel
per frame, filter them with a binary classifier, write a CSV of positives,
and render an overlay video with those boxes. Edit the globals below and run.

Rules implemented per request:
- Foreground pixels must have an average RGB intensity >= MIN_AVG_INTENSITY.
- Contours are extracted from this binary mask and drawn on the original image.
 - Output is saved to OUTPUT_DIR.
- Prints simple stats about detections.

Pipeline
- Stage 1 (optional): persistent‑bright removal computed from SOURCE_VIDEO_PATH
  over the frame range encoded in the long‑exposure name, e.g.
  100_<video_stem>_<mode>_000000-000099.
- Stage 2: threshold the long‑exposure by MIN_AVG_INTENSITY.
- Stage 3: contour extraction on the resulting foreground mask.
- Stage 4: area filtering via MIN_CONTOUR_AREA / MAX_CONTOUR_AREA.
- Stage 5: render long‑exposure annotated image (+ masks) to OUTPUT_DIR.
- Stage 6: print summary stats for long‑exposure detections.
- Stage 7 (optional): for each kept contour, per‑frame brightest‑pixel 10×10 crop,
  classify via CLASSIFIER_MODEL_PATH, and write a CSV of positives.
- Stage 8 (optional): draw per‑frame boxes from the CSV onto frames and save an
  overlay video, plus per‑frame summary stats.

Dependencies: pip install opencv-python numpy tqdm torch torchvision
"""

from __future__ import annotations
from pathlib import Path
import shutil
import csv
from collections import defaultdict, Counter
import cv2
import numpy as np


# ===================== Globals (edit these) =====================

# Path to the long‑exposure PNG (or JPG) image
INPUT_IMAGE_PATH = Path(
    '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/long exposure shots/20240606_cam1_GS010064/100_20240606_cam1_GS010064_lighten_000000-000099.png'
)


# Output directory for all generated files. The script clears this directory
# at start, then writes outputs here.
OUTPUT_DIR: Path | None = "/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/test1/pyrallis_exp/raw long exposure exp/temp long exp with contours drawn"

# Required: original source video path used to create the long‑exposure image
SOURCE_VIDEO_PATH: Path | None = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/raw input videos/20240606_cam1_GS010064.mp4'

# ----------------- Stage toggles -----------------
# Enable/disable individual stages of the pipeline
ENABLE_STAGE1_PERSISTENT = True     # Stage 1: persistent‑bright removal over frame range
ENABLE_STAGE2_THRESHOLD  = True     # Stage 2: long‑exposure threshold by MIN_AVG_INTENSITY
ENABLE_STAGE3_CONTOURS   = True     # Stage 3: contour extraction
ENABLE_STAGE4_AREA_FILTER= True     # Stage 4: area filtering of contours
ENABLE_STAGE5_RENDER     = True     # Stage 5: render annotated image + masks
ENABLE_STAGE6_STATS      = True     # Stage 6: print summary stats
ENABLE_STAGE7_PERFRAME   = False     # Stage 7: per‑frame 10×10 classification + CSV
ENABLE_STAGE8_OVERLAY    = False     # Stage 8: overlay video (depends on Stage 7)

# Foreground rule: pixel kept if mean(R,G,B) >= this value (0..255)
MIN_AVG_INTENSITY = 30

# Thickness and color of drawn bounding boxes
BBOX_THICKNESS = 1
BBOX_COLOR = (0, 0, 255)  # BGR (red)

# Optional: area-based filtering (in pixels)
# - Set MIN_CONTOUR_AREA to 0 to disable minimum filter.
# - Set MAX_CONTOUR_AREA to 0 (or None) to disable maximum filter.
MIN_CONTOUR_AREA = 4
MAX_CONTOUR_AREA: int | None = 230000

# Retrieval mode: "external" for outer boxes, "tree" for full hierarchy
RETRIEVAL_MODE = "external"  # or "tree"

# ----------------- Stage 1 (persistent bright) -----------------
# Pixels whose average intensity across the specified frame range is
# >= this threshold are treated as background and removed before
# contouring. The frame range is parsed from the long‑exposure image
# name: <interval>_<video_stem>_<mode>_<start>-<end>
PERSISTENT_MIN_AVG_INTENSITY = 160
PERSISTENT_RESIZE_TO_LONG = True  # resize video frames to match long-exposure size

# Misc
PROGRESS_EVERY = 25
SAVE_INTERMEDIATE_MASKS = True

# ----------------- Model filter over 10x10 crops -----------------
# Binary classifier checkpoint path (trained on 10x10 RGB crops).
# If None, model filtering is skipped.
CLASSIFIER_MODEL_PATH: Path | None = "/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/models and other data/pyrallis gopro models resnet18/resnet18_pyrallis_gopro_best_model v2.pt"  # e.g. Path("/path/to/resnet18_best.pt")

# Index of the positive class in the classifier's softmax output
POSITIVE_CLASS_INDEX = 1

# Minimum positive probability (0..1) to accept a candidate crop
POSITIVE_THRESHOLD = 0.98

# Optional: torchvision backbone to build for loading checkpoint
CLASSIFIER_BACKBONE = "resnet18"  # one of: resnet18|resnet34|resnet50|resnet101|resnet152

# Normalize input like ImageNet? (keep False if trained without)
CLASSIFIER_IMAGENET_NORM = False

# CSV path for per-frame accepted candidates (set to None to auto-place in OUTPUT_DIR)
CSV_OUTPUT_PATH: Path | None = "/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/test1/pyrallis_exp/raw long exposure exp/contour detection pipeline op data/csvs/perframe_candidates.csv"  # e.g. Path("./perframe_candidates.csv")

# Output path for annotated per-frame video
VIDEO_OUTPUT_PATH: Path | None = "/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/test1/pyrallis_exp/raw long exposure exp/contour detection pipeline op data/videos/perframe_candidates_overlay.mp4"  # e.g. Path("./overlay.mp4")


# ===================== Impl =====================

def _to_path(p: str | Path | None) -> Path | None:
    """Coerce str/Path/None to Path (expanded)."""
    if p is None:
        return None
    return Path(p).expanduser()

def _make_binary_mask(img_bgr: np.ndarray, min_avg: int) -> np.ndarray:
    """Return binary mask (uint8 0/255) where mean RGB >= min_avg."""
    mean_rgb = img_bgr.mean(axis=2)  # shape (H, W)
    mask = (mean_rgb >= float(min_avg)).astype(np.uint8) * 255
    return mask


def _find_contours(mask: np.ndarray, mode: str) -> list[np.ndarray]:
    mode = str(mode).lower().strip()
    if mode == "external":
        retrieval = cv2.RETR_EXTERNAL
    elif mode == "tree":
        retrieval = cv2.RETR_TREE
    else:
        raise ValueError("RETRIEVAL_MODE must be 'external' or 'tree'")
    contours, _hier = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _annotate(img_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]],
              color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    out = img_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, int(thickness))
    return out


def _build_contour_rect_masks(
    contours: list[np.ndarray]
) -> list[dict]:
    """For each contour return dict with bounding rect and local rect mask.

    Returns list of dicts: { 'rect': (x,y,w,h), 'mask': uint8(h,w) }.
    """
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        # localize contour points to rect coordinates
        c_local = c.copy()
        c_local[:, 0, 0] = c_local[:, 0, 0] - x
        c_local[:, 0, 1] = c_local[:, 0, 1] - y
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c_local], contourIdx=-1, color=255, thickness=-1)
        rects.append({"rect": (int(x), int(y), int(w), int(h)), "mask": mask})
    return rects


def _centered_box(cx: int, cy: int, bw: int, bh: int, W: int, H: int) -> tuple[int, int, int, int]:
    """Return top-left x,y and w,h for a box of size (bw,bh) centered at (cx,cy) clipped inside WxH.

    Ensures the returned box fits fully inside the image by shifting if needed.
    """
    bw = int(bw); bh = int(bh)
    half_w = bw // 2
    half_h = bh // 2
    x = int(cx) - half_w
    y = int(cy) - half_h
    # shift to fit
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + bw > W:
        x = max(0, W - bw)
    if y + bh > H:
        y = max(0, H - bh)
    # handle degenerate small images
    if W < bw:
        x = 0; bw = W
    if H < bh:
        y = 0; bh = H
    return int(x), int(y), int(bw), int(bh)


def _load_classifier(model_path: Path | None):
    """Load a simple 2-class classifier. Returns (model, device, transform) or (None, None, None)."""
    if model_path is None:
        return None, None, None
    import torch
    import torch.nn as nn
    from torchvision import models, transforms as T

    dev = (
        torch.device("cuda") if torch.cuda.is_available()
        else (torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else torch.device("cpu"))
    )

    # build backbone
    backbone_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }
    if CLASSIFIER_BACKBONE not in backbone_map:
        raise ValueError(f"Unknown CLASSIFIER_BACKBONE: {CLASSIFIER_BACKBONE}")
    model = backbone_map[CLASSIFIER_BACKBONE](weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(str(model_path), map_location=dev)
    # allow different save formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)
    model.eval().to(dev)

    # transforms to 10x10 tensor
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
    """Return positive probability for a BGR patch using the loaded classifier."""
    import torch
    # convert to RGB
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    x = transform(patch_rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pos_idx = int(POSITIVE_CLASS_INDEX)
    return float(prob[pos_idx])


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


def _stage1_persistent_mask_from_video(
    video_path: Path,
    start_idx: int,
    end_idx: int,
    target_wh: tuple[int, int],
    min_avg: int,
    resize_to_target: bool = True,
) -> tuple[np.ndarray, int]:
    """Compute persistent‑bright mask by streaming frames from video.

    Returns (mask_uint8, used_frames).
    """
    cap, VW, VH, VCOUNT = _open_video(video_path)
    try:
        start = max(0, int(start_idx))
        end = max(start, int(end_idx))
        if VCOUNT > 0:
            end = min(end, VCOUNT - 1)

        Wt, Ht = target_wh
        accum = np.zeros((Ht, Wt), dtype=np.float32)
        used = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        total = (end - start + 1)
        while idx <= end:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if (w, h) != (Wt, Ht):
                if not resize_to_target:
                    raise ValueError(
                        f"Video frame size {w}x{h} != long-exposure {Wt}x{Ht}. Enable PERSISTENT_RESIZE_TO_LONG."
                    )
                frame = cv2.resize(frame, (Wt, Ht), interpolation=cv2.INTER_AREA)
            mean_rgb = frame.mean(axis=2)
            accum += mean_rgb.astype(np.float32)
            used += 1
            idx += 1
            if used % max(1, PROGRESS_EVERY) == 0:
                print(f"Stage1 persistent avg: {used}/{total} frames", end="\r")

        if used == 0:
            raise RuntimeError("No frames processed for persistent mask")
        avg = accum / float(used)
        mask = (avg >= float(min_avg)).astype(np.uint8) * 255
        print(" " * 60, end="\r")
        return mask, used
    finally:
        cap.release()


def run():
    in_path = _to_path(INPUT_IMAGE_PATH)
    assert in_path.exists(), f"Input image not found: {in_path}"

    # Prepare/clear output directory
    out_dir = _to_path(OUTPUT_DIR)
    if out_dir is None:
        raise ValueError("OUTPUT_DIR must be set to a valid directory path")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing contents
    for child in list(out_dir.iterdir()):
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except Exception as e:
            print(f"Warning: could not remove {child}: {e}")

    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {in_path}")
    H, W = img.shape[:2]

    # ================= Stage 1: persistent-bright removal (optional) =================
    persistent_mask = None
    used_frames = 0
    if ENABLE_STAGE1_PERSISTENT and SOURCE_VIDEO_PATH is not None:
        src_video = _to_path(SOURCE_VIDEO_PATH)
        if not src_video or not src_video.exists():
            raise FileNotFoundError(f"SOURCE_VIDEO_PATH not found: {SOURCE_VIDEO_PATH}")
        meta = _parse_meta_from_image_path(in_path)
        start, end = int(meta["start"]), int(meta["end"])  # inclusive
        persistent_mask, used_frames = _stage1_persistent_mask_from_video(
            src_video,
            start,
            end,
            (W, H),
            int(PERSISTENT_MIN_AVG_INTENSITY),
            resize_to_target=bool(PERSISTENT_RESIZE_TO_LONG),
        )

    # ================= Stage 2: long-exposure threshold =================
    if ENABLE_STAGE2_THRESHOLD:
        base_thresh_mask = _make_binary_mask(img, int(MIN_AVG_INTENSITY))
    else:
        # If disabled, default to keeping all pixels
        base_thresh_mask = np.full((H, W), 255, dtype=np.uint8)
    if persistent_mask is not None:
        # keep thresholded foreground but drop persistent-bright pixels
        mask = cv2.bitwise_and(base_thresh_mask, cv2.bitwise_not(persistent_mask))
    else:
        mask = base_thresh_mask

    # ================= Stage 3: contours =================
    contours = _find_contours(mask, RETRIEVAL_MODE) if ENABLE_STAGE3_CONTOURS else []
    contours_before = len(contours)

    # ================= Stage 4: area filtering + boxes =================
    boxes: list[tuple[int, int, int, int]] = []
    kept_contours = 0
    for c in contours:
        if ENABLE_STAGE4_AREA_FILTER:
            area = float(cv2.contourArea(c))
            if MIN_CONTOUR_AREA and area < float(MIN_CONTOUR_AREA):
                continue
            if MAX_CONTOUR_AREA and area > float(MAX_CONTOUR_AREA):
                continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((int(x), int(y), int(w), int(h)))
        kept_contours += 1

    # ================= Stage 5: render long‑exposure image =================
    annotated = _annotate(img, boxes, color=BBOX_COLOR, thickness=int(BBOX_THICKNESS))

    # 5) Save outputs
    out_img_path = out_dir / f"{in_path.stem}_contours_t{MIN_AVG_INTENSITY}.png"
    out_mask_path = out_dir / f"{in_path.stem}_mask_t{MIN_AVG_INTENSITY}.png"
    if ENABLE_STAGE5_RENDER:
        cv2.imwrite(str(out_img_path), annotated)
        cv2.imwrite(str(out_mask_path), mask)
        if SAVE_INTERMEDIATE_MASKS:
            # Save stage masks for debugging
            if persistent_mask is not None:
                persist_path = out_mask_path.parent / f"{in_path.stem}_persistent_mask_t{PERSISTENT_MIN_AVG_INTENSITY}.png"
                cv2.imwrite(str(persist_path), persistent_mask)
            base_mask_path = out_mask_path.parent / f"{in_path.stem}_threshold_mask_t{MIN_AVG_INTENSITY}.png"
            cv2.imwrite(str(base_mask_path), base_thresh_mask)

    # ================= Stage 6: long‑exposure stats =================
    total_contours = contours_before
    num_boxes = len(boxes)
    box_areas = np.array([w * h for (_, _, w, h) in boxes], dtype=np.float64) if boxes else np.array([], dtype=np.float64)
    coverage = float(box_areas.sum()) / float(W * H) * 100.0 if box_areas.size else 0.0
    aspect_ratios = np.array([(max(w, 1) / max(h, 1)) for (_, _, w, h) in boxes], dtype=np.float64) if boxes else np.array([], dtype=np.float64)
    bright_ratio = float((mask > 0).sum()) / float(W * H) * 100.0

    if ENABLE_STAGE6_STATS:
        print("=== Contour Detection Summary ===")
        print(f"Input:        {in_path}")
        print(f"Saved image:  {out_img_path}")
        print(f"Saved mask:   {out_mask_path}")
        print(f"Image size:   {W}x{H}  ({W*H} px)")
        print(f"Stage1:       enabled={ENABLE_STAGE1_PERSISTENT}  frames used={used_frames}  persistent threshold >= {PERSISTENT_MIN_AVG_INTENSITY}")
        if persistent_mask is not None:
            persist_ratio = float((persistent_mask > 0).sum()) / float(W * H) * 100.0
            print(f"              persistent bright px: {persist_ratio:.2f}%")
        print(f"Stage2:       enabled={ENABLE_STAGE2_THRESHOLD}  mean RGB >= {MIN_AVG_INTENSITY}")
        max_area_disp = "None" if not MAX_CONTOUR_AREA else str(MAX_CONTOUR_AREA)
        print(f"Contours:     {total_contours} found before filter, {num_boxes} kept (min area {MIN_CONTOUR_AREA}, max area {max_area_disp})")
        print(f"Bright px:    {bright_ratio:.2f}% of image")
        if num_boxes:
            print(f"Box area:     total={box_areas.sum():.0f}  mean={box_areas.mean():.1f}  median={np.median(box_areas):.1f}  max={box_areas.max():.0f}")
            print(f"Coverage:     {coverage:.3f}% of image")
            print(f"Aspect ratio: mean={aspect_ratios.mean():.2f}  median={np.median(aspect_ratios):.2f}")
        else:
            print("No boxes kept after filtering.")

    # ================= Stage 7 (optional): per‑frame 10x10 model filtering + CSV =================
    # For each kept contour, scan each contributing frame, find the brightest pixel inside
    # the contour for that frame, center a 10x10 box there, classify, and keep if positive.
    if ENABLE_STAGE7_PERFRAME and SOURCE_VIDEO_PATH is not None and num_boxes > 0:
        src_video = _to_path(SOURCE_VIDEO_PATH)
        meta = _parse_meta_from_image_path(in_path)
        start, end = int(meta["start"]), int(meta["end"])  # inclusive
        n_frames = (end - start + 1)

        # Precompute rect masks for kept contours
        # Reconstruct kept contours aligned with boxes; handle duplicate rects via multiset
        rect_counts = Counter(boxes)
        filtered_contours = []
        for c in contours:
            r = cv2.boundingRect(c)
            r = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            if rect_counts.get(r, 0) > 0:
                filtered_contours.append(c)
                rect_counts[r] -= 1
        rect_masks = _build_contour_rect_masks(filtered_contours)

        # Load classifier (optional)
        model, dev, transform = _load_classifier(_to_path(CLASSIFIER_MODEL_PATH))
        if model is None:
            print("Model filtering skipped: CLASSIFIER_MODEL_PATH is not set.")
            return

        try:
            from tqdm import tqdm
        except Exception:
            def tqdm(x, **kwargs):
                return x

        cap2, VW, VH, VCOUNT = _open_video(src_video)
        try:
            # resize frames to long-exposure size if needed
            resize_to_target = bool(PERSISTENT_RESIZE_TO_LONG)
            if VCOUNT > 0:
                end = min(end, VCOUNT - 1)
                n_frames = (end - start + 1)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, start)

            # Result accumulators
            rows: list[dict] = []
            frame_boxes: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
            positives_per_contour = Counter()
            positives_per_frame = Counter()

            print("Per-frame filtering: scanning frames and classifying 10x10 crops…")
            for idx in tqdm(range(start, end + 1), total=n_frames, ncols=100):
                ok, frame = cap2.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != (W, H):
                    if not resize_to_target:
                        raise ValueError(
                            f"Video frame size {frame.shape[1]}x{frame.shape[0]} != long-exposure {W}x{H}. Enable PERSISTENT_RESIZE_TO_LONG."
                        )
                    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

                # process each kept contour
                for ci, rm in enumerate(rect_masks):
                    x, y, w, h = rm["rect"]
                    loc_mask = rm["mask"]  # uint8 h,w in [0,255]
                    roi = frame[y:y + h, x:x + w]
                    if roi.size == 0:
                        continue
                    # brightness = mean over channels
                    mean_roi = roi.mean(axis=2)
                    # apply mask: set outside to -1 so max is inside
                    masked = mean_roi.copy()
                    masked[loc_mask == 0] = -1.0
                    # brightest pixel location
                    flat_idx = int(np.argmax(masked))
                    by = int(flat_idx // w)
                    bx = int(flat_idx % w)
                    cx = x + bx
                    cy = y + by
                    # 10x10 box centered at brightest
                    rx, ry, rw, rh = _centered_box(cx, cy, 10, 10, W, H)
                    patch = frame[ry:ry + rh, rx:rx + rw]
                    if patch.shape[0] != 10 or patch.shape[1] != 10:
                        patch = cv2.resize(patch, (10, 10), interpolation=cv2.INTER_LINEAR)

                    p_pos = _predict_patch(model, dev, transform, patch)
                    if p_pos >= float(POSITIVE_THRESHOLD):
                        rec = {
                            "frame_idx": idx,
                            "video_name": Path(src_video).name,
                            "x": int(rx),
                            "y": int(ry),
                            "w": int(rw),
                            "h": int(rh),
                            "conf": float(p_pos),
                            "contour_id": int(ci),
                            "long_exposure": in_path.name,
                        }
                        rows.append(rec)
                        frame_boxes[idx].append((int(rx), int(ry), int(rw), int(rh)))
                        positives_per_contour[ci] += 1
                        positives_per_frame[idx] += 1

            # Write CSV
            csv_path = _to_path(CSV_OUTPUT_PATH)
            if csv_path is None:
                csv_path = out_dir / f"{in_path.stem}_perframe_candidates_thr{int(POSITIVE_THRESHOLD*100):02d}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_idx", "video_name", "x", "y", "w", "h", "conf", "contour_id", "long_exposure"
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

            print(f"Saved per-frame CSV → {csv_path}  (positives: {len(rows)})")

            # Rebuild frame_boxes by reading the CSV (use the CSV as source-of-truth)
            frame_boxes = defaultdict(list)
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    idxr = int(r["frame_idx"]) if r.get("frame_idx") is not None else 0
                    xr = int(float(r["x"]))
                    yr = int(float(r["y"]))
                    wr = int(float(r["w"]))
                    hr = int(float(r["h"]))
                    frame_boxes[idxr].append((xr, yr, wr, hr))

            # ================= Stage 8 (optional): render overlay video =================
            if ENABLE_STAGE8_OVERLAY:
                # Render annotated video using the per-frame boxes
                fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vid_out_path = _to_path(VIDEO_OUTPUT_PATH)
                if vid_out_path is None:
                    vid_out_path = out_dir / f"{in_path.stem}_overlay.mp4"
                writer = cv2.VideoWriter(str(vid_out_path), fourcc, float(fps), (W, H))
                cap2.set(cv2.CAP_PROP_POS_FRAMES, start)
                print("Rendering video with overlays…")
                for idx in tqdm(range(start, end + 1), total=n_frames, ncols=100):
                    ok, frame = cap2.read()
                    if not ok:
                        break
                    if (frame.shape[1], frame.shape[0]) != (W, H):
                        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                    bxs = frame_boxes.get(idx, [])
                    for (rx, ry, rw, rh) in bxs:
                        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), BBOX_COLOR, int(BBOX_THICKNESS))
                    writer.write(frame)
                writer.release()
                print(f"Saved overlay video → {vid_out_path}")

            # ------- Per-frame stats -------
            total_crops = len(rect_masks) * n_frames
            print("=== Per-frame Filtering Summary ===")
            print(f"Frames:        {n_frames} ({start}..{end})  fps={fps:.2f}")
            print(f"Contours kept: {len(rect_masks)}")
            print(f"Crops tested:  {total_crops}")
            print(f"Positives:     {len(rows)}  ({(len(rows)/max(1,total_crops))*100.0:.2f}%)")
            if positives_per_frame:
                mean_pf = np.mean(list(positives_per_frame.values()))
                med_pf = np.median(list(positives_per_frame.values()))
                print(f"Positives/frame: mean={mean_pf:.2f} median={med_pf:.2f}")
            if positives_per_contour:
                mean_pc = np.mean(list(positives_per_contour.values()))
                med_pc = np.median(list(positives_per_contour.values()))
                print(f"Positives/contour: mean={mean_pc:.2f} median={med_pc:.2f}")
        finally:
            cap2.release()


if __name__ == "__main__":
    run()
