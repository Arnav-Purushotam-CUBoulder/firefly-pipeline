#!/usr/bin/env python3
"""
Crop 10x10 patches for 'firefly' rows from a CSV + video.

Input CSV schema (tab or comma delimited; set DELIMITER below):
    frame    x    y    w    h    class
Example:
    1    1584    522    3    3    firefly

Notes:
  • Only rows where class == 'firefly' (case-insensitive) are processed.
  • (x, y) are treated as the **geometric center** of the 10×10 crop.
  • Frame numbering in the CSV can start at 1 or 0; set FRAME_BASE accordingly.
  • Crops are saved as PNGs into OUTPUT_DIR.

Requires: opencv-python, numpy, torch, torchvision, Pillow
"""

from pathlib import Path
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as T
import math
from typing import Dict, List, Tuple

# ─── GLOBALS: set these to your environment ───────────────────

INPUT_CSV   = Path('/Users/arnavps/Desktop/tremulans and forresti seperate models inference data/tremulans inference data/csv files/fixed_to_center_val_4k-5k_tremulans_clip.csv')     # your CSV/TSV file
INPUT_VIDEO = Path('/Users/arnavps/Desktop/FFT resnet fixing FPs/20px tremulans inference data/raw input videos/val_4k-5k_tremulans_clip.mp4')           # your video file
OUTPUT_DIR  = Path("forresti, fixing FPs and box overlap/Proof of concept code/test1/bug fixing forresti/bug fixing data/output crops tremulans model only")        # where 10x10 crops will be written
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CSV delimiter — set to '\t' for TSV, or ',' for CSV
DELIMITER   = ','

# Only rows with this class label are cropped
TARGET_CLASS = "firefly"       # case-insensitive match

# Crop size (centered on (x, y))
PATCH_W = 10
PATCH_H = 10

# CSV 'frame' numbering base:
#   If your CSV's first frame is labeled "1", set FRAME_BASE = 1 (so frame 1 -> video index 0)
#   If your CSV's first frame is labeled "0", set FRAME_BASE = 0
FRAME_BASE = 0

# Zero-pad for filenames when saving crops
FRAME_ZPAD = 6
FILENAME_PREFIX = "crop_"      # prefix for saved crop files (optional)

# Optional: Inclusive frame ranges to process (CSV frame numbers). Example: [[0, 300], [1200, 1300]]
# If empty, process ALL frames.
FRAME_RANGES: List[List[int]] = []

# Logits output CSV (rows for the crops we actually saved)
OUTPUT_LOGITS_CSV = Path('/Users/arnavps/Desktop/tremulans and forresti seperate models inference data/tremulans inference data/csv files/tremulans_model_xyt_val_4k-5k.csv')

# CNN model params for logits
MODEL_PATH   = Path('frontalis, tremulans and forresti global models/resnet18_Tremulans_best_model.pt')
BACKBONE     = 'resnet18'   # one of: resnet18, resnet34, resnet50, resnet101, resnet152
BACK_IDX     = 0
FIRE_IDX     = 1

# Offset to add to 't' (frame index) right before the FINAL save of OUTPUT_LOGITS_CSV
OUTPUT_T_OFFSET = 3999

# ─── helpers ───────────────────────────────────────────────────

def _center_crop_geometric(img: np.ndarray, cx: float, cy: float, w: int, h: int):
    """
    Return a w×h crop whose geometric center is exactly (cx, cy) in image coordinates
    (after clamping center so the crop stays inside the frame). Uses cv2.getRectSubPix
    for subpixel-accurate cropping. Also returns (x0, y0) = top-left (rounded int).
    """
    H, W = img.shape[:2]
    if W == 0 or H == 0:
        return np.zeros((h, w, 3), dtype=img.dtype), 0, 0

    half_w = w / 2.0
    half_h = h / 2.0

    # Clamp center so the requested patch lies inside the image bounds
    cx_clamped = min(max(cx, half_w), max(half_w, W - half_w))
    cy_clamped = min(max(cy, half_h), max(half_h, H - half_h))

    # Subpixel-exact crop; result is always size (h, w)
    crop = cv2.getRectSubPix(img, (int(w), int(h)), (float(cx_clamped), float(cy_clamped)))

    # Provide an integer "anchor" for filename/debug (not used for the crop itself)
    x0 = int(round(cx_clamped - half_w))
    y0 = int(round(cy_clamped - half_h))
    return crop, x0, y0


def _normalize_ranges(ranges):
    """Clean, sort, and merge inclusive [start, end] ranges."""
    clean = []
    for pair in ranges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        if a > b:
            a, b = b, a
        clean.append([a, b])
    if not clean:
        return []
    clean.sort(key=lambda ab: ab[0])
    merged = [clean[0]]
    for s, e in clean[1:]:
        ms, me = merged[-1]
        if s <= me + 1:
            merged[-1][1] = max(me, e)
        else:
            merged.append([s, e])
    return merged


def _in_ranges(x: int, merged_ranges) -> bool:
    """Return True if x is in any inclusive [s,e] of merged_ranges."""
    if not merged_ranges:
        return True  # no filter → allow all
    for s, e in merged_ranges:
        if s <= x <= e:
            return True
        if x < s:
            return False
    return False


def _read_firefly_rows(csv_path: Path, delimiter: str):
    """
    Parse CSV/TSV and return a dict:
        { frame_number(int): [(cx(float), cy(float)), ...], ... }
    Only includes rows where class == TARGET_CLASS (case-insensitive).

    IMPORTANT: This treats CSV x,y AS THE DESIRED GEOMETRIC CENTER for cropping.
    No conversion from top-left is done here.
    """
    by_frame: Dict[int, List[Tuple[float, float]]] = {}
    total_rows = 0
    kept_rows = 0
    with csv_path.open('r', newline='') as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        required = {'frame', 'x', 'y', 'w', 'h', 'class'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required columns. Found: {reader.fieldnames}")

        for r in reader:
            total_rows += 1
            cls = (r.get('class') or '').strip().lower()
            if cls != TARGET_CLASS.lower():
                continue
            try:
                fr = int(r['frame'])
                # Read centers as float to preserve subpixel positions if present
                cx  = float(r['x'])
                cy  = float(r['y'])
            except Exception:
                continue
            by_frame.setdefault(fr, []).append((cx, cy))
            kept_rows += 1

    return by_frame, total_rows, kept_rows


def _crop_and_save_for_frame(frame_img: np.ndarray,
                             frame_num: int,
                             points: List[Tuple[float, float]]) -> int:
    """
    Given a frame image and a list of (cx,cy) centers, save 10x10 crops.
    Returns number of crops successfully saved.
    """
    saved = 0
    for (cx, cy) in points:
        crop, x0, y0 = _center_crop_geometric(frame_img, cx, cy, PATCH_W, PATCH_H)
        # crop from getRectSubPix is always PATCH_H × PATCH_W
        out_name = f"{FILENAME_PREFIX}frame_{frame_num:0{FRAME_ZPAD}d}_x{int(round(cx))}_y{int(round(cy))}.png"
        cv2.imwrite(str(OUTPUT_DIR / out_name), crop)
        saved += 1
    return saved


# ─── logits model helpers ─────────────────────────────────────

def _build_resnet(name: str, num_classes: int = 2) -> nn.Module:
    fns = {
        'resnet18':  models.resnet18,
        'resnet34':  models.resnet34,
        'resnet50':  models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    fn = fns.get(name)
    if fn is None:
        raise ValueError(f'unknown backbone {name}')
    net = fn(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    nn.init.xavier_uniform_(net.fc.weight); nn.init.zeros_(net.fc.bias)
    return net

TFM = T.ToTensor()  # converts H×W×C RGB uint8 -> float tensor in [0,1]

def _crop_logits(model, device, bgr_crop_10x10: np.ndarray) -> tuple[float, float]:
    # Convert BGR -> RGB, to PIL, then tensor
    rgb = cv2.cvtColor(bgr_crop_10x10, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    ten = TFM(pil).unsqueeze(0).to(device)  # [1,3,10,10]
    logits = model(ten)[0]
    fire = float(logits[FIRE_IDX].item())
    back = float(logits[BACK_IDX].item())
    return back, fire


def _fire_prob_from_logits(logit_back: float, logit_fire: float) -> float:
    """Compute P(firefly) from background/firefly logits via stable softmax."""
    m = max(logit_back, logit_fire)
    eb = math.exp(logit_back - m)
    ef = math.exp(logit_fire - m)
    return ef / (ef + eb + 1e-12)


# ─── main ─────────────────────────────────────────────────────

def main():
    # Parse CSV rows grouped by frame
    fire_rows_by_frame, total_rows, kept_rows = _read_firefly_rows(INPUT_CSV, DELIMITER)
    if not fire_rows_by_frame:
        print("No matching 'firefly' rows found. Exiting.")
        return

    # Prepare frame range filter
    ranges_merged = _normalize_ranges(FRAME_RANGES)
    apply_filter = bool(ranges_merged)  # if False, process all frames

    # Open video
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_VIDEO}")

    # Get total frame count if available
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    # Load model for logits
    device = ('mps' if torch.backends.mps.is_available()
              else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_resnet(BACKBONE).to(device)
    state = torch.load(str(MODEL_PATH), map_location=device)
    state = state if 'model' not in state else state['model']
    model.load_state_dict(state, strict=True)
    model.eval()

    target_frames_sorted = sorted(fire_rows_by_frame.keys())
    crops_saved = 0
    frames_processed = 0
    frames_missing = 0

    # Collect logits rows corresponding to actually saved crops
    logits_rows: list[dict] = []

    with torch.no_grad():
        for fr in target_frames_sorted:
            # If ranges are provided, skip frames outside the ranges
            if apply_filter and not _in_ranges(fr, ranges_merged):
                continue

            vid_idx = fr - FRAME_BASE  # convert CSV frame to 0-based index for OpenCV
            if vid_idx < 0:
                frames_missing += 1
                continue

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                frames_missing += 1
                continue

            frames_processed += 1
            points = fire_rows_by_frame.get(fr, [])

            for (cx, cy) in points:
                crop_bgr, _, _ = _center_crop_geometric(frame, cx, cy, PATCH_W, PATCH_H)

                # save crop image
                out_name = f"{FILENAME_PREFIX}frame_{fr:0{FRAME_ZPAD}d}_x{int(round(cx))}_y{int(round(cy))}.png"
                cv2.imwrite(str(OUTPUT_DIR / out_name), crop_bgr)
                crops_saved += 1

                # compute logits for this crop and stash a row
                back_logit, fire_logit = _crop_logits(model, device, crop_bgr)
                logits_rows.append({
                    'x': float(cx),
                    'y': float(cy),
                    't': vid_idx,  # t = frame index used for video (fr - FRAME_BASE)
                    'background_logit': back_logit,
                    'firefly_logit': fire_logit,
                })

    cap.release()

    # Write logits CSV for the crops we actually saved
    OUTPUT_LOGITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_LOGITS_CSV.open('w', newline='') as f:
        fieldnames = ['x', 'y', 't', 'background_logit', 'firefly_logit']
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        for r in logits_rows:
            wri.writerow(r)

    # ─── RELOAD & FILTER BY FIRELY CONFIDENCE (>= 0.5) ─────────
    with OUTPUT_LOGITS_CSV.open('r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    filtered = []
    for r in rows:
        try:
            lb = float(r['background_logit'])
            lf = float(r['firefly_logit'])
        except Exception:
            continue
        p_fire = _fire_prob_from_logits(lb, lf)
        if p_fire >= 0.5:
            filtered.append(r)

    # Apply final frame index offset to 't' before the FINAL save
    for r in filtered:
        try:
            r['t'] = int(round(float(r['t']))) + int(OUTPUT_T_OFFSET)
        except Exception:
            # if malformed, leave as-is
            pass

    with OUTPUT_LOGITS_CSV.open('w', newline='') as f:
        fieldnames = ['x', 'y', 't', 'background_logit', 'firefly_logit']
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        for r in filtered:
            wri.writerow(r)

    # ─── summary ───
    print("Done.")
    print(f"  CSV rows read                 : {total_rows}")
    print(f"  Firefly rows kept from CSV    : {kept_rows}")
    if total_frames is not None:
        print(f"  Video total frames (reported) : {total_frames}")
    print(f"  Unique frames with fireflies  : {len(fire_rows_by_frame)}")
    print(f"  Frames successfully processed : {frames_processed}")
    print(f"  Frames missing/unreadable     : {frames_missing}")
    print(f"  Crops saved                   : {crops_saved}")
    print(f"  Output directory              : {OUTPUT_DIR}")
    print(f"  Logits CSV (filtered >=0.5)   : {OUTPUT_LOGITS_CSV}")

if __name__ == "__main__":
    main()
