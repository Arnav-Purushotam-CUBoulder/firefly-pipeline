#!/usr/bin/env python3
import sys
import math
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# ================= Global Configuration (edit these) =================
INPUT_VIDEO_PATH = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/raw input videos/20240606_cam1_GS010064.mp4'
OUTPUT_CROPS_DIR = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/output data/20240606_cam1_GS010064/raw 10x10 crops from sbd'

# Number of frames to process; if None or <=0, process entire video
MAX_FRAMES = 20  # Optional hard cap (int) or None

# Saving options
CROP_SIZE = 10                 # Fixed crop size: 10x10
SAVE_COLOR = True            # Save crops in color (True) or grayscale (False)
IMAGE_EXT = ".png"           # ".png" or ".jpg"
OVERWRITE_EXISTING = False    # Overwrite if file exists
MAX_CROPS_PER_FRAME = None    # Limit crops per frame (int) or None for no limit
FILENAME_PREFIX = "crop"     # Output filename prefix

# Region-of-interest (ROI) scanning — process only a square per frame
# Each ROI square covers this fraction of the full frame area
ROI_AREA_FRACTION = 1.0 / 20.0   # 1/20th of the frame area
ROI_RANDOM_ORDER = False         # If True, randomize tile order across frames

# Blob detector parameters
SBD_MIN_AREA_PX = 0.5
SBD_MAX_AREA_SCALE = 1.0      # Fraction of full frame area allowed (<=0 means no limit)
SBD_MIN_DIST = 0.25           # Min distance between blobs (in px)
SBD_MIN_REPEAT = 1            # Detector min repeatability

# Preprocessing toggles
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE: Tuple[int, int] = (8, 8)

USE_TOPHAT = False
TOPHAT_KSIZE = 7              # Odd number (5,7,9,...)

USE_DOG = False
DOG_SIGMA1 = 0.8
DOG_SIGMA2 = 1.6

# Runtime/UI
DRY_RUN = False
LOG_EVERY_N_FRAMES = 10       # Live stats cadence
BAR_LEN = 50
# ====================================================================


def progress(i: int, total: int, tag: str = "", live: str = "") -> None:
    total = max(1, int(total))
    i = min(max(0, int(i)), total)
    frac = i / total
    fill = int(frac * BAR_LEN)
    bar = "=" * fill + " " * (BAR_LEN - fill)
    sys.stdout.write(f"\r{tag} [{bar}] {int(frac*100):3d}% {live}")
    sys.stdout.flush()
    if i >= total:
        sys.stdout.write("\n")


def log(msg: str) -> None:
    print(msg, flush=True)


def _make_blob_detector(min_area: float, max_area: float, min_dist: float = 0.25, min_repeat: int = 1):
    p = cv2.SimpleBlobDetector_Params()

    p.filterByColor = False

    p.filterByArea = True
    p.minArea = float(min_area)
    p.maxArea = float(max_area)

    p.filterByCircularity = False
    p.filterByConvexity = False
    p.filterByInertia = False

    p.minThreshold = 0
    p.maxThreshold = 255
    p.thresholdStep = 1
    p.minRepeatability = int(min_repeat)

    p.minDistBetweenBlobs = float(min_dist)
    return cv2.SimpleBlobDetector_create(p)


def _preprocess(gray: np.ndarray) -> np.ndarray:
    inp = gray
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        inp = clahe.apply(inp)
    if USE_TOPHAT:
        ksize = TOPHAT_KSIZE if (TOPHAT_KSIZE >= 3 and TOPHAT_KSIZE % 2 == 1) else 7
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        inp = cv2.morphologyEx(inp, cv2.MORPH_TOPHAT, k)
    if USE_DOG:
        g1 = cv2.GaussianBlur(inp, (0, 0), DOG_SIGMA1)
        g2 = cv2.GaussianBlur(inp, (0, 0), DOG_SIGMA2)
        inp = cv2.subtract(g1, g2)
    return inp


def _extract_crop_10x10(image: np.ndarray, cx: float, cy: float) -> np.ndarray:
    h, w = image.shape[:2]
    half = CROP_SIZE // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + CROP_SIZE
    y1 = y0 + CROP_SIZE

    xi0 = max(0, x0)
    yi0 = max(0, y0)
    xi1 = min(w, x1)
    yi1 = min(h, y1)

    patch = image[yi0:yi1, xi0:xi1]
    if patch.shape[:2] == (CROP_SIZE, CROP_SIZE):
        return patch

    if image.ndim == 2:
        padded = np.zeros((CROP_SIZE, CROP_SIZE), dtype=image.dtype)
    else:
        padded = np.zeros((CROP_SIZE, CROP_SIZE, image.shape[2]), dtype=image.dtype)

    oy = yi0 - y0
    ox = xi0 - x0
    padded[oy:oy + patch.shape[0], ox:ox + patch.shape[1]] = patch
    return padded


def main() -> None:
    vid_path = Path(INPUT_VIDEO_PATH).expanduser()
    out_dir = Path(OUTPUT_CROPS_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Starting blob cropper")
    log(f"Input : {vid_path}")
    log(f"Output: {out_dir}")

    t_open0 = time.time()
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {vid_path}")
    log(f"Video opened in {(time.time()-t_open0):.2f}s")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Determine how many frames to process robustly even if frame count is unknown (0)
    if total_frames > 0:
        frames_target = min(total_frames, MAX_FRAMES) if isinstance(MAX_FRAMES, int) and MAX_FRAMES > 0 else total_frames
    else:
        frames_target = int(MAX_FRAMES) if isinstance(MAX_FRAMES, int) and MAX_FRAMES > 0 else 0

    if frames_target <= 0:
        log("Frame count unknown and MAX_FRAMES not set; nothing to do.")
        cap.release()
        return

    log(f"Meta: frames={total_frames}, res={width}x{height} @ {fps:.2f} fps")
    log(f"Processing {frames_target} frame(s)")
    progress(0, frames_target, tag="blob-crops", live="starting…")

    max_area = (width * height if SBD_MAX_AREA_SCALE <= 0
                else int(width * height * SBD_MAX_AREA_SCALE))
    detector = _make_blob_detector(
        min_area=SBD_MIN_AREA_PX,
        max_area=max_area,
        min_dist=SBD_MIN_DIST,
        min_repeat=SBD_MIN_REPEAT,
    )

    # Build ROI tiles (square regions) that each cover ~ROI_AREA_FRACTION of frame
    roi_side = max(3, int(round(math.sqrt(max(1e-9, ROI_AREA_FRACTION) * width * height))))
    # Ensure roi_side is not larger than frame dims
    roi_side = max(3, min(roi_side, width, height))

    # Non-overlapping grid with stride == roi_side; include last tile at borders
    xs = list(range(0, max(1, width - roi_side + 1), roi_side))
    ys = list(range(0, max(1, height - roi_side + 1), roi_side))
    if xs[-1] != width - roi_side:
        xs.append(width - roi_side)
    if ys[-1] != height - roi_side:
        ys.append(height - roi_side)
    tiles = [(x, y, x + roi_side, y + roi_side) for y in ys for x in xs]
    tiles_count = len(tiles)
    if ROI_RANDOM_ORDER and tiles_count > 1:
        import random
        random.shuffle(tiles)

    log(f"ROI tile side: {roi_side}px, tiles: {tiles_count}")

    start_time = time.time()
    frames_processed = 0
    total_detections = 0
    total_saved = 0
    frames_with_dets = 0
    per_frame_counts = []
    write_errors = 0
    skipped_due_to_limit = 0

    fr = 0
    first_decode_t0 = time.time()
    while fr < frames_target:
        ok, frame = cap.read()
        if not ok:
            break
        if frames_processed == 0:
            log(f"First frame decoded in {(time.time()-first_decode_t0):.2f}s")

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Select ROI tile for this frame (cycle through tiles)
        if tiles_count > 0:
            tile_idx = frames_processed % tiles_count
            x0, y0, x1, y1 = tiles[tile_idx]
        else:
            x0, y0, x1, y1 = 0, 0, width, height
            tile_idx = 0

        # Crop ROI and preprocess
        gray = gray_full[y0:y1, x0:x1]
        inp = _preprocess(gray)
        kps = detector.detect(inp)

        det_count = len(kps)
        total_detections += det_count
        if det_count > 0:
            frames_with_dets += 1

        saved_this_frame = 0
        to_iter = kps
        if isinstance(MAX_CROPS_PER_FRAME, int) and MAX_CROPS_PER_FRAME >= 0:
            to_iter = kps[:max(0, int(MAX_CROPS_PER_FRAME))]

        for idx, kp in enumerate(to_iter):
            if isinstance(MAX_CROPS_PER_FRAME, int) and MAX_CROPS_PER_FRAME >= 0 and saved_this_frame >= MAX_CROPS_PER_FRAME:
                break

            cx, cy = kp.pt
            # Choose source image based on SAVE_COLOR, but only within ROI
            src_full = frame if SAVE_COLOR else gray_full
            src_roi = src_full[y0:y1, x0:x1]
            crop = _extract_crop_10x10(src_roi, cx, cy)

            fname = f"{FILENAME_PREFIX}_f{fr:06d}_t{tile_idx:04d}_k{idx:03d}{IMAGE_EXT}"
            fpath = out_dir / fname

            if DRY_RUN:
                saved = True
            else:
                if fpath.exists() and not OVERWRITE_EXISTING:
                    saved = True
                else:
                    saved = cv2.imwrite(str(fpath), crop)

            if saved:
                total_saved += 1
                saved_this_frame += 1
            else:
                write_errors += 1

        if isinstance(MAX_CROPS_PER_FRAME, int) and MAX_CROPS_PER_FRAME >= 0:
            skipped_due_to_limit += max(0, det_count - saved_this_frame)

        per_frame_counts.append(saved_this_frame)
        frames_processed += 1
        fr += 1

        # Live progress every frame with saved-crops count
        elapsed = max(1e-6, time.time() - start_time)
        proc_fps = frames_processed / elapsed
        avg_per_frame = total_saved / max(1, frames_processed)
        live = (
            f"frm {frames_processed}/{frames_target} | tile {tile_idx+1}/{tiles_count} | det {total_detections} | "
            f"saved {total_saved} | last {saved_this_frame} | "
            f"avg {avg_per_frame:.2f}/frm | {proc_fps:.1f} fps"
        )
        progress(frames_processed, frames_target, tag="blob-crops", live=live)

    cap.release()
    progress(frames_processed, frames_target, tag="blob-crops", live="")

    elapsed = max(1e-6, time.time() - start_time)
    proc_fps = frames_processed / elapsed
    arr = np.array(per_frame_counts, dtype=np.float32) if per_frame_counts else np.array([0.0], dtype=np.float32)
    mean_crops = float(arr.mean())
    med_crops = float(np.median(arr))
    p95_crops = float(np.percentile(arr, 95)) if arr.size > 0 else 0.0

    log("=== Summary ===")
    log(f"Video: {vid_path}")
    log(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")
    log(f"Processed frames: {frames_processed}/{frames_target} ({(100*frames_processed/max(1,frames_target)):.1f}%)")
    log(f"Detections: {total_detections}")
    log(f"Crops saved: {total_saved} (errors: {write_errors})")
    if isinstance(MAX_CROPS_PER_FRAME, int) and MAX_CROPS_PER_FRAME >= 0:
        log(f"Skipped due to per-frame cap: {skipped_due_to_limit}")
    log(f"Frames with detections: {frames_with_dets}")
    log(
        f"Crops/frame — mean: {mean_crops:.2f}, median: {med_crops:.2f}, p95: {p95_crops:.2f}, "
        f"min: {int(arr.min())}, max: {int(arr.max())}"
    )
    log(f"Processing speed: {proc_fps:.2f} frames/sec")
    if DRY_RUN:
        log("[DRY RUN] No crops written.")
    log("===============")


if __name__ == "__main__":
    main()
