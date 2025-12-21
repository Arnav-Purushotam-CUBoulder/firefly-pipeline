#!/usr/bin/env python3
"""
V3 Daytime Pipeline – Negative Patch Generator (blob-based)
───────────────────────────────────────────────────────────

Purpose
-------
Given a folder of RGB images that are “hard negatives” for the v3 daytime
Stage 3 patch classifier, this script:
  • runs a SimpleBlobDetector over each image, and
  • extracts a PATCH_SIZE×PATCH_SIZE crop centred at each detected blob,
  • saving all such crops into an output folder for use as negative-class
    training patches.

Configuration
-------------
Edit the global variables below:

  INPUT_IMAGES_DIR  – folder containing source images (PNG/JPEG/etc.).
  OUTPUT_DIR        – folder where all negative patches will be written.
  PATCH_SIZE        – side length of square patches (pixels, e.g. 10).
  OUTPUT_PREFIX     – filename prefix for saved patches.
  OUTPUT_EXT        – output format (".png" recommended).

Blob detector knobs (edit as needed):
  SBD_MIN_AREA_PX       – minimum blob area in pixels.
  SBD_MAX_AREA_SCALE    – fraction of image area allowed for blobs
                          (<=0 means no upper bound).
  SBD_MIN_DIST          – minimum distance between blob centres (pixels).
  SBD_MIN_REPEAT        – detector repeatability parameter.

Basic preprocessing:
  USE_CLAHE, USE_TOPHAT, USE_DOG toggles (optional contrast enhancement).

Behavior
--------
  • For each image in INPUT_IMAGES_DIR (extensions .png/.jpg/.jpeg):
      - Convert to grayscale, apply simple preprocessing.
      - Detect blobs with OpenCV’s SimpleBlobDetector.
      - For each blob centre, extract a PATCH_SIZE×PATCH_SIZE crop centred
        there, padding with black if near the borders.
      - Save each crop as:
            <OUTPUT_PREFIX>_<image_stem>_kXXXX_xX_yY.<ext>
  • At the end, prints summary stats to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# ───────── Global configuration (edit these) ─────────
INPUT_IMAGES_DIR: str = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset/negative edge cases generation/images'   # Folder containing hard-negative images
OUTPUT_DIR: str = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset/negative edge cases generation/patches'         # Folder where 10x10 patches will be saved

PATCH_SIZE: int = 10         # Size of square patch (e.g. 10×10)

OUTPUT_PREFIX: str = "negblob"
OUTPUT_EXT: str = ".png"     # ".png" is lossless and preferred

OVERWRITE_EXISTING: bool = False

# Blob detector parameters
SBD_MIN_AREA_PX: float = 0.5
SBD_MAX_AREA_SCALE: float = 1.0   # Fraction of image area allowed (<=0 => no upper bound)
SBD_MIN_DIST: float = 0.25        # Min distance between blob centres (in px)
SBD_MIN_REPEAT: int = 1           # SimpleBlobDetector repeatability

# Preprocessing toggles
USE_CLAHE: bool = True
CLAHE_CLIP: float = 2.0
CLAHE_TILE: Tuple[int, int] = (8, 8)

USE_TOPHAT: bool = False
TOPHAT_KSIZE: int = 7            # Odd number

USE_DOG: bool = False
DOG_SIGMA1: float = 0.8
DOG_SIGMA2: float = 1.6
# ─────────────────────────────────────────────────────


def _log(msg: str) -> None:
    print(msg, flush=True)


def _make_blob_detector(min_area: float, max_area: float, min_dist: float, min_repeat: int):
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
    """Apply simple contrast enhancement / filtering before blob detection."""
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


def _extract_centered_patch(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """Return size×size patch centred at (cx,cy), padded with black if near borders."""
    h, w = image.shape[:2]
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size

    xi0 = max(0, x0)
    yi0 = max(0, y0)
    xi1 = min(w, x1)
    yi1 = min(h, y1)

    patch = image[yi0:yi1, xi0:xi1]
    if patch.shape[:2] == (size, size):
        return patch

    if image.ndim == 2:
        padded = np.zeros((size, size), dtype=image.dtype)
    else:
        padded = np.zeros((size, size, image.shape[2]), dtype=image.dtype)

    oy = yi0 - y0
    ox = xi0 - x0
    padded[oy:oy + patch.shape[0], ox:ox + patch.shape[1]] = patch
    return padded


def generate_negative_patches() -> None:
    in_dir = Path(INPUT_IMAGES_DIR).expanduser()
    out_dir = Path(OUTPUT_DIR).expanduser()

    if not in_dir.exists() or not in_dir.is_dir():
        _log(f"ERROR: INPUT_IMAGES_DIR not found or not a directory: {in_dir}")
        return

    if PATCH_SIZE <= 0:
        _log("ERROR: PATCH_SIZE must be a positive integer.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in sorted(in_dir.iterdir()) if p.is_file() and p.suffix.lower() in img_exts]
    if not images:
        _log(f"ERROR: No images found in {in_dir}")
        return

    total_imgs = 0
    total_blobs = 0
    total_saved = 0

    _log("=== Negative Patch Generator (blob-based) ===")
    _log(f"Input images dir : {in_dir}")
    _log(f"Output dir       : {out_dir}")
    _log(f"Patch size       : {PATCH_SIZE}x{PATCH_SIZE}")

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            _log(f"WARNING: Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]
        area = float(w * h)
        min_area = float(SBD_MIN_AREA_PX)
        max_area = float(area * SBD_MAX_AREA_SCALE) if SBD_MAX_AREA_SCALE > 0 else float(area)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        proc = _preprocess(gray)
        detector = _make_blob_detector(min_area, max_area, SBD_MIN_DIST, SBD_MIN_REPEAT)

        keypoints = detector.detect(proc)
        num_kp = len(keypoints)
        total_imgs += 1
        total_blobs += num_kp

        _log(f"{img_path.name}: blobs={num_kp}")

        # Live counter of total blobs detected so far
        sys.stdout.write(
            f"\r[progress] images={total_imgs}/{len(images)}  total_blobs={total_blobs}"
        )
        sys.stdout.flush()

        for idx, kp in enumerate(keypoints):
            cx, cy = kp.pt
            patch = _extract_centered_patch(img, cx, cy, PATCH_SIZE)

            fname = (
                f"{OUTPUT_PREFIX}_{img_path.stem}_k{idx:04d}_"
                f"x{int(round(cx))}_y{int(round(cy))}{OUTPUT_EXT}"
            )
            fpath = out_dir / fname

            if not OVERWRITE_EXISTING and fpath.exists():
                total_saved += 1
                continue

            ok = cv2.imwrite(str(fpath), patch)
            if ok:
                total_saved += 1

    # Move to next line after live progress
    sys.stdout.write("\n")

    _log("=== Summary ===")
    _log(f"Images processed : {total_imgs}")
    _log(f"Blobs detected   : {total_blobs}")
    _log(f"Patches saved    : {total_saved}")
    _log(f"Output folder    : {out_dir}")
    _log("=============================================")


def main(argv: list[str] | None = None) -> int:
    _ = argv  # unused
    generate_negative_patches()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
