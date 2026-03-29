#!/usr/bin/env python3
"""
Day-time pipeline parameters and paths (single source of truth).

Edit ROOT to point at your working folder. ROOT must contain a folder
named "original videos" with your input videos. Each stage writes its
own outputs under ROOT.
"""
from __future__ import annotations

from pathlib import Path
from typing import List


# Root folder for this pipeline (EDIT THIS)
# - All stage outputs are saved here.
# - Must contain a subfolder named "original videos" with input videos.
ROOT: str | Path = "/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/daytime pipeline v1 inference output data"

# Normalize ROOT to a Path object even if provided as a string
if not isinstance(ROOT, Path):
    ROOT = Path(str(ROOT)).expanduser()

# Input videos folder (inside ROOT)
# - Place any number of input videos here.
ORIGINAL_VIDEOS_DIR: Path = ROOT / "original videos"

# Stage output roots (one folder per stage). Each stage writes under its own dir.
# Stage 1: trajectories; Stage 2: patch classifier; Stage 3: gaussian centroid; Stage 4: rendering
STAGE1_DIR: Path = ROOT / "stage1_trajectories"
STAGE2_DIR: Path = ROOT / "stage2_patch_classifier"
# New Stage 3: merge overlapping boxes (union-find)
STAGE3_DIR: Path = ROOT / "stage3_merge"
# Stage 4: Gaussian centroid refinement (moved from old Stage 3)
STAGE4_DIR: Path = ROOT / "stage4_gaussian_centroid"
# Stage 5: Rendering (moved from old Stage 4)
STAGE5_DIR: Path = ROOT / "stage5_rendering"

# Patch model (ResNet18 binary classifier). Provide a direct path, not under ROOT.
# - Set to the absolute path of your trained .pt file.
# - Exactly two classes expected (index 0 = negative, 1 = positive by default).
PATCH_MODEL_PATH: Path = Path("/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/models and other data/pyrallis gopro models resnet18/resnet18_pyrallis_gopro_best_model.pt")

# Torch/Transforms
# - IMAGENET_NORMALIZE: apply standard ImageNet mean/std.
# - NUM_WORKERS: DataLoader workers (not heavily used here; transforms are in-memory).
IMAGENET_NORMALIZE: bool = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
NUM_WORKERS: int = 0

# Stage 2 classifier (patch model)
# - STAGE2_INPUT_SIZE: model input side length (square resize).
# - STAGE2_BATCH_SIZE: inference batch size.
# - *_POSITIVE_CLASS_INDEX: index of the positive class logit in model output.
# - *_POSITIVE_THRESHOLD: probability threshold to label positive.
# - STAGE2_DEVICE: 'auto' | 'cpu' | 'cuda' | 'mps' (auto chooses best available)
STAGE2_INPUT_SIZE: int = 10  # match training (10x10 patches)
STAGE2_BATCH_SIZE: int = 4096
STAGE2_POSITIVE_CLASS_INDEX: int = 1
STAGE2_POSITIVE_THRESHOLD: float = 0.96
STAGE2_DEVICE: str = 'auto'

# Stage 2 (patch classifier) — uses STAGE2_* parameters above

# General
# - VIDEO_EXTS: file extensions to treat as videos.
# - RUN_PRE_RUN_CLEANUP: run Stage 0 automatically before processing.
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
RUN_PRE_RUN_CLEANUP: bool = True

# Global switch to disable saving of non-essential artifacts ("extras").
# - If False, stages avoid writing debug/visualization crops/videos where possible.
# - Stage 1 essentials: the two CSVs it produces.
# - Stage 2 essentials: the patches CSV; crops are extras.
# - Stage 3 essentials: merged CSV (no extras).
# - Stage 4 essentials: refined CSV; crops are extras.
# - Stage 5 essentials: rendered video (no extras).
SAVE_EXTRAS: bool = True

# Frame cap for fast iteration / testing
# - MAX_FRAMES: if an integer, process only the first N frames of each video.
#               If None, process the entire video.
MAX_FRAMES: int | None = 500



# Stage 1 — Threshold → BG-sub → Long-exposure OR → CC
# Thresholding (applied to grayscale)
THRESHOLD_8BIT: int = 70           # 0..255

# Background subtraction (OpenCV MOG2)
BGS_DETECT_SHADOWS: bool = True     # shadows get value 127
BGS_HISTORY: int = 1000
BGS_LEARNING_RATE: float = -1.0     # -1 => OpenCV decides

# Long-exposure (OR) from FG mask video
LONG_EXP_START_FRAME: int = 30      # skip early frames for BG model warm-up
FG_MASK_THRESHOLD: int = 127        # treat pixels >= this as FG (ignore shadows=127)
LONG_EXP_DILATE_ITERS: int = 1      # 0=off; 1–2 thicken trails slightly
LONG_EXP_DILATE_KERNEL: int = 3     # odd size (3/5/7)
LONG_EXP_BLUR_KSIZE: int = 0        # 0=off; else odd (3/5) slight blur before OR
LONG_EXP_CHUNK_SIZE: int = 500      # build one long-exposure image per N frames
STAGE1_SKIP_BG_SUB: bool = False    # If True, skip MOG2; OR directly from thresholded video

# Connected components (area filter)
CC_MIN_AREA: int = 5               # keep components with area >= this
CC_MAX_AREA: int = 3000             # 0 = no upper cap; else limit very large blobs

# Stage 1 outputs toggles
STAGE1_SAVE_OVERLAY: bool = False   # also save per-chunk overlay on a video frame
STAGE1_OVERLAY_DRAW_CC: bool = True # draw CC boxes+ids on overlay if saved
STAGE1_SAVE_PATCHES: bool = True    # save per-component per-time patches (10x10 PNGs)
STAGE1_WRITE_PER_FRAME_CSV: bool = True  # write telemetry CSV (x,y,t,global_id)

# Legacy aliases for compatibility (do not modify)
CHUNK_SIZE: int = LONG_EXP_CHUNK_SIZE
TRAILS_INTENSITY_THRESHOLD: int = FG_MASK_THRESHOLD

# Stage 2 — patch size used to crop from frames
# - The classifier sees a resized version (STAGE2_INPUT_SIZE); this controls crop size from frames.
PATCH_SIZE_PX: int = 10

# Stage 3 — Gaussian centroid refinement
# - GAUSS_SIGMA: 0 for uniform centroid; >0 for Gaussian-weighted intensity centroid.
GAUSS_SIGMA: float = 1.0

# Stage 4 — Rendering
# - RENDER_CODEC: e.g., "mp4v", "avc1", "MJPG".
# - RENDER_FPS_HINT: None to use source fps; else override.
RENDER_CODEC: str = "mp4v"
RENDER_FPS_HINT: float | None = None
# - RENDER_BOX_THICKNESS: thickness of bboxes in rendered videos (Stage 1 overlay and Stage 4)
RENDER_BOX_THICKNESS: int = 1

# Stage 3 — Merge (union-find) params
MERGE_DIST_THRESHOLD_PX: float = 10.0





def list_videos() -> List[Path]:
    """Return all video files under ORIGINAL_VIDEOS_DIR."""
    ORIGINAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for p in sorted(ORIGINAL_VIDEOS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    return out


__all__ = [
    # paths
    "ROOT",
    "ORIGINAL_VIDEOS_DIR",
    "STAGE1_DIR",
    "STAGE2_DIR",
    "STAGE3_DIR",
    "STAGE4_DIR",
    "STAGE5_DIR",
    # models
    "PATCH_MODEL_PATH",
    # params
    "RUN_PRE_RUN_CLEANUP",
    "SAVE_EXTRAS",
    "MAX_FRAMES",
    "LONG_EXP_CHUNK_SIZE",
    "THRESHOLD_8BIT",
    "LONG_EXP_START_FRAME",
    "FG_MASK_THRESHOLD",
    "LONG_EXP_DILATE_ITERS",
    "LONG_EXP_DILATE_KERNEL",
    "LONG_EXP_BLUR_KSIZE",
    "BGS_DETECT_SHADOWS",
    "BGS_HISTORY",
    "BGS_LEARNING_RATE",
    "STAGE1_SKIP_BG_SUB",
    # legacy aliases
    "CHUNK_SIZE",
    "TRAILS_INTENSITY_THRESHOLD",
    "CC_MIN_AREA",
    "CC_MAX_AREA",
    "PATCH_SIZE_PX",
    # torch + classification params
    "IMAGENET_NORMALIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NUM_WORKERS",
    "STAGE2_INPUT_SIZE",
    "STAGE2_BATCH_SIZE",
    "STAGE2_POSITIVE_CLASS_INDEX",
    "STAGE2_POSITIVE_THRESHOLD",
    "STAGE2_DEVICE",
    "GAUSS_SIGMA",
    "RENDER_CODEC",
    "RENDER_FPS_HINT",
    "RENDER_BOX_THICKNESS",
    "MERGE_DIST_THRESHOLD_PX",
    "STAGE1_SAVE_OVERLAY",
    "STAGE1_SAVE_PATCHES",
    "STAGE1_OVERLAY_DRAW_CC",
    "STAGE1_WRITE_PER_FRAME_CSV",
    # helpers
    "list_videos",
]
