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
STAGE3_DIR: Path = ROOT / "stage3_gaussian_centroid"
STAGE4_DIR: Path = ROOT / "stage4_rendering"

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
STAGE2_BATCH_SIZE: int = 64
STAGE2_POSITIVE_CLASS_INDEX: int = 1
STAGE2_POSITIVE_THRESHOLD: float = 0.75
STAGE2_DEVICE: str = 'auto'

# Stage 2 (patch classifier) — uses STAGE2_* parameters above

# General
# - VIDEO_EXTS: file extensions to treat as videos.
# - RUN_PRE_RUN_CLEANUP: run Stage 0 automatically before processing.
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
RUN_PRE_RUN_CLEANUP: bool = True

# Frame cap for fast iteration / testing
# - MAX_FRAMES: if an integer, process only the first N frames of each video.
#               If None, process the entire video.
MAX_FRAMES: int | None = 500

# Stage 1 — Long-exposure OR image per chunk, CC candidates
# - CHUNK_SIZE: build a trails image every CHUNK_SIZE frames.
# - TRAILS_INTENSITY_THRESHOLD: grayscale threshold (0..255) to consider a pixel "bright".
# - CC_MIN_AREA/CC_MAX_AREA: area filter for connected components (0 => unlimited max).
CHUNK_SIZE: int = 500
TRAILS_INTENSITY_THRESHOLD: int = 230
CC_MIN_AREA: int = 20
CC_MAX_AREA: int = 10_000

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
    # models
    "PATCH_MODEL_PATH",
    # params
    "RUN_PRE_RUN_CLEANUP",
    "MAX_FRAMES",
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
    # helpers
    "list_videos",
]
