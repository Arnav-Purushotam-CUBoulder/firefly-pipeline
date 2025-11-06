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
ROOT: str | Path = "/home/guest/Desktop/arnav's files/day time pipeline inference output data"

# Normalize ROOT to a Path object even if provided as a string
if not isinstance(ROOT, Path):
    ROOT = Path(str(ROOT)).expanduser()

# Input videos folder (inside ROOT)
# - Place any number of input videos here.
ORIGINAL_VIDEOS_DIR: Path = ROOT / "original videos"

# Stage output roots (one folder per stage). Each stage writes under its own dir.
# Stage 1: detection (cuCIM); Stage 2: patch classifier; Stage 3: gaussian centroid; Stage 4: rendering
STAGE1_DIR: Path = ROOT / "stage1_detect"
STAGE2_DIR: Path = ROOT / "stage2_patch_classifier"
STAGE3_DIR: Path = ROOT / "stage3_gaussian_centroid"
STAGE4_DIR: Path = ROOT / "stage4_rendering"

# Patch model (ResNet18 binary classifier). Provide a direct path, not under ROOT.
# - Set to the absolute path of your trained .pt file.
# - Exactly two classes expected (index 0 = negative, 1 = positive by default).
PATCH_MODEL_PATH: Path = Path("/home/guest/Desktop/arnav's files/day time pipeline model/resnet18_pyrallis_gopro_best_model.pt")

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
STAGE2_POSITIVE_THRESHOLD: float = 0.98
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
MAX_FRAMES: int | None = 100

# Stage 1 — cuCIM GPU blob detector (mirrors night-time pipeline)
# - CUCIM_DETECTOR: 'log' | 'dog' | 'doh'
# - CUCIM_MIN_SIGMA/Max/Num/SigmaRatio: scale-space parameters
# - CUCIM_THRESHOLD: relative threshold per scale (0..1 of per-scale max)
# - CUCIM_OVERLAP: 0..1 IoU for NMS-merge in cuCIM
# - CUCIM_LOG_SCALE: whether sigma is logarithmically spaced (LoG only)
# - CUCIM_MIN_AREA_PX: min bbox area to keep
# - CUCIM_MAX_AREA_SCALE: fraction of frame area for max area (<=0 => no max)
# - CUCIM_PAD_PX: padding applied around the square box derived from sigma
# - CUCIM_USE_CLAHE/CLIP/TILE: optional CLAHE preprocessing
# - CUCIM_BATCH_SIZE: frames per GPU batch
CUCIM_DETECTOR: str = 'log'
CUCIM_MIN_SIGMA: float = 0.75
CUCIM_MAX_SIGMA: float = 4.0
CUCIM_NUM_SIGMA: int = 10
CUCIM_SIGMA_RATIO: float = 1.6
CUCIM_THRESHOLD: float = 0.08
CUCIM_OVERLAP: float = 0.5
CUCIM_LOG_SCALE: bool = False
CUCIM_MIN_AREA_PX: float = 4
CUCIM_MAX_AREA_SCALE: float = 1.0
CUCIM_PAD_PX: int = 2
CUCIM_USE_CLAHE: bool = True
CUCIM_CLAHE_CLIP: float = 2.0
CUCIM_CLAHE_TILE = 8
CUCIM_BATCH_SIZE: int = 50  # number of frames to send to GPU at once
# Optional: keep only detections whose mean grayscale intensity (on the
# detection grayscale used for cuCIM) is >= this 0..255 threshold. 0 disables.
CUCIM_MIN_MEAN_INTENSITY_U8: int = 120

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
    # cuCIM detection params (Stage 1)
    "CUCIM_DETECTOR",
    "CUCIM_MIN_SIGMA",
    "CUCIM_MAX_SIGMA",
    "CUCIM_NUM_SIGMA",
    "CUCIM_SIGMA_RATIO",
    "CUCIM_THRESHOLD",
    "CUCIM_OVERLAP",
    "CUCIM_LOG_SCALE",
    "CUCIM_MIN_AREA_PX",
    "CUCIM_MAX_AREA_SCALE",
    "CUCIM_PAD_PX",
    "CUCIM_USE_CLAHE",
    "CUCIM_CLAHE_CLIP",
    "CUCIM_CLAHE_TILE",
    "CUCIM_BATCH_SIZE",
    "CUCIM_MIN_MEAN_INTENSITY_U8",
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
