#!/usr/bin/env python3
"""
Day-time pipeline v3 (long-exposure + YOLO detection) parameters.

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
ROOT: str | Path = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/v3 day time pipeline inference data/test1'

# Normalize ROOT to a Path object even if provided as a string
if not isinstance(ROOT, Path):
    ROOT = Path(str(ROOT)).expanduser()

# Input videos folder (inside ROOT)
# - Place any number of input videos here.
ORIGINAL_VIDEOS_DIR: Path = ROOT / "original videos"

# Stage output roots (one folder per stage). Each stage writes under its own dir.
# Stage 1: long-exposure images; Stage 2: YOLO detections CSVs.
STAGE1_DIR: Path = ROOT / "stage1_long_exposure"
STAGE2_DIR: Path = ROOT / "stage2_yolo_detections"

# General
# - VIDEO_EXTS: file extensions to treat as videos.
# - RUN_PRE_RUN_CLEANUP: run Stage 0 automatically before processing.
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
RUN_PRE_RUN_CLEANUP: bool = True

# Frame cap for fast iteration / testing
# - MAX_FRAMES: if an integer, process only the first N frames of each video
#               when forming long-exposure images. If None, process the full video.
MAX_FRAMES: int | None = 500

# Stage 1 — long-exposure generation from raw video
# - LONG_EXPOSURE_MODE: 'lighten' | 'average' | 'trails'
# - INTERVAL_FRAMES: None/<=0 → single image over [0..N-1]; else chunk size.
# - PROGRESS_EVERY: progress bar update frequency in frames.
LONG_EXPOSURE_MODE: str = "lighten"
INTERVAL_FRAMES: int | None = 100
PROGRESS_EVERY: int = 50

# Average-mode options
AVERAGE_USE_GAMMA: bool = False
AVERAGE_GAMMA: float = 2.2

# Trails-mode options
BGS_DETECT_SHADOWS: bool = True
BGS_HISTORY: int = 1000
BGS_LEARNING_RATE: float = -1.0          # -1 => OpenCV decides
FG_MASK_THRESHOLD: int = 200             # ignore shadows=127
TRAILS_DILATE_ITERS: int = 1             # 0=off
TRAILS_DILATE_KERNEL: int = 3            # odd size
TRAILS_BLUR_KSIZE: int = 0               # 0=off; else odd size
TRAILS_OVERLAY: bool = False             # overlay trails on first frame
TRAILS_OVERLAY_ALPHA: float = 0.70

# Stage 2 — YOLO detection on long-exposure images
# Path to trained YOLO weights (.pt)
YOLO_MODEL_WEIGHTS: Path = Path(
    "/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/"
    "background subtraction detection method/actual background subtraction code/"
    "forresti, fixing FPs and box overlap/Proof of concept code/"
    "test1/pyrallis_exp/raw long exposure exp/yolo train output data/"
    "runs_firefly/train_20251030_2129592/weights/best.pt"
)

# Inference params
# - YOLO_IMG_SIZE: None → auto from input image size (max(H,W))
# - YOLO_CONF_THRES: confidence threshold
# - YOLO_IOU_THRES: IoU threshold for NMS
# - YOLO_DEVICE: 'auto' | 'cpu' | 'cuda' | 'mps' | CUDA index
YOLO_IMG_SIZE: int | None = None
YOLO_CONF_THRES: float = 0.01
YOLO_IOU_THRES: float = 0.15
YOLO_DEVICE: str | int | None = "cpu"


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
    # general
    "RUN_PRE_RUN_CLEANUP",
    "VIDEO_EXTS",
    "MAX_FRAMES",
    # long-exposure params
    "LONG_EXPOSURE_MODE",
    "INTERVAL_FRAMES",
    "PROGRESS_EVERY",
    "AVERAGE_USE_GAMMA",
    "AVERAGE_GAMMA",
    "BGS_DETECT_SHADOWS",
    "BGS_HISTORY",
    "BGS_LEARNING_RATE",
    "FG_MASK_THRESHOLD",
    "TRAILS_DILATE_ITERS",
    "TRAILS_DILATE_KERNEL",
    "TRAILS_BLUR_KSIZE",
    "TRAILS_OVERLAY",
    "TRAILS_OVERLAY_ALPHA",
    # YOLO params
    "YOLO_MODEL_WEIGHTS",
    "YOLO_IMG_SIZE",
    "YOLO_CONF_THRES",
    "YOLO_IOU_THRES",
    "YOLO_DEVICE",
    # helpers
    "list_videos",
]

