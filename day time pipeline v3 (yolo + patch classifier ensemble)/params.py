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
ROOT: str | Path = "~/Desktop/arnav's files/day time pipeline inference output data"

# Normalize ROOT to a Path object even if provided as a string
if not isinstance(ROOT, Path):
    ROOT = Path(str(ROOT)).expanduser()

# Input videos folder (inside ROOT)
# - Place any number of input videos here.
ORIGINAL_VIDEOS_DIR: Path = ROOT / "original videos"

# Stage output roots (one folder per stage). Each stage writes under its own dir.
# Stage 1: long-exposure images; Stage 2: YOLO detections CSVs;
# Stage 3: patch classifier; Stage 4: rendered videos; Stage 5+: post-pipeline test suite outputs.
STAGE1_DIR: Path = ROOT / "stage1_long_exposure"
STAGE2_DIR: Path = ROOT / "stage2_yolo_detections"
STAGE3_DIR: Path = ROOT / "stage3_patch_classifier"
STAGE4_DIR: Path = ROOT / "stage4_rendering"
STAGE5_DIR: Path = ROOT / "stage5 validation"

# General
# - VIDEO_EXTS: file extensions to treat as videos.
# - RUN_PRE_RUN_CLEANUP: run Stage 0 automatically before processing.
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
RUN_PRE_RUN_CLEANUP: bool = True

# Frame cap for fast iteration / testing
# - MAX_FRAMES: if an integer, process only the first N frames of each video
#               when forming long-exposure images. If None, process the full video.
MAX_FRAMES: int | None = None

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
# Local model directory for this v3 pipeline.
MODEL_DIR: Path = Path(__file__).resolve().parent / "v3 day time pipeline models"
# Path to trained YOLO weights (.pt)
YOLO_MODEL_WEIGHTS: Path = MODEL_DIR / "yolo_best.pt"

# Inference params
# - YOLO_IMG_SIZE: None → auto from input image size (max(H,W))
# - YOLO_CONF_THRES: confidence threshold
# - YOLO_IOU_THRES: IoU threshold for NMS
# - YOLO_DEVICE: 'auto' | 'cpu' | 'cuda' | 'mps' | CUDA index
# - YOLO_BATCH_SIZE: number of long-exposure images per model.predict call.
# - YOLO_HALF_ON_CUDA: use FP16 inference on CUDA devices.
YOLO_IMG_SIZE: int | None = None
YOLO_CONF_THRES: float = 0.01
YOLO_IOU_THRES: float = 0.15
YOLO_DEVICE: str | int | None = "auto"
YOLO_BATCH_SIZE: int = 8
YOLO_HALF_ON_CUDA: bool = True

# Stage 3 — patch classifier on per-frame crops
# Patch model (ResNet18 binary classifier). Provide a direct path.
PATCH_MODEL_PATH: Path = MODEL_DIR / "resnet18_pyrallis_gopro_best_model_v3.pt"

# Torch / transforms
IMAGENET_NORMALIZE: bool = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Stage 3 classifier settings
# - STAGE3_INPUT_SIZE: model input side length (square resize).
# - STAGE3_BATCH_SIZE_GPU/CPU: batch sizes per device type.
# - *_POSITIVE_CLASS_INDEX: index of positive class in logits.
# - *_POSITIVE_THRESHOLD: probability threshold to mark positive.
# - STAGE3_DEVICE: 'auto' | 'cpu' | 'cuda' | 'mps'
# - STAGE3_USE_AMP: enable mixed-precision inference on CUDA.
# - STAGE3_ENABLE_CUDNN_BENCHMARK: tune cuDNN kernels for fixed-size inference.
# - STAGE3_ALLOW_TF32: allow TF32 matmul/cuDNN on Ampere+ GPUs.
STAGE3_INPUT_SIZE: int = 10
STAGE3_BATCH_SIZE_GPU: int = 4096
STAGE3_BATCH_SIZE_CPU: int = 512
STAGE3_POSITIVE_CLASS_INDEX: int = 1
STAGE3_POSITIVE_THRESHOLD: float = 0.80
STAGE3_DEVICE: str = "auto"
STAGE3_USE_AMP: bool = True
STAGE3_ENABLE_CUDNN_BENCHMARK: bool = True
STAGE3_ALLOW_TF32: bool = True

# Stage 3 — crop size used to extract patches from frames
PATCH_SIZE_PX: int = 10

# Stage 4 — rendering
# - RENDER_CODEC: e.g. "mp4v", "avc1", "MJPG".
# - RENDER_FPS_HINT: None to use source fps; else override.
RENDER_CODEC: str = "mp4v"
RENDER_FPS_HINT: float | None = None
# If True and Stage3.1 produced `*_patches_motion_all.csv`, draw rejected
# detections in blue and kept detections in red.
STAGE4_DRAW_STAGE3_1_REJECTED: bool = False

# Stage 3.1 — trajectory grouping + intensity selection
# Groups Stage3 detections into trajectories in (x,y,t) space, then computes
# per-trajectory intensity curves and selects "flash-like" (hill-shaped) ones.
RUN_STAGE3_1_TRAJECTORY_INTENSITY_SELECTOR: bool = True
# Backward-compat name (do not use for new code)
RUN_STAGE3_1_MOTION_FILTER: bool = RUN_STAGE3_1_TRAJECTORY_INTENSITY_SELECTOR
# Max XY pixels per frame to link detections into the same trajectory.
# If a firefly moves ~10–12 px/frame, this must be > that (e.g. 15–25).
STAGE3_1_LINK_RADIUS_PX: float = 15.0
STAGE3_1_MAX_FRAME_GAP: int = 6
STAGE3_1_TIME_SCALE: float = 1.0
STAGE3_1_MIN_TRACK_POINTS: int = 3
STAGE3_1_EXPORT_TRAJECTORY_CROPS: bool = True
STAGE3_1_TRAJECTORY_CROPS_DIRNAME: str = "stage3_1_trajectory_crops"
# Plot intensity curves (one curve per trajectory) as an SVG under STAGE3_DIR/<stem>/.
STAGE3_1_PLOT_TRAJECTORY_INTENSITY: bool = True
STAGE3_1_TRAJECTORY_INTENSITY_SVG_NAME: str = "stage3_1_trajectory_intensity_curves.svg"
# Also write a second SVG containing only trajectories whose intensity curve
# range (max(sum)-min(sum)) is at least this threshold.
STAGE3_1_PLOT_TRAJECTORY_INTENSITY_HIGHVAR: bool = True
STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE: int = 3000
STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_SVG_NAME: str = "stage3_1_trajectory_intensity_curves_highvar.svg"
STAGE3_1_EXPORT_HIGHVAR_TRAJECTORY_CROPS: bool = True
STAGE3_1_HIGHVAR_TRAJECTORY_CROPS_DIRNAME: str = "stage3_1_highvar_trajectory_crops"
# "Firefly-flash-like" hill filter: curve rises then falls (unimodal-ish).
STAGE3_1_HIGHVAR_REQUIRE_HILL_SHAPE: bool = True
STAGE3_1_HILL_SMOOTH_WINDOW: int = 1              # 1 disables smoothing
STAGE3_1_HILL_MIN_UP_STEPS: int = 2               # min positive steps before peak
STAGE3_1_HILL_MIN_DOWN_STEPS: int = 2             # min negative steps after peak
STAGE3_1_HILL_MIN_MONOTONIC_FRAC: float = 0.60    # before/after peak sign consistency
STAGE3_1_HILL_PEAK_POS_MIN_FRAC: float = 0.0     # peak not too close to start
STAGE3_1_HILL_PEAK_POS_MAX_FRAC: float = 1.0     # peak not too close to end
# Render a video under STAGE3_DIR/<video_stem>/ showing Stage3.1 selections:
# - selected (hill/flash-like) boxes in red
# - rejected boxes in blue
STAGE3_1_RENDER_HIGHVAR_VIDEO: bool = True
STAGE3_1_HIGHVAR_VIDEO_NAME: str = "stage3_1_highvar_trajectories.mp4"

# Stage 3.2 — Gaussian centroid + logits CSV + annotated crops (selected only)
RUN_STAGE3_2: bool = True
STAGE3_2_DIRNAME: str = "stage3_2"
STAGE3_2_GAUSSIAN_SIGMA: float = 1.0   # 0 => plain intensity centroid
STAGE3_2_SAVE_ANNOTATED_CROPS: bool = True
STAGE3_2_MARK_CENTROID_RED_PIXEL: bool = True


# ------------------------------------------------------------------
# Post-pipeline test suite (Stage 5–9): GT validation + overlays + analysis
# ------------------------------------------------------------------
# These stages are implemented inside this v3 pipeline folder.
RUN_STAGE5_VALIDATE: bool = True
RUN_STAGE6_OVERLAY: bool = True
RUN_STAGE7_FN_ANALYSIS: bool = True
RUN_STAGE8_FP_ANALYSIS: bool = True
RUN_STAGE9_DETECTION_SUMMARY: bool = True

# Test output directories (under ROOT)
# - Stage 5 writes per-video folders under STAGE5_DIR/<video_stem>/.
STAGE6_DIR: Path = ROOT / "stage6 overlay videos"
STAGE7_DIR: Path = ROOT / "stage7 fn analysis"
STAGE8_DIR: Path = ROOT / "stage8 fp analysis"
STAGE9_DIR: Path = ROOT / "stage9 detection summary"

# Ground truth (GT) configuration
# Supported GT filenames under GT_CSV_DIR (matched per input video stem):
# - gt_<video_stem>.csv            (your common convention)
# - <video_stem>.csv
# - <video_stem>_gt.csv
# - gt.csv                         (single-video fallback)
GT_CSV_DIR: Path = ROOT / "ground truth"
GT_CSV_PATH: Path | None = None
GT_T_OFFSET: int = 0

# Validation sweep thresholds (pixels)
DIST_THRESHOLDS_PX: list[float] = [10.0]

# Stage 5 crop size (also used for Stage 6/7/8 box visuals)
STAGE5_CROP_W: int = int(PATCH_SIZE_PX)
STAGE5_CROP_H: int = int(PATCH_SIZE_PX)
STAGE6_GT_BOX_W: int = STAGE5_CROP_W
STAGE6_GT_BOX_H: int = STAGE5_CROP_H
OVERLAY_BOX_THICKNESS: int = 1

# Stage 5 options
STAGE5_ONLY_FIREFLY_ROWS: bool = True
STAGE5_SHOW_PER_FRAME: bool = False
STAGE5_MODEL_PATH: Path | None = PATCH_MODEL_PATH  # used only for FN confidence scoring
STAGE5_BACKBONE: str = "resnet18"
STAGE5_IMAGENET_NORM: bool = bool(IMAGENET_NORMALIZE)
STAGE5_PRINT_LOAD_STATUS: bool = True

# Stage 5 GT filtering + dedupe (same features as the copied validator)
STAGE5_GT_AREA_THRESHOLD_PX: int = 4
STAGE5_GT_BRIGHT_MAX_THRESHOLD: int = 50
STAGE5_MIN_PIXEL_BRIGHTNESS_FOR_AREA_CALC: int = 50
STAGE5_GT_DEDUPE_DIST_PX: float = 2.0


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
    "MODEL_DIR",
    "YOLO_MODEL_WEIGHTS",
    "YOLO_IMG_SIZE",
    "YOLO_CONF_THRES",
    "YOLO_IOU_THRES",
    "YOLO_DEVICE",
    "YOLO_BATCH_SIZE",
    "YOLO_HALF_ON_CUDA",
    # patch classifier + render
    "PATCH_MODEL_PATH",
    "IMAGENET_NORMALIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "STAGE3_INPUT_SIZE",
    "STAGE3_BATCH_SIZE_GPU",
    "STAGE3_BATCH_SIZE_CPU",
    "STAGE3_POSITIVE_CLASS_INDEX",
    "STAGE3_POSITIVE_THRESHOLD",
    "STAGE3_DEVICE",
    "STAGE3_USE_AMP",
    "STAGE3_ENABLE_CUDNN_BENCHMARK",
    "STAGE3_ALLOW_TF32",
    "PATCH_SIZE_PX",
    "RENDER_CODEC",
    "RENDER_FPS_HINT",
    "STAGE4_DRAW_STAGE3_1_REJECTED",
    "RUN_STAGE3_1_TRAJECTORY_INTENSITY_SELECTOR",
    "RUN_STAGE3_1_MOTION_FILTER",
    "STAGE3_1_LINK_RADIUS_PX",
    "STAGE3_1_MAX_FRAME_GAP",
    "STAGE3_1_TIME_SCALE",
    "STAGE3_1_MIN_TRACK_POINTS",
    "STAGE3_1_EXPORT_TRAJECTORY_CROPS",
    "STAGE3_1_TRAJECTORY_CROPS_DIRNAME",
    "STAGE3_1_PLOT_TRAJECTORY_INTENSITY",
    "STAGE3_1_TRAJECTORY_INTENSITY_SVG_NAME",
    "STAGE3_1_PLOT_TRAJECTORY_INTENSITY_HIGHVAR",
    "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE",
    "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_SVG_NAME",
    "STAGE3_1_EXPORT_HIGHVAR_TRAJECTORY_CROPS",
    "STAGE3_1_HIGHVAR_TRAJECTORY_CROPS_DIRNAME",
    "STAGE3_1_HIGHVAR_REQUIRE_HILL_SHAPE",
    "STAGE3_1_HILL_SMOOTH_WINDOW",
    "STAGE3_1_HILL_MIN_UP_STEPS",
    "STAGE3_1_HILL_MIN_DOWN_STEPS",
    "STAGE3_1_HILL_MIN_MONOTONIC_FRAC",
    "STAGE3_1_HILL_PEAK_POS_MIN_FRAC",
    "STAGE3_1_HILL_PEAK_POS_MAX_FRAC",
    "STAGE3_1_RENDER_HIGHVAR_VIDEO",
    "STAGE3_1_HIGHVAR_VIDEO_NAME",
    "RUN_STAGE3_2",
    "STAGE3_2_DIRNAME",
    "STAGE3_2_GAUSSIAN_SIGMA",
    "STAGE3_2_SAVE_ANNOTATED_CROPS",
    "STAGE3_2_MARK_CENTROID_RED_PIXEL",
    # post-pipeline test suite (Stage 5–9)
    "RUN_STAGE5_VALIDATE",
    "RUN_STAGE6_OVERLAY",
    "RUN_STAGE7_FN_ANALYSIS",
    "RUN_STAGE8_FP_ANALYSIS",
    "RUN_STAGE9_DETECTION_SUMMARY",
    "STAGE6_DIR",
    "STAGE7_DIR",
    "STAGE8_DIR",
    "STAGE9_DIR",
    "GT_CSV_DIR",
    "GT_CSV_PATH",
    "GT_T_OFFSET",
    "DIST_THRESHOLDS_PX",
    "STAGE5_CROP_W",
    "STAGE5_CROP_H",
    "STAGE6_GT_BOX_W",
    "STAGE6_GT_BOX_H",
    "OVERLAY_BOX_THICKNESS",
    "STAGE5_ONLY_FIREFLY_ROWS",
    "STAGE5_SHOW_PER_FRAME",
    "STAGE5_MODEL_PATH",
    "STAGE5_BACKBONE",
    "STAGE5_IMAGENET_NORM",
    "STAGE5_PRINT_LOAD_STATUS",
    "STAGE5_GT_AREA_THRESHOLD_PX",
    "STAGE5_GT_BRIGHT_MAX_THRESHOLD",
    "STAGE5_MIN_PIXEL_BRIGHTNESS_FOR_AREA_CALC",
    "STAGE5_GT_DEDUPE_DIST_PX",
    # helpers
    "list_videos",
]
