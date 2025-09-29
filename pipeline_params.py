#!/usr/bin/env python3
"""Centralized configuration values for the firefly detection pipeline."""

from pathlib import Path

# Root & I/O locations
ROOT = Path("/home/guest/Desktop/arnav's files/firefly pipeline inference data/tremulans inference")

DIR_ORIG_VIDEOS = ROOT / 'original videos'
DIR_BGS_VIDEOS = ROOT / 'BS videos'
DIR_CSV = ROOT / 'csv files'
DIR_OUT_BGS = ROOT / 'BS initial output annotated videos'
DIR_OUT_ORIG = ROOT / 'original initial output annotated videos'
DIR_OUT_ORIG_10 = ROOT / 'original 10px overlay annotated videos'
DIR_STAGE8_CROPS = ROOT / 'stage8 crops'
DIR_STAGE9_OUT = ROOT / 'stage9 validation'
DIR_STAGE10_OUT = ROOT / 'stage10 overlay videos'
DIR_STAGE8_9_OUT = ROOT / 'stage8.9 gt centroid crops'

# Audit toggle
ENABLE_AUDIT = False

# Global knobs / flags
MAX_FRAMES = None
BBOX_THICKNESS = 1
DRAW_BACKGROUND_BOXES = True

# Stage toggles
RUN_STAGE1 = True
RUN_STAGE2 = True
RUN_STAGE3 = True
RUN_STAGE4 = True
RUN_STAGE5 = True
RUN_STAGE6 = True
RUN_STAGE7 = True
RUN_STAGE8 = True
RUN_STAGE8_5 = True
RUN_STAGE8_6 = True
RUN_STAGE8_7 = True
RUN_STAGE8_9 = False
RUN_STAGE8_5_AFTER_8_7 = True
RUN_STAGE9 = False
RUN_STAGE10 = True
RUN_STAGE11 = True
RUN_STAGE12 = True
RUN_STAGE13 = True
RUN_STAGE14 = True
RUN_PRE_RUN_CLEANUP = True

# Stage-1 variant selection
STAGE1_VARIANT = 'cucim'
# Optional: override Stage-1 just for Stage 8.6 replay
# e.g., use 'cc_cuda' for a fast blackout pass while keeping 'cucim' for the main pass
STAGE8_6_STAGE1_IMPL = STAGE1_VARIANT

# Stage-1 cuCIM GPU blob detector
CUCIM_DETECTOR = 'log'  # 'log' | 'dog' | 'doh'
CUCIM_MIN_SIGMA = 0.75
CUCIM_MAX_SIGMA = 4.0
CUCIM_NUM_SIGMA = 10
CUCIM_SIGMA_RATIO = 1.6
CUCIM_THRESHOLD = 0.08
CUCIM_OVERLAP = 0.5
CUCIM_LOG_SCALE = False
CUCIM_MIN_AREA_PX = 4
CUCIM_MAX_AREA_SCALE = 1.0
CUCIM_PAD_PX = 2
CUCIM_USE_CLAHE = True
CUCIM_CLAHE_CLIP = 2.0
CUCIM_CLAHE_TILE = 8
CUCIM_BATCH_SIZE = 250     #number of frames to be sent to the gpu at once

# Stage-1 CUDA CC tuning
CC_BATCH_SIZE = 64
CC_PREPROC_BACKEND = 'opencv_cuda'

CC_MIN_AREA_PX = 1
CC_MAX_AREA_SCALE = 1.00
CC_USE_CLAHE = True
CC_CLAHE_CLIP = 2.0
CC_CLAHE_TILE = 8
CC_USE_TOPHAT = True
CC_TOPHAT_KSIZE = 11
CC_USE_DOG = True
CC_DOG_SIGMA1 = 0.6
CC_DOG_SIGMA2 = 1.2
CC_ADAPTIVE_C = 1.0

CC_THRESHOLD_METHOD = 'fixed'
CC_FIXED_THRESHOLD = 90
CC_OPEN_KSIZE = 1
CC_CONNECTIVITY = 8

# Stage-1 SimpleBlobDetector knobs
SBD_MIN_AREA_PX = 0.5
SBD_MAX_AREA_SCALE = 1.0
SBD_MIN_DIST = 0.25
SBD_MIN_REPEAT = 1
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
USE_TOPHAT = False
TOPHAT_KSIZE = 7
USE_DOG = False
DOG_SIGMA1 = 0.8
DOG_SIGMA2 = 1.6

# Stage 2 — centroid recenter threshold
BRIGHT_MAX_THRESHOLD = 50

# Stage 3 — area filter
AREA_THRESHOLD_PX = 6

# Stage 4 — CNN classify/filter
USE_CNN_FILTER = True
CNN_MODEL_PATH = Path("/home/guest/Desktop/arnav's files/firefly pipeline models/resnet18_Tremulans_best_model.pt")
CNN_BACKBONE = 'resnet18'
CNN_CLASS_TO_KEEP = 1
CNN_PATCH_W = 10
CNN_PATCH_H = 10
FIREFLY_CONF_THRESH = 0.5
DROP_BACKGROUND_ROWS = False
IMAGENET_NORMALIZE = False
PRINT_LOAD_STATUS = True
FAIL_IF_WEIGHTS_MISSING = True
DEBUG_SAVE_PATCHES_DIR = None

# Stage 7 — union-find prune
STAGE7_DIST_THRESHOLD_PX = 20.0
STAGE7_VERBOSE = True

# Stage 8 — Gaussian centroid recenter
STAGE8_PATCH_W = 10
STAGE8_PATCH_H = 10
STAGE8_GAUSSIAN_SIGMA = 0.0
STAGE8_VERBOSE = True

# Stage 8.5 / 9 shared
MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION = 20

# Stage 8.6 — neighbor hunt
STAGE8_6_RUNS = 1
STAGE8_6_DEDUPE_PX = 4.0
STAGE8_6_DISABLE_CLAHE = False

# Stage 8.7 — large-flash fixer
STAGE8_7_INTENSITY_THR = 70
STAGE8_7_DEDUPE_PX = 10.0
STAGE8_7_MIN_SQUARE_AREA_PX = 75
STAGE8_7_GAUSSIAN_SIGMA = STAGE8_GAUSSIAN_SIGMA

# Stage 8.9 — GT recenter
STAGE8_9_CROP_W = 10
STAGE8_9_CROP_H = 10
STAGE8_9_GAUSSIAN_SIGMA = STAGE8_GAUSSIAN_SIGMA

# Stage 9 — validation vs ground truth
GT_CSV_PATH = ROOT / 'ground truth' / 'gt.csv'
GT_T_OFFSET = 4000
DIST_THRESHOLDS_PX = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
STAGE9_CROP_W = 10
STAGE9_CROP_H = 10
STAGE9_ONLY_FIREFLY_ROWS = True
STAGE9_SHOW_PER_FRAME = False
STAGE9_MODEL_PATH = CNN_MODEL_PATH
STAGE9_BACKBONE = CNN_BACKBONE
STAGE9_IMAGENET_NORM = IMAGENET_NORMALIZE
STAGE9_PRINT_LOAD_STATUS = PRINT_LOAD_STATUS
STAGE9_GT_DEDUPE_DIST_PX = 4.0

# Stage 10 — overlay
STAGE10_GT_BOX_W = STAGE9_CROP_W
STAGE10_GT_BOX_H = STAGE9_CROP_H

# Rendering options (Stage 5/6)
SAVE_ANN_BG = True
SAVE_ANN_ORIG = True
SAVE_ANN_10PX = True

# File discovery
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

CLEANUP_KEEP_DIRS = ('ground truth', 'original videos')
CLEANUP_GT_FILENAME = 'gt.csv'

__all__ = [name for name in globals() if name.isupper()]
