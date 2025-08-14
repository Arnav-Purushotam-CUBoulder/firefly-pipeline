#!/usr/bin/env python3
"""
Orchestrator: holds ALL global variables and calls each stage.

Pipeline:
  1) detect_blobs_to_csv():  ORIGINAL video → dynamic boxes → CSV (frame,x,y,w,h)
  2) recenter_boxes_with_centroid(): original video + CSV → recenter with intensity-weighted centroid → update same CSV
  3) filter_boxes_by_area(): drop rows with w*h < AREA_THRESHOLD_PX, save a snapshot CSV
  4) classify_and_filter_csv(): label rows via CNN (adds 'class' column; may drop/keep background)
  5) prune_overlaps_keep_heaviest_unionfind(): group by centroid distance, keep heaviest RGB-sum per group
  6) render_from_csv(): draw boxes from CSVs (honors 'class' + draw_background)
  7) render_fixed_10px_from_csv(): draw fixed 10×10 boxes centered on each detection
  8) recenter_gaussian_centroid(): refine centers with (Gaussian-optional) intensity centroid, overwrite CSV
  9) stage9_validate_against_gt(): validate predictions vs ground truth, save FP/TP/FN crops & metrics
  10) overlay_gt_vs_model(): render GT (GREEN), Model (RED), overlap (YELLOW) on a single video (runs only if Stage 9 ran)
      and render per-threshold videos (TP=YELLOW, FP=RED, FN=GREEN).
"""

from pathlib import Path
import sys

# ──────────────────────────────────────────────────────────────
# Imports for each stage
# ──────────────────────────────────────────────────────────────
from stage1_detect import detect_blobs_to_csv
from stage2_recenter import recenter_boxes_with_centroid
from stage3_area_filter import filter_boxes_by_area
from stage4_cnn_filter import classify_and_filter_csv
from stage5_render import render_from_csv
from stage6_10px_renderer import render_fixed_10px_from_csv
from stage7_merge import prune_overlaps_keep_heaviest_unionfind
from stage8_gaussian_centroid import recenter_gaussian_centroid
from stage9_validate import stage9_validate_against_gt
from stage10_overlay_gt_vs_model import overlay_gt_vs_model  # includes per-threshold TP/FP/FN videos

# ──────────────────────────────────────────────────────────────
# Root & I/O locations
# ──────────────────────────────────────────────────────────────
ROOT = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/orc pipeline frontalis only inference data')

# Input videos (under your root)
DIR_ORIG_VIDEOS = ROOT / 'original videos'         # put original .mp4/.avi here
DIR_BGS_VIDEOS  = ROOT / 'BS videos'               # (optional) background-subtracted versions with same basenames

# CSVs (shared across stages)
DIR_CSV         = ROOT / 'csv files'

# Renders / outputs (all under your root so everything stays together)
DIR_OUT_BGS     = ROOT / 'BS initial output annotated videos'
DIR_OUT_ORIG    = ROOT / 'original initial output annotated videos'
DIR_OUT_ORIG_10 = ROOT / 'original 10px overlay annotated videos'

# Stage 8 crops (optional diagnostic)
DIR_STAGE8_CROPS = ROOT / 'stage8 crops'

# Stage 9 outputs
DIR_STAGE9_OUT   = ROOT / 'stage9 validation'

# Stage 10 outputs
DIR_STAGE10_OUT  = ROOT / 'stage10 overlay videos'

# Ensure output/working directories exist
for d in [DIR_CSV, DIR_OUT_BGS, DIR_OUT_ORIG, DIR_OUT_ORIG_10, DIR_STAGE8_CROPS, DIR_STAGE9_OUT, DIR_STAGE10_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Global knobs / flags
# ──────────────────────────────────────────────────────────────
MAX_FRAMES = 500              # e.g., 5000 to truncate
BBOX_THICKNESS = 1
DRAW_BACKGROUND_BOXES = True   # stage 5/6

# Which stages to run
RUN_STAGE1 = True
RUN_STAGE2 = True
RUN_STAGE3 = True
RUN_STAGE4 = True
RUN_STAGE5 = True
RUN_STAGE6 = True
RUN_STAGE7 = True
RUN_STAGE8 = True
RUN_STAGE9 = True
RUN_STAGE10 = True   # will only execute if RUN_STAGE9 is also True

# ──────────────────────────────────────────────────────────────
# Stage-specific parameters
# ──────────────────────────────────────────────────────────────

# Stage 1 — detector & preproc knobs
SBD_MIN_AREA_PX     = 0.5
SBD_MAX_AREA_SCALE  = 1.0
SBD_MIN_DIST        = 0.25
SBD_MIN_REPEAT      = 1
USE_CLAHE           = True
CLAHE_CLIP          = 2.0
CLAHE_TILE          = (8, 8)
USE_TOPHAT          = False
TOPHAT_KSIZE        = 7
USE_DOG             = False
DOG_SIGMA1          = 0.8
DOG_SIGMA2          = 1.6

# Stage 2 — centroid recenter threshold (drop dim crops)
BRIGHT_MAX_THRESHOLD = 50

# Stage 3 — area filter
AREA_THRESHOLD_PX = 4

# Stage 4 — CNN classify/filter
USE_CNN_FILTER             = True
CNN_MODEL_PATH             = Path('latest frontalis code (contains post processing features from resnet forresti)/resnet frontalis models/colored_ResNet_18_Frontalis_best_model.pt')  # ← SET THIS to your .pt file
CNN_BACKBONE               = 'resnet18'
CNN_CLASS_TO_KEEP          = 1               # firefly class idx
CNN_PATCH_W                = 10
CNN_PATCH_H                = 10
FIREFLY_CONF_THRESH        = 0.5
DROP_BACKGROUND_ROWS       = False
IMAGENET_NORMALIZE         = False
PRINT_LOAD_STATUS          = True
FAIL_IF_WEIGHTS_MISSING    = True
DEBUG_SAVE_PATCHES_DIR     = None            # e.g., ROOT/'stage4_patches'

# Stage 7 — union-find prune
STAGE7_DIST_THRESHOLD_PX   = 20.0
STAGE7_VERBOSE             = True

# Stage 8 — Gaussian centroid recenter
STAGE8_PATCH_W             = 10
STAGE8_PATCH_H             = 10
STAGE8_GAUSSIAN_SIGMA      = 0.0   # 0.0 => intensity centroid; >0 => Gaussian-weighted
STAGE8_VERBOSE             = True

# Stage 9 — validation vs ground truth
GT_CSV_PATH                = ROOT / 'ground truth' / 'gt.csv'  # GT (x,y,t), t is raw & will be normalized
GT_T_OFFSET                = 9000
DIST_THRESHOLDS_PX         = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]                   # sweep
STAGE9_CROP_W              = 10
STAGE9_CROP_H              = 10
STAGE9_ONLY_FIREFLY_ROWS   = True
STAGE9_SHOW_PER_FRAME      = False
# For FN scoring inside Stage 9:
STAGE9_MODEL_PATH          = CNN_MODEL_PATH
STAGE9_BACKBONE            = CNN_BACKBONE
STAGE9_IMAGENET_NORM       = IMAGENET_NORMALIZE
STAGE9_PRINT_LOAD_STATUS   = PRINT_LOAD_STATUS

# Stage 10 — overlay
STAGE10_GT_BOX_W           = STAGE9_CROP_W   # keep consistent with Stage 9 crop size
STAGE10_GT_BOX_H           = STAGE9_CROP_H

# Rendering options (Stage 5/6)
SAVE_ANN_BG                = True
SAVE_ANN_ORIG              = True
SAVE_ANN_10PX              = True

# File discovery
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}


def _iter_videos(dir_path: Path):
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in VIDEO_EXTS])


def main():
    # Sanity echo so you always know which root is used
    print(f"[orchestrator] ROOT: {ROOT}")

    orig_videos = _iter_videos(DIR_ORIG_VIDEOS)
    if not orig_videos:
        print(f"[orchestrator] No videos found in: {DIR_ORIG_VIDEOS}")
        sys.exit(0)

    for orig_path in orig_videos:
        base = orig_path.stem
        print(f"\n=== Processing: {base} ===")

        # Optional BS video with same basename
        bs_path = None
        cand = DIR_BGS_VIDEOS / orig_path.name
        if cand.exists():
            bs_path = cand

        # CSV path for this video
        csv_path = DIR_CSV / f'{base}.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Stage 1 — detect
        if RUN_STAGE1:
            detect_blobs_to_csv(
                orig_path=orig_path,
                csv_path=csv_path,
                max_frames=MAX_FRAMES,
                sbd_min_area_px=SBD_MIN_AREA_PX,
                sbd_max_area_scale=SBD_MAX_AREA_SCALE,
                sbd_min_dist=SBD_MIN_DIST,
                sbd_min_repeat=SBD_MIN_REPEAT,
                use_clahe=USE_CLAHE,
                clahe_clip=CLAHE_CLIP,
                clahe_tile=CLAHE_TILE,
                use_tophat=USE_TOPHAT,
                tophat_ksize=TOPHAT_KSIZE,
                use_dog=USE_DOG,
                dog_sigma1=DOG_SIGMA1,
                dog_sigma2=DOG_SIGMA2,
            )

        # Stage 2 — recenter via intensity centroid (drop dim crops)
        if RUN_STAGE2:
            recenter_boxes_with_centroid(
                orig_path=orig_path,
                csv_path=csv_path,
                max_frames=MAX_FRAMES,
                bright_max_threshold=BRIGHT_MAX_THRESHOLD,
            )

        # Stage 3 — area filter (in-place) + snapshot
        if RUN_STAGE3:
            snapshot_csv = csv_path.parent / f"{csv_path.stem}_area_snapshot.csv"
            filter_boxes_by_area(
                csv_path=csv_path,
                area_threshold_px=AREA_THRESHOLD_PX,
                snapshot_csv_path=snapshot_csv,
            )

        # Stage 4 — CNN classify/filter (adds logits/confidence/class)
        if RUN_STAGE4 and USE_CNN_FILTER:
            classify_and_filter_csv(
                orig_path=orig_path,
                csv_path=csv_path,
                max_frames=MAX_FRAMES,
                use_cnn_filter=USE_CNN_FILTER,
                model_path=CNN_MODEL_PATH,
                backbone=CNN_BACKBONE,
                class_to_keep=CNN_CLASS_TO_KEEP,
                patch_w=CNN_PATCH_W,
                patch_h=CNN_PATCH_H,
                firefly_conf_thresh=FIREFLY_CONF_THRESH,
                drop_background_rows=DROP_BACKGROUND_ROWS,
                imagenet_normalize=IMAGENET_NORMALIZE,
                print_load_status=PRINT_LOAD_STATUS,
                fail_if_weights_missing=FAIL_IF_WEIGHTS_MISSING,
                debug_save_patches_dir=DEBUG_SAVE_PATCHES_DIR,
            )

        # Stage 5 — render dynamic boxes on BS and/or original
        if RUN_STAGE5:
            if SAVE_ANN_BG and bs_path is not None:
                out_bg_path = DIR_OUT_BGS / f"{base}_bs_annotated.mp4"
                render_from_csv(
                    video_path=bs_path,
                    csv_path=csv_path,
                    out_path=out_bg_path,
                    color=(0, 0, 255),
                    thickness=BBOX_THICKNESS,
                    max_frames=MAX_FRAMES,
                    draw_background=DRAW_BACKGROUND_BOXES,
                    background_color=(0, 255, 0),
                )
            if SAVE_ANN_ORIG:
                out_orig_path = DIR_OUT_ORIG / f"{base}_orig_annotated.mp4"
                render_from_csv(
                    video_path=orig_path,
                    csv_path=csv_path,
                    out_path=out_orig_path,
                    color=(0, 0, 255),
                    thickness=BBOX_THICKNESS,
                    max_frames=MAX_FRAMES,
                    draw_background=DRAW_BACKGROUND_BOXES,
                    background_color=(0, 255, 0),
                )

        # Stage 6 — render fixed 10×10 on original
        if RUN_STAGE6 and SAVE_ANN_10PX:
            out_orig_10px_path = DIR_OUT_ORIG_10 / f"{base}_orig_10px.mp4"
            render_fixed_10px_from_csv(
                video_path=orig_path,
                csv_path=csv_path,
                out_path=out_orig_10px_path,
                thickness=BBOX_THICKNESS,
                max_frames=MAX_FRAMES,
                color=(0, 0, 255),
                draw_background=DRAW_BACKGROUND_BOXES,
                background_color=(0, 255, 0),
            )

        # Stage 7 — prune overlaps (union-find keep-heaviest)
        if RUN_STAGE7:
            prune_overlaps_keep_heaviest_unionfind(
                orig_video_path=orig_path,
                csv_path=csv_path,
                dist_threshold_px=STAGE7_DIST_THRESHOLD_PX,
                max_frames=MAX_FRAMES,
                verbose=STAGE7_VERBOSE,
            )

        # Stage 8 — Gaussian centroid recenter (rewrites CSV; x,y become centers; w,h fixed to patch; adds xy_semantics='center')
        if RUN_STAGE8:
            recenter_gaussian_centroid(
                orig_video_path=orig_path,
                csv_path=csv_path,
                centroid_patch_w=STAGE8_PATCH_W,
                centroid_patch_h=STAGE8_PATCH_H,
                gaussian_sigma=STAGE8_GAUSSIAN_SIGMA,
                max_frames=MAX_FRAMES,
                verbose=STAGE8_VERBOSE,
                crop_dir=DIR_STAGE8_CROPS / base,   # optional per-video dump
            )

        # Stage 9 — validate vs ground truth (writes normalized GT to stage9 dir and copies to CSV dir; saves TP/FP/FN crops)
        ran_stage9 = False
        if RUN_STAGE9:
            out9 = DIR_STAGE9_OUT / base
            out9.mkdir(parents=True, exist_ok=True)
            stage9_validate_against_gt(
                orig_video_path=orig_path,
                pred_csv_path=csv_path,
                gt_csv_path=GT_CSV_PATH,
                out_dir=out9,
                dist_thresholds=DIST_THRESHOLDS_PX,
                crop_w=STAGE9_CROP_W,
                crop_h=STAGE9_CROP_H,
                gt_t_offset=GT_T_OFFSET,
                max_frames=MAX_FRAMES,
                only_firefly_rows=STAGE9_ONLY_FIREFLY_ROWS,
                show_per_frame=STAGE9_SHOW_PER_FRAME,
                model_path=STAGE9_MODEL_PATH,                # used only for FN confidence
                backbone=STAGE9_BACKBONE,
                imagenet_normalize=STAGE9_IMAGENET_NORM,
                print_load_status=STAGE9_PRINT_LOAD_STATUS,
            )
            ran_stage9 = True

        # Stage 10 — overlay GT (GREEN), model (RED), overlap (YELLOW) on one video
        #            and render per-threshold TP/FP/FN videos (TP=YELLOW, FP=RED, FN=GREEN)
        if RUN_STAGE10 and ran_stage9:
            out_overlay_path = DIR_STAGE10_OUT / f'{base}_gt_vs_model.mp4'
            overlay_gt_vs_model(
                orig_video_path=orig_path,
                pred_csv_path=csv_path,
                out_video_path=out_overlay_path,
                gt_norm_csv_path=None,                 # auto-find *_norm_offset*.csv (Stage 9 wrote & copied into CSV dir)
                thickness=BBOX_THICKNESS,
                gt_box_w=STAGE10_GT_BOX_W,
                gt_box_h=STAGE10_GT_BOX_H,
                only_firefly_rows=STAGE9_ONLY_FIREFLY_ROWS,
                max_frames=MAX_FRAMES,
                stage9_dir_hint=DIR_STAGE9_OUT,        # helps it find thr_* folders quickly
                render_threshold_overlays=True,        # also writes per-threshold TP/FP/FN videos
                thr_box_w=STAGE10_GT_BOX_W,
                thr_box_h=STAGE10_GT_BOX_H,
            )

        print(f'done {base}')

if __name__ == "__main__":
    main()
