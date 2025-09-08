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
import time
from audit_trail import AuditTrail

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
from stage8_5_blob_area_filter import stage8_5_prune_by_blob_area
from stage9_validate import stage9_validate_against_gt
from stage10_overlay_gt_vs_model import overlay_gt_vs_model  # includes per-threshold TP/FP/FN videos
from stage11_fn_analysis import stage11_fn_nearest_tp_analysis
from stage12_fp_analysis import stage12_fp_nearest_tp_analysis
from stage8_6_neighbor_hunt import stage8_6_run
from stage8_7_large_flash_bfs import stage8_7_expand_large_fireflies
from stage8_9_gt_gaussian_centroid import stage8_9_recenter_gt_gaussian_centroid
from stage13_audit_analysis import stage13_audit_trail_analysis

from stage8_sync import rebuild_fireflies_logits_from_main



# ──────────────────────────────────────────────────────────────
# Root & I/O locations
# ──────────────────────────────────────────────────────────────
ROOT = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/orc pipeline FFT model inference output/tremulans inference data')


# ──────────────────────────────────────────────────────────────
# Audit trail setup
# ──────────────────────────────────────────────────────────────
RUN_TAG = time.strftime("%Y%m%d_%H%M%S")
AUDIT_ROOT = (ROOT / 'audit' / RUN_TAG).resolve()
print(f"[orchestrator] AUDIT_ROOT: {AUDIT_ROOT}")
AUDIT = AuditTrail(AUDIT_ROOT, run_tag=RUN_TAG)



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

DIR_STAGE8_9_OUT       = ROOT / 'stage8.9 gt centroid crops'

# Stage 9 outputs
DIR_STAGE9_OUT   = ROOT / 'stage9 validation'

# Stage 10 outputs
DIR_STAGE10_OUT  = ROOT / 'stage10 overlay videos'

# Ensure output/working directories exist
for d in [DIR_CSV, DIR_OUT_BGS, DIR_OUT_ORIG, DIR_OUT_ORIG_10,
          DIR_STAGE8_CROPS, DIR_STAGE9_OUT, DIR_STAGE10_OUT, DIR_STAGE8_9_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Global knobs / flags
# ──────────────────────────────────────────────────────────────
MAX_FRAMES = None              # e.g., 5000 to truncate
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
RUN_STAGE8_5 = True
RUN_STAGE8_6 = True
RUN_STAGE8_9 = True
# After your existing toggles (RUN_STAGE8_5, RUN_STAGE8_6, RUN_STAGE8_7, …)
RUN_STAGE8_5_AFTER_8_7 = True


# THESE ARE THE VALIDATION STAGES, WILL ONLY RUN IF YOU HAVE GROUND TRUTH
RUN_STAGE9 = True
RUN_STAGE10 = True   # will only execute if RUN_STAGE9 is also True
RUN_STAGE11 = True
RUN_STAGE12 = True



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
AREA_THRESHOLD_PX = 6

# Stage 4 — CNN classify/filter
USE_CNN_FILTER             = True
CNN_MODEL_PATH             = Path('/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/frontalis, tremulans and forresti global models/resnet18_FFT_best_model.pt')  # ← SET THIS to your .pt file
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

# Stage 8.5,9 — area calculation brightness floor for the largest CC (strict '>').
MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION = 20


STAGE8_6_RUNS = 1          # try 2–3 to dig deeper
STAGE8_6_DEDUPE_PX = 4.0   # (optional) merge safety, defaults to 2.0 if omitted






# Stage 8.7 — large-flash fixer (BFS)
RUN_STAGE8_7                              = True
STAGE8_7_INTENSITY_THR                    = 70     # BFS_NEIGHBOR_PIXEL_INTENSITY_THRESHOLD
STAGE8_7_DEDUPE_PX                        = 10.0    # union-find distance among new squares
STAGE8_7_MIN_SQUARE_AREA_PX               = 75    # >100 to exceed 10x10
STAGE8_7_GAUSSIAN_SIGMA                   = STAGE8_GAUSSIAN_SIGMA  # reuse if you want








# Stage 8.9 — recenter GT via Gaussian centroid (pre-Stage 9)
STAGE8_9_CROP_W        = 10              # use 10×10 to match Stage 9
STAGE8_9_CROP_H        = 10
STAGE8_9_GAUSSIAN_SIGMA = STAGE8_GAUSSIAN_SIGMA  # reuse Stage-8 sigma; set >0 for Gaussian









# Stage 9 — validation vs ground truth
GT_CSV_PATH                = ROOT / 'ground truth' / 'gt.csv'  # GT (x,y,t), t is raw & will be normalized
GT_T_OFFSET                = 4000
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

# Stage 9 — GT dedup distance (in pixels) for merging duplicate GT points (Stage-7 style)
STAGE9_GT_DEDUPE_DIST_PX = 4.0





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
            AUDIT.record_params('01_detect')
            AUDIT.copy_snapshot('01_detect', csv_path)

        # Stage 2 — recenter via intensity centroid (drop dim crops)
        if RUN_STAGE2:
            recenter_boxes_with_centroid(
                orig_path=orig_path,
                csv_path=csv_path,
                max_frames=MAX_FRAMES,
                bright_max_threshold=BRIGHT_MAX_THRESHOLD,
                audit=AUDIT,
                audit_video_path=orig_path,

            )
            AUDIT.record_params('02_recenter', bright_max_threshold=BRIGHT_MAX_THRESHOLD)
            AUDIT.copy_snapshot('02_recenter', csv_path)

        # Stage 3 — area filter (in-place) + snapshot
        if RUN_STAGE3:
            snapshot_csv = csv_path.parent / f"{csv_path.stem}_area_snapshot.csv"
            filter_boxes_by_area(
                csv_path=csv_path,
                area_threshold_px=AREA_THRESHOLD_PX,
                snapshot_csv_path=snapshot_csv,
                audit=AUDIT,
            )
            AUDIT.record_params('03_area_filter', area_threshold_px=AREA_THRESHOLD_PX)
            AUDIT.copy_snapshot('03_area_filter', csv_path)

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
                audit=AUDIT,
            )
            AUDIT.record_params('04_cnn', model=str(CNN_MODEL_PATH), backbone=CNN_BACKBONE)
            AUDIT.copy_snapshot('04_cnn', csv_path)

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
                audit=AUDIT,
            )
            AUDIT.record_params('07_merge', dist_threshold_px=STAGE7_DIST_THRESHOLD_PX)
            AUDIT.copy_snapshot('07_merge', csv_path)
            rebuild_fireflies_logits_from_main(csv_path)


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
                audit=AUDIT,
            )
            AUDIT.record_params('08_gauss', patch_w=STAGE8_PATCH_W, patch_h=STAGE8_PATCH_H, sigma=STAGE8_GAUSSIAN_SIGMA)
            AUDIT.copy_snapshot('08_gauss', csv_path)
            rebuild_fireflies_logits_from_main(csv_path)
        
        # Stage 8.5 — prune firefly detections by blob area (> brightness floor), keep files in sync
        if RUN_STAGE8_5:
            stage8_5_prune_by_blob_area(
                orig_video_path=orig_path,
                csv_path=csv_path,
                area_threshold_px=AREA_THRESHOLD_PX,  # same scalar you use elsewhere
                min_pixel_brightness_to_be_considered_in_area_calculation=MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION,
                max_frames=MAX_FRAMES,
                verbose=True,
                audit=AUDIT,
            )
            AUDIT.record_params('08_5_blob_area', area_threshold_px=AREA_THRESHOLD_PX)
            AUDIT.copy_snapshot('08_5_blob_area', csv_path)
            rebuild_fireflies_logits_from_main(csv_path)

        if RUN_STAGE8_6:
            _added = stage8_6_run(
                orig_video_path=orig_path,
                main_csv_path=csv_path,
                num_runs=STAGE8_6_RUNS,
            )
            AUDIT.record_params('08_6_neighbor_hunt', runs=STAGE8_6_RUNS)
            AUDIT.copy_snapshot('08_6_neighbor_hunt', csv_path)
            rebuild_fireflies_logits_from_main(csv_path)

         # ── NEW: Stage 8.7 — grow large flashes & replace 10x10 shards
        if RUN_STAGE8_7:
            stage8_7_expand_large_fireflies(
                orig_video_path=orig_path,
                main_csv_path=csv_path,
                neighbor_intensity_threshold=STAGE8_7_INTENSITY_THR,
                dedupe_dist_px=STAGE8_7_DEDUPE_PX,
                min_square_area_px=STAGE8_7_MIN_SQUARE_AREA_PX,
                gaussian_sigma=STAGE8_7_GAUSSIAN_SIGMA,
                max_frames=MAX_FRAMES,
                verbose=True,
            )
            AUDIT.record_params('08_7_large_flash_bfs')
            AUDIT.copy_snapshot('08_7_large_flash_bfs', csv_path)
            rebuild_fireflies_logits_from_main(csv_path)

        # Re-run 8.5 AFTER 8.7 so replacements/center shifts are re-checked
        if RUN_STAGE8_5_AFTER_8_7:
            stage8_5_prune_by_blob_area(
                orig_video_path=orig_path,
                csv_path=csv_path,
                area_threshold_px=AREA_THRESHOLD_PX,
                min_pixel_brightness_to_be_considered_in_area_calculation=MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION,
                max_frames=MAX_FRAMES,
                verbose=True,
                audit=AUDIT,
            )
            AUDIT.record_params('08_5_blob_area', area_threshold_px=AREA_THRESHOLD_PX)
            AUDIT.copy_snapshot('08_5_blob_area', csv_path)
            rebuild_fireflies_logits_from_main(csv_path)




        # Stage 8.9 — GT recenter (produces x,y,t + debug crops), runs once per video
        if RUN_STAGE8_9:
            out89 = DIR_STAGE8_9_OUT / base
            stage8_9_recenter_gt_gaussian_centroid(
            orig_video_path=orig_path,
            gt_csv_path=GT_CSV_PATH,                 # overwritten in place to x,y,t
            crop_w=STAGE8_9_CROP_W,
            crop_h=STAGE8_9_CROP_H,
            gaussian_sigma=STAGE8_9_GAUSSIAN_SIGMA,
            gt_t_offset=GT_T_OFFSET,
            max_frames=MAX_FRAMES,
            out_crop_dir=out89,
            verbose=True,
            audit=AUDIT,
        )
            AUDIT.record_params('08_9_gt_centroid', crop_w=STAGE8_9_CROP_W,
                    crop_h=STAGE8_9_CROP_H, sigma=STAGE8_9_GAUSSIAN_SIGMA,
                    gt_t_offset=GT_T_OFFSET)









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
                # NEW: forward GT filters from orchestrator constants
                gt_area_threshold_px=AREA_THRESHOLD_PX,
                gt_bright_max_threshold=BRIGHT_MAX_THRESHOLD,
                # NEW: brightness floor for area calculation (largest CC uses '>' this value)
                min_pixel_brightness_to_be_considered_in_area_calculation=MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION,
                gt_dedupe_dist_threshold_px=STAGE9_GT_DEDUPE_DIST_PX,  # NEW
            )
            ran_stage9 = True
        
        # Stage 11 — FN analysis (per-threshold nearest-TP distances → CSV + full-frame images)
        if RUN_STAGE11 and ran_stage9:
            stage11_fn_nearest_tp_analysis(
                stage9_video_dir=DIR_STAGE9_OUT / base,
                orig_video_path=orig_path,          # NEW: needed to grab full frames
                box_w=STAGE10_GT_BOX_W,             # optional (keeps visuals consistent)
                box_h=STAGE10_GT_BOX_H,             # optional
                color=(0, 255, 255),                # optional: yellow boxes for FN & nearest TP
                thickness=BBOX_THICKNESS,           # optional
                verbose=True,
        )
            
            
        # Stage 12 — FP analysis (per-threshold nearest-TP distances → CSV + full-frame images)
        if RUN_STAGE12 and ran_stage9:
            stage12_fp_nearest_tp_analysis(
                stage9_video_dir=DIR_STAGE9_OUT / base,
                orig_video_path=orig_path,
                box_w=STAGE10_GT_BOX_W,   # keep consistent visuals
                box_h=STAGE10_GT_BOX_H,
                color=(255, 0, 255),      # magenta for FP↔TP pairs (same color for both boxes)
                thickness=BBOX_THICKNESS,
                verbose=True,
        )
        
        if ran_stage9:
            stage13_audit_trail_analysis(
                stage9_video_dir=DIR_STAGE9_OUT / base,
                audit_root=AUDIT_ROOT,
                pred_csv_path=csv_path,
                gt_csv_path=GT_CSV_PATH,
                gt_t_offset=GT_T_OFFSET,
                radius_px=4.0,      # tweak if you want stricter/looser matching
                verbose=True,
            )





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
