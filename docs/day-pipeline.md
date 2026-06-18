# Day-Time Pipeline

Path:

```text
Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)
```

Main config:

```text
Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/params.py
```

Main runner:

```text
Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/orchestrator.py
```

For detailed takeover docs, start with:

```text
docs/day-pipeline/README.md
```

Detailed pages:

- [day-pipeline/stage-flow.md](day-pipeline/stage-flow.md)
- [day-pipeline/configuration.md](day-pipeline/configuration.md)
- [day-pipeline/data-contracts.md](day-pipeline/data-contracts.md)
- [day-pipeline/validation-and-analysis.md](day-pipeline/validation-and-analysis.md)
- [day-pipeline/operations-and-debugging.md](day-pipeline/operations-and-debugging.md)

## Purpose

The day pipeline handles videos where the background is bright enough that simple bright-blob detection is not enough. It compresses time into long-exposure images, detects candidate regions with YOLO, expands those regions back to individual frames, filters frame patches with a binary patch classifier, then optionally performs trajectory/intensity selection and centroid refinement.

## Input Layout

`params.ROOT` must contain:

```text
original videos/
```

Optional validation input:

```text
ground truth/
```

Supported GT names under `ground truth/`:

- `gt_<video_stem>.csv`
- `<video_stem>.csv`
- `<video_stem>_gt.csv`
- `gt.csv` for single-video fallback

Default root:

```text
/mnt/Samsung_SSD_2TB/temp to delete/day time pipeline inference output data
```

## Models

Default YOLO model:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/models/global models/20260414/best_firefly_yolo.pt
```

Default patch classifier:

```text
/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/tmp_day_night_combo_train_and_infer__20260328__004250/models/day/global_all_species.pt
```

When running through the gateway, both can be overridden with `--day-yolo-model` and `--day-patch-model`.

## Run

From repo root:

```bash
python3 "Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/orchestrator.py"
```

For safer single-video use, prefer the gateway:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/day_video.mp4" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --route-override day \
  --force-tests \
  --max-concurrent 1
```

## Stages

| Stage | File | What it does | Main output |
| --- | --- | --- | --- |
| 0 | `stage0_cleanup.py` | Deletes generated outputs under `ROOT`, preserving input/GT folders. | Clean working root |
| 1 | `stage1_long_exposure.py` | Builds long-exposure PNGs from video frame intervals. | `stage1_long_exposure/<stem>/*.png` |
| 2 | `stage2_yolo_detect.py` | Runs YOLO on long-exposure images and writes aggregated boxes. | `stage2_yolo_detections/<stem>/<stem>.csv` |
| 3 | `stage3_patch_classifier.py` | Expands boxes over frames, recenters 10x10 crops on brightest pixels, classifies patches. | `stage3_patch_classifier/<stem>/<stem>_patches.csv` |
| 3.1 | `stage3_1_trajectory_intensity_selector.py` | Links detections into trajectories and keeps flash-like intensity curves. | `*_patches_motion.csv`, `*_patches_motion_all.csv` |
| 3.2 | `stage3_2_gaussian_centroids_and_logits.py` | Refines selected detections to centroid/logit outputs. | `stage3_2/*_logits.csv`, `stage3_2 xyt for 3d reconstruction/<stem>.csv` |
| 4 | `stage4_render.py` | Renders predicted boxes on source video. | `stage4_rendering/<stem>/<stem>_patches.mp4` |
| 5 | `stage5_validate.py` | Matches model predictions to GT. | `stage5 validation/<stem>/thr_*` |
| 6 | `stage6_overlay_gt_vs_model.py` | Renders GT/model/overlap videos. | `stage6 overlay videos/` |
| 7 | `stage7_fn_analysis.py` | Optional false-negative analysis. | `stage7 fn analysis/` |
| 8 | `stage8_fp_analysis.py` | Optional false-positive analysis. | `stage8 fp analysis/` |
| 9 | `stage9_detection_summary.py` | Optional JSON detection summary. | `stage9 detection summary/` |

## Important Parameters

Common edit points in `params.py`:

- `ROOT` - working folder.
- `RUN_PRE_RUN_CLEANUP` - defaults to `True`.
- `MAX_FRAMES` - frame cap for fast iteration.
- `LONG_EXPOSURE_MODE` - `lighten`, `average`, or `trails`.
- `INTERVAL_FRAMES` - default `100`.
- `YOLO_MODEL_WEIGHTS`, `YOLO_CONF_THRES`, `YOLO_IOU_THRES`, `YOLO_DEVICE`.
- `PATCH_MODEL_PATH`, `STAGE3_POSITIVE_THRESHOLD`, `STAGE3_DEVICE`.
- `RUN_STAGE3_1_TRAJECTORY_INTENSITY_SELECTOR`.
- `RUN_STAGE3_2`.
- `RUN_STAGE5_VALIDATE`, `RUN_STAGE6_OVERLAY`, `RUN_STAGE7_FN_ANALYSIS`, `RUN_STAGE8_FP_ANALYSIS`, `RUN_STAGE9_DETECTION_SUMMARY`.
- `GT_CSV_DIR`, `GT_T_OFFSET`, `DIST_THRESHOLDS_PX`.

## Output Semantics

Stage 2 boxes are long-exposure image detections with `frame_range`.

Stage 3 boxes are per-frame patch detections. `x,y` are top-left crop coordinates.

Stage 3.1 keeps only selected flash-like trajectories in `*_patches_motion.csv`. The `*_patches_motion_all.csv` file includes rejected trajectories and explains selection with columns such as `traj_id`, `traj_size`, `traj_intensity_range`, and `traj_is_selected`.

Stage 3.2 writes two important artifacts:

- Full logits CSV with `x,y,t,firefly_logit,background_logit`.
- Reconstruction CSV with exactly `x,y,t`.

Use Stage 3.2 `x,y,t` for downstream 3D reconstruction, not raw Stage 3 top-left boxes.

## Operational Risks

- Stage 0 cleanup deletes generated outputs under `ROOT`.
- Do not set `ROOT` to the raw data root or integrated model/data root.
- Stage 5 validation can accept annotator-style GT or normalized `x,y,t`, but `GT_T_OFFSET` must be correct.
- Stage 7, 8, and 9 analyses are disabled by default in `params.py`.
- Ultralytics may have trouble with apostrophes in model paths. The Stage 2 code copies weights to an apostrophe-safe cache path when needed.
