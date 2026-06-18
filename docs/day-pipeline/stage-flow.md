# Day Pipeline Stage Flow

Main runner:

```text
Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/orchestrator.py
```

Main config:

```text
Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/params.py
```

## Overall Flow

```text
source video
  -> Stage 1 long exposure PNGs
  -> Stage 2 YOLO boxes on long exposures
  -> Stage 3 per-frame patch classifier positives
  -> Stage 3.1 trajectory/intensity selection
  -> Stage 3.2 centroid/logit export
  -> Stage 4 rendered prediction video
  -> Stage 5-9 validation and analysis, if GT exists and toggles are enabled
```

The orchestrator always runs Stage 1, Stage 2, Stage 3, and Stage 4. Stage 3.1 and Stage 3.2 are controlled by toggles. Stage 5-9 run only when their toggles are enabled and a matching GT CSV is found.

## Stage 0 - Cleanup

File:

```text
stage0_cleanup.py
```

Triggered when:

```python
RUN_PRE_RUN_CLEANUP = True
```

Behavior:

- Ensures `ROOT` exists.
- Deletes every directory under `ROOT` except:
  - `original videos`
  - `ground truth`
- Leaves files directly under `ROOT` alone.

Risk:

- This is safe only when `ROOT` is a disposable inference-output folder.
- Never point `ROOT` at the raw-video root, model zoo, integrated dataset root, or YOLO dataset root.

## Stage 1 - Long Exposure

File:

```text
stage1_long_exposure.py
```

Input:

```text
ROOT/original videos/<video>.mp4
```

Output:

```text
ROOT/stage1_long_exposure/<video_stem>/<interval>_<video_stem>_<mode>_<start>-<end>.png
```

Purpose:

- Converts a sequence of frames into one or more long-exposure PNGs.
- Makes small moving flashes visible as streaks or bright regions for YOLO.

Important parameters:

- `LONG_EXPOSURE_MODE`
  - `lighten` takes the per-pixel maximum over the frame interval.
  - `average` averages frames, optionally with gamma.
  - `trails` uses background subtraction/trail accumulation.
- `INTERVAL_FRAMES`
  - Default is `100`.
  - `None` or `<= 0` means one image across the processed video span.
- `MAX_FRAMES`
  - Caps processed frames for iteration.

Downstream dependency:

- Stage 2 parses the frame range from the filename. Do not rename Stage 1 PNGs casually.

## Stage 2 - YOLO Detection

File:

```text
stage2_yolo_detect.py
```

Input:

```text
ROOT/stage1_long_exposure/<video_stem>/*.png
```

Output:

```text
ROOT/stage2_yolo_detections/<video_stem>/<video_stem>.csv
ROOT/stage2_yolo_detections/<video_stem>/annotated/*.png
```

CSV schema:

```text
x,y,w,h,frame_range,video_name
```

Purpose:

- Runs Ultralytics YOLO on long-exposure images.
- Produces candidate boxes that are broad in time but localized in image space.

Important parameters:

- `YOLO_MODEL_WEIGHTS`
- `YOLO_CONF_THRES`
- `YOLO_IOU_THRES`
- `YOLO_IMG_SIZE`
- `YOLO_DEVICE`
- `YOLO_BATCH_SIZE`
- `YOLO_HALF_ON_CUDA`

Implementation notes:

- The stage resolves `YOLO_DEVICE=auto` to CUDA, then MPS, then CPU.
- If the weights path contains an apostrophe, the code copies weights into `~/.cache/firefly_pipeline/ultralytics_weights` before loading. This avoids an Ultralytics path parsing issue.

## Stage 3 - Patch Classifier

File:

```text
stage3_patch_classifier.py
```

Input:

```text
ROOT/stage2_yolo_detections/<video_stem>/<video_stem>.csv
ROOT/original videos/<video>.mp4
```

Output:

```text
ROOT/stage3_patch_classifier/<video_stem>/<video_stem>_patches.csv
ROOT/stage3_patch_classifier/<video_stem>/crops/positives/*.png
```

CSV schema:

```text
frame_idx,video_name,x,y,w,h,conf,det_id
```

Purpose:

- For each YOLO candidate box and each frame in that candidate's frame range:
  - opens the source video frame,
  - looks inside the YOLO ROI,
  - finds the brightest pixel,
  - centers a 10x10 patch on that pixel,
  - classifies the patch with a ResNet18 binary classifier,
  - writes only positive crops.

Important parameters:

- `PATCH_MODEL_PATH`
- `PATCH_SIZE_PX`
- `STAGE3_INPUT_SIZE`
- `STAGE3_POSITIVE_THRESHOLD`
- `STAGE3_DEVICE`
- `STAGE3_BATCH_SIZE_GPU`
- `STAGE3_BATCH_SIZE_CPU`
- `STAGE3_USE_AMP`
- `IMAGENET_NORMALIZE`

Coordinate semantics:

- `x,y` are top-left patch coordinates.
- `w,h` are patch dimensions, usually 10x10.
- These are not final centroid coordinates.

## Stage 3.1 - Trajectory and Intensity Selection

File:

```text
stage3_1_trajectory_intensity_selector.py
```

Toggle:

```python
RUN_STAGE3_1_TRAJECTORY_INTENSITY_SELECTOR = True
```

Input:

```text
ROOT/stage3_patch_classifier/<video_stem>/<video_stem>_patches.csv
ROOT/stage3_patch_classifier/<video_stem>/crops/positives/*.png
```

Outputs:

```text
ROOT/stage3_patch_classifier/<video_stem>/<video_stem>_patches_motion.csv
ROOT/stage3_patch_classifier/<video_stem>/<video_stem>_patches_motion_all.csv
ROOT/stage3_patch_classifier/<video_stem>/stage3_1_trajectory_crops/
ROOT/stage3_patch_classifier/<video_stem>/stage3_1_trajectory_intensity_curves.svg
ROOT/stage3_patch_classifier/<video_stem>/stage3_1_trajectory_intensity_curves_highvar.svg
ROOT/stage3_patch_classifier/<video_stem>/stage3_1_highvar_trajectories.mp4
```

Purpose:

- Links Stage 3 detections into trajectories in `(x,y,t)` space.
- Computes an intensity curve for each trajectory using the saved Stage 3 crops.
- Selects trajectories whose intensity curve looks like a firefly flash.

Added columns:

```text
traj_id,traj_size,traj_motion_xy,traj_intensity_range,traj_is_selected
```

Selection knobs:

- `STAGE3_1_LINK_RADIUS_PX`
- `STAGE3_1_MAX_FRAME_GAP`
- `STAGE3_1_MIN_TRACK_POINTS`
- `STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE`
- `STAGE3_1_HIGHVAR_REQUIRE_HILL_SHAPE`
- `STAGE3_1_HILL_MIN_UP_STEPS`
- `STAGE3_1_HILL_MIN_DOWN_STEPS`
- `STAGE3_1_HILL_MIN_MONOTONIC_FRAC`

Practical tuning:

- If moving fireflies are split into many tiny tracks, increase `STAGE3_1_LINK_RADIUS_PX` or `STAGE3_1_MAX_FRAME_GAP`.
- If real flashes are rejected, lower the high-variation or hill-shape requirements.
- If noisy constant bright points remain, tighten hill-shape or range requirements.

## Stage 3.2 - Gaussian Centroids and Logits

File:

```text
stage3_2_gaussian_centroids_and_logits.py
```

Toggle:

```python
RUN_STAGE3_2 = True
```

Preferred input:

```text
ROOT/stage3_patch_classifier/<video_stem>/<video_stem>_patches_motion.csv
```

Fallback input:

```text
ROOT/stage3_patch_classifier/<video_stem>/<video_stem>_patches_motion_all.csv
```

with `traj_is_selected == 1`.

Outputs:

```text
ROOT/stage3_patch_classifier/<video_stem>/stage3_2/<video_stem>_stage3_2_firefly_background_logits.csv
ROOT/stage3_2 xyt for 3d reconstruction/<video_stem>.csv
```

Logits schema:

```text
x,y,t,firefly_logit,background_logit
```

3D reconstruction schema:

```text
x,y,t
```

Purpose:

- Converts selected top-left patch detections into refined full-frame centroid coordinates.
- Computes intensity or Gaussian-weighted centroids inside each saved patch.
- Converts patch confidence to firefly/background logits.

Important parameters:

- `STAGE3_2_GAUSSIAN_SIGMA`
- `STAGE3_2_SAVE_ANNOTATED_CROPS`
- `STAGE3_2_MARK_CENTROID_RED_PIXEL`
- `STAGE3_2_XYT_EXPORT_DIR`

Use this stage's `x,y,t` as the final day pipeline point output.

## Stage 4 - Render

File:

```text
stage4_render.py
```

Input preference:

1. `*_patches_motion_all.csv`, when `STAGE4_DRAW_STAGE3_1_REJECTED=True`
2. `*_patches_motion.csv`, when present
3. `*_patches.csv`, fallback

Output:

```text
ROOT/stage4_rendering/<video_stem>/<video_stem>_patches.mp4
```

Purpose:

- Renders prediction boxes on the source video.
- Useful for fast visual inspection before opening validation outputs.

Important parameters:

- `RENDER_CODEC`
- `RENDER_FPS_HINT`
- `RENDER_ENFORCE_SOURCE_RESOLUTION`
- `OVERLAY_BOX_THICKNESS`
- `STAGE4_DRAW_STAGE3_1_REJECTED`

## Stage 5-9 - Test Suite

The orchestrator calls the test suite after Stage 4 if any of these are enabled:

```python
RUN_STAGE5_VALIDATE
RUN_STAGE6_OVERLAY
RUN_STAGE7_FN_ANALYSIS
RUN_STAGE8_FP_ANALYSIS
RUN_STAGE9_DETECTION_SUMMARY
```

The test suite runs only when the orchestrator can find GT for the current video. See [validation-and-analysis.md](validation-and-analysis.md).

