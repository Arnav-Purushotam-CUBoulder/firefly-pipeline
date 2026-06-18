# Night Pipeline Stage Flow

Main runner:

```text
Pipelines/night_time_pipeline/orchestrator.py
```

Main config:

```text
Pipelines/night_time_pipeline/pipeline_params.py
```

## Overall Flow

```text
source video
  -> Stage 1 bright blob detection
  -> Stage 2 intensity recentering
  -> Stage 3 area filtering
  -> Stage 4 CNN firefly/background classification
  -> Stage 5 dynamic-box render
  -> Stage 6 fixed 10x10 render
  -> Stage 7 duplicate merge
  -> Stage 8 Gaussian/intensity centroid and logits sync
  -> Stage 8.5 blob-area pruning
  -> Stage 8.6 neighbor hunt
  -> Stage 8.7 large-flash repair
  -> Stage 8.5 after repair
  -> Stage 8.9 GT recenter
  -> Stage 9 validation
  -> Stage 11/12 FN/FP analysis
  -> Stage 14 summary
  -> Stage 10 overlay videos
```

The orchestrator runs stages in the actual order above. Stage 10 overlays run after Stage 14 in the code, even though the stage number is lower, because overlays need Stage 9 validation outputs.

## Stage 0 - Cleanup

File:

```text
stage0_cleanup.py
```

Triggered at import time when:

```python
RUN_PRE_RUN_CLEANUP = True
```

Behavior:

1. Deletes directories under `ROOT` except configured keep dirs.
2. Keeps, by default:
   - `ground truth`
   - `original videos`
3. Removes old `ground truth/gt.csv`.
4. Copies a CSV located directly under `ROOT` into `ground truth/gt.csv` if one exists.

Risk:

- This can overwrite/remove the active GT file.
- Inspect `ROOT` before direct night runs.
- Gateway runs can disable cleanup where needed to protect concurrent outputs.

## Stage 1 - Bright Blob Detection

Files:

```text
stage1_detect_cucim.py
stage1_detect_cc_cuda.py
stage1_detect_cc_cpu.py
stage1_detect.py
```

Input:

```text
ROOT/original videos/<video>.mp4
```

Output:

```text
ROOT/csv files/<video_stem>.csv
```

Initial schema:

```text
frame,x,y,w,h
```

Purpose:

- Finds bright candidate regions in each frame.
- Writes dynamic boxes before classifier filtering.

Variant selection:

```python
STAGE1_VARIANT = "cucim"
```

Supported variants:

- `cucim` - default GPU cuCIM blob detector.
- `cc_cuda` - CUDA connected-components detector.
- `cc_cpu` - CPU connected-components detector.
- `blob` - OpenCV SimpleBlobDetector path.

## Stage 2 - Intensity Recenter

File:

```text
stage2_recenter.py
```

Input/output:

```text
ROOT/csv files/<video_stem>.csv
```

Purpose:

- Opens each candidate crop in the original video.
- Recenters boxes based on intensity centroid.
- Drops dim crops whose maximum brightness is below `BRIGHT_MAX_THRESHOLD`.

Important parameter:

```python
BRIGHT_MAX_THRESHOLD = 50
```

## Stage 3 - Area Filter

File:

```text
stage3_area_filter.py
```

Input/output:

```text
ROOT/csv files/<video_stem>.csv
```

Snapshot:

```text
ROOT/csv files/<video_stem>_area_snapshot.csv
```

Purpose:

- Removes rows with bright area below `AREA_THRESHOLD_PX`.
- Saves a snapshot around the area-filter point in the pipeline.

Important parameter:

```python
AREA_THRESHOLD_PX = 6
```

## Stage 4 - CNN Classify and Filter

File:

```text
stage4_cnn_filter.py
```

Input/output:

```text
ROOT/csv files/<video_stem>.csv
```

Purpose:

- Extracts 10x10 patches.
- Runs a ResNet18 binary classifier.
- Adds class/logit/confidence columns.
- Optionally drops background rows.

Important parameters:

- `USE_CNN_FILTER`
- `CNN_MODEL_PATH`
- `CNN_BACKBONE`
- `CNN_CLASS_TO_KEEP`
- `CNN_PATCH_W`
- `CNN_PATCH_H`
- `FIREFLY_CONF_THRESH`
- `DROP_BACKGROUND_ROWS`
- `FAIL_IF_WEIGHTS_MISSING`

Default behavior keeps background rows:

```python
DROP_BACKGROUND_ROWS = False
```

Validation and rendering usually use only class `firefly` where configured.

## Stage 5 - Dynamic Render

File:

```text
stage5_render.py
```

Outputs:

```text
ROOT/BS initial output annotated videos/<video_stem>_bs_annotated.mp4
ROOT/original initial output annotated videos/<video_stem>_orig_annotated.mp4
```

Purpose:

- Draws current dynamic boxes on the original video and optional BS video.
- Firefly and background rows can be drawn in different colors.

## Stage 6 - Fixed 10px Render

File:

```text
stage6_10px_renderer.py
```

Output:

```text
ROOT/original 10px overlay annotated videos/<video_stem>_orig_10px.mp4
```

Purpose:

- Draws fixed 10x10 boxes centered on each detection.
- Useful for comparing against patch classifier geometry and GT crops.

## Stage 7 - Merge Duplicates

File:

```text
stage7_merge.py
```

Input/output:

```text
ROOT/csv files/<video_stem>.csv
```

Purpose:

- Per frame, builds connected components of boxes whose centroids are within `STAGE7_DIST_THRESHOLD_PX`.
- Keeps the heaviest row in each component using RGB-sum weight from the original video.
- Preserves extra columns such as class/logits/confidence.
- Rebuilds `<video_stem>_fireflies_logits.csv` after the merge.

Important parameter:

```python
STAGE7_DIST_THRESHOLD_PX = 20.0
```

## Stage 8 - Gaussian/Intensity Centroid

File:

```text
stage8_gaussian_centroid.py
```

Input/output:

```text
ROOT/csv files/<video_stem>.csv
```

Auxiliary output:

```text
ROOT/stage8 crops/<video_stem>/
ROOT/csv files/<video_stem>_fireflies_logits.csv
```

Purpose:

- Refines each detection center inside a small crop.
- Rewrites the main CSV to center semantics.
- Sets fixed 10x10 `w,h`.
- Adds/sets `xy_semantics=center`.
- Rebuilds firefly logits after completion.

Important parameters:

- `STAGE8_PATCH_W`
- `STAGE8_PATCH_H`
- `STAGE8_GAUSSIAN_SIGMA`

## Stage 8.5 - Blob Area Prune

File:

```text
stage8_5_blob_area_filter.py
```

Purpose:

- Re-checks firefly rows after center refinement.
- Measures largest connected component of pixels above a brightness floor.
- Removes rows whose bright component area is below threshold.
- Keeps the main CSV and firefly logits CSV in sync.

Important parameters:

- `AREA_THRESHOLD_PX`
- `MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION`

## Stage 8.6 - Neighbor Hunt

File:

```text
stage8_6_neighbor_hunt.py
```

Output root:

```text
ROOT/stage8.6/<video_stem>/run_XX/
```

Purpose:

- Masks current detections with black 10x10 squares.
- Runs detection/classification again on the blacked-out video.
- Finds neighboring detections that were hidden by stronger detections.
- Merges new detections back into the main CSV with proximity dedupe.

Important parameters:

- `STAGE8_6_RUNS`
- `STAGE8_6_DEDUPE_PX`
- `STAGE8_6_STAGE1_IMPL`
- `STAGE8_6_DISABLE_CLAHE`

## Stage 8.7 - Large Flash Repair

File:

```text
stage8_7_large_flash_bfs.py
```

Output root:

```text
ROOT/stage8.7/<video_stem>/replacements/
```

Purpose:

- Grows bright regions from each firefly center using BFS.
- Builds a square that covers the grown region.
- Replaces fragmented 10x10 shards with a larger centered square when the region is large enough.
- Updates main CSV and logits.

Important parameters:

- `STAGE8_7_INTENSITY_THR`
- `STAGE8_7_DEDUPE_PX`
- `STAGE8_7_MIN_SQUARE_AREA_PX`
- `STAGE8_7_GAUSSIAN_SIGMA`

## Stage 8.5 After 8.7

The orchestrator runs Stage 8.5 again after Stage 8.7 when:

```python
RUN_STAGE8_5_AFTER_8_7 = True
```

This re-validates repaired/recentered boxes against the same bright-area filter.

## Stage 8.9 - GT Recenter

File:

```text
stage8_9_gt_gaussian_centroid.py
```

Purpose:

- Converts `x,y,w,h,frame` GT into `x,y,t`.
- Uses `GT_T_OFFSET` only to locate the corresponding video frame.
- Writes raw frame index to `t`.
- Overwrites the input GT CSV.
- Skips if the GT CSV is already `x,y,t`.

Output:

```text
ROOT/ground truth/gt.csv
ROOT/stage8.9 gt centroid crops/<video_stem>/
```

Important risk:

- Stage 8.9 treats the input `x,y` as the point/crop center. Verify the annotation CSV semantics before relying on this conversion.

## Stage 9-14

Validation and analysis are covered in [repair-and-validation.md](repair-and-validation.md).

