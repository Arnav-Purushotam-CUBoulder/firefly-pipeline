# Night Repair and Validation Stages

The night pipeline has more post-classifier repair logic than the day pipeline. This is because night videos often contain bright halos, adjacent flashes, detector shards, and duplicate candidates.

## Stage 8.5 - Blob Area Filter

File:

```text
stage8_5_blob_area_filter.py
```

Runs:

- Once after Stage 8.
- Again after Stage 8.7 if `RUN_STAGE8_5_AFTER_8_7=True`.

Behavior:

1. Reads the main CSV.
2. Keeps only firefly rows for pruning.
3. For each detection, extracts a centered crop.
4. Thresholds pixels with intensity greater than `MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION`.
5. Finds the largest connected component.
6. Removes detections whose largest component area is below `AREA_THRESHOLD_PX`.
7. Mirrors removals into the firefly logits CSV.

Use it to remove tiny classifier positives that do not contain a meaningful bright component.

## Stage 8.6 - Neighbor Hunt

File:

```text
stage8_6_neighbor_hunt.py
```

Purpose:

- Find additional detections near already-detected fireflies.
- Useful when a bright flash suppresses nearby candidates in the original detection pass.

Per run:

1. Reads the current post-Stage-8/8.5 main CSV.
2. Creates a blacked-out version of the video by masking existing firefly centers.
3. Runs Stage 1, 2, 3, 4, 7, 8, and 8.5 on the blacked-out video using orchestrator params.
4. Merges newly found firefly rows into the main CSV.
5. Dedupes by proximity.
6. Rebuilds the firefly logits CSV.

Outputs:

```text
ROOT/stage8.6/<video_stem>/run_XX/
```

Important knobs:

- `STAGE8_6_RUNS`
- `STAGE8_6_DEDUPE_PX`
- `STAGE8_6_STAGE1_IMPL`
- `STAGE8_6_DISABLE_CLAHE`

Failure mode:

- If Stage 8.6 adds many false positives, reduce runs, tighten Stage 1 thresholds, or increase dedupe.

## Stage 8.7 - Large Flash BFS Repair

File:

```text
stage8_7_large_flash_bfs.py
```

Purpose:

- Replace multiple small detections that belong to one large flash with a larger centered square.

Behavior:

1. For each firefly row, starts from the current center.
2. Grows a region over neighboring pixels with gray intensity above `STAGE8_7_INTENSITY_THR`.
3. Computes an intensity/Gaussian centroid over the grown region.
4. Builds the smallest square centered at that centroid that covers the region.
5. Keeps candidates larger than `STAGE8_7_MIN_SQUARE_AREA_PX`.
6. Dedupes candidate replacements by `STAGE8_7_DEDUPE_PX`.
7. Deletes contributing originals and inserts replacement rows.
8. Preserves class/logit/confidence columns where possible.
9. Updates the firefly logits CSV.

Outputs:

```text
ROOT/stage8.7/<video_stem>/replacements/
```

Tuning symptoms:

- Large flashes remain split: lower `STAGE8_7_INTENSITY_THR` or `STAGE8_7_MIN_SQUARE_AREA_PX`.
- Repairs grow into unrelated bright regions: raise `STAGE8_7_INTENSITY_THR`.
- Multiple repairs remain close together: increase `STAGE8_7_DEDUPE_PX`.

## Stage 8.9 - GT Recenter

File:

```text
stage8_9_gt_gaussian_centroid.py
```

Purpose:

- Convert annotation-style GT into validator-ready `x,y,t`.
- Apply the same kind of intensity/Gaussian recentering to GT as model predictions.

Input schema:

```text
x,y,w,h,frame
```

Output schema:

```text
x,y,t
```

The function overwrites `GT_CSV_PATH`.

Important behavior:

- If GT already has `x,y,t`, Stage 8.9 prints a skip message and leaves it alone.
- `GT_T_OFFSET` is used only to index into the video.
- The written `t` value is the raw frame number parsed from `frame`.
- Stage 9 later subtracts `GT_T_OFFSET`.

Important risk:

- Stage 8.9 treats input `x,y` as the crop center. If the original annotation CSV stores top-left box coordinates, verify or convert before Stage 8.9.

## Stage 9 - Validation

File:

```text
stage9_validate.py
```

Input:

```text
ROOT/csv files/<video_stem>.csv
ROOT/ground truth/gt.csv
```

Output:

```text
ROOT/stage9 validation/<video_stem>/
```

Behavior:

1. Reads GT as `x,y,t`.
2. Subtracts `GT_T_OFFSET`.
3. Writes normalized GT into Stage 9 output.
4. Filters normalized GT by brightness and area.
5. Dedupes GT by distance and crop weight.
6. Reads predictions from the main CSV.
7. Keeps only firefly rows when `STAGE9_ONLY_FIREFLY_ROWS=True`.
8. Requires `background_logit` and `firefly_logit` columns.
9. Greedily matches predictions to GT per frame for each threshold.
10. Writes `fps.csv`, `tps.csv`, and `fns.csv`.
11. Saves TP/FP/FN crops and confidence values.

Threshold output:

```text
thr_10.0/
  fps.csv
  tps.csv
  fns.csv
```

Important parameters:

- `DIST_THRESHOLDS_PX`
- `GT_T_OFFSET`
- `STAGE9_CROP_W`
- `STAGE9_CROP_H`
- `STAGE9_ONLY_FIREFLY_ROWS`
- `STAGE9_GT_DEDUPE_DIST_PX`
- `STAGE9_MODEL_PATH`

## Stage 10 - Overlays

File:

```text
stage10_overlay_gt_vs_model.py
```

Runs only if Stage 9 ran.

Output:

```text
ROOT/stage10 overlay videos/
```

Legend:

- GT: green.
- Model: red.
- Overlap: yellow.

It also renders threshold-specific TP/FP/FN overlays.

## Stage 11 - FN Analysis

File:

```text
stage11_fn_analysis.py
```

Behavior:

- Reads Stage 9 threshold folders.
- For each FN, finds nearest TP in the same frame.
- Writes `fn_nearest_tp.csv`.
- Renders full-frame debug images.
- Renders FN-vs-model context frames.

Use this to diagnose whether misses are true misses, time shifts, coordinate shifts, or match-radius issues.

## Stage 12 - FP Analysis

File:

```text
stage12_fp_analysis.py
```

Behavior:

- Reads Stage 9 threshold folders.
- For each FP, finds nearest TP in the same frame.
- Writes `fp_nearest_tp.csv`.
- Renders full-frame debug images.
- Renders FP-vs-GT context frames.

Use this to distinguish model noise from missing annotations.

## Stage 14 - Detection Summary

File:

```text
stage14_detection_summary.py
```

Behavior:

- Aggregates TP/FP/FN details into JSON.
- Includes crop-derived brightness/area values where available.
- Includes nearest-TP data from Stage 11/12 when present.

Use this for batch tuning, AI-assisted review, or handoff summaries.

