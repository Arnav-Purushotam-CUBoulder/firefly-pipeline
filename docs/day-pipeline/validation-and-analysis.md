# Day Validation and Analysis

Validation is implemented inside the day pipeline folder but mirrors the night validator closely.

## When Validation Runs

The orchestrator checks whether any test-suite toggle is enabled:

```python
RUN_STAGE5_VALIDATE
RUN_STAGE6_OVERLAY
RUN_STAGE7_FN_ANALYSIS
RUN_STAGE8_FP_ANALYSIS
RUN_STAGE9_DETECTION_SUMMARY
```

If none are enabled, the test suite is skipped.

If at least one is enabled, the orchestrator looks for GT for the video. If no matching GT CSV is found, it prints a skip message and continues.

## Prediction Preparation

Before Stage 5, the orchestrator builds:

```text
ROOT/stage5 validation/<video_stem>/<video_stem>.csv
```

with schema:

```text
x,y,w,h,t,class,xy_semantics,firefly_logit,background_logit
```

Preferred source:

- Stage 3.2 logits, because those coordinates are refined centroids.

Fallback source:

- Stage 3 patch detections, converted from top-left boxes to centers, with logits synthesized from `conf`.

This adaptation is why Stage 5 can use the same style of validation code as the night pipeline.

## Stage 5 - Validate Against GT

File:

```text
stage5_validate.py
```

Output root:

```text
ROOT/stage5 validation/<video_stem>
```

Main behavior:

1. Reads GT and normalizes it to `x,y,t`.
2. Subtracts `GT_T_OFFSET` from GT time values.
3. Optionally filters GT points by brightness and connected-component area.
4. Deduplicates nearby GT points per frame.
5. Reads predictions with logits and confidence.
6. Greedily matches predictions to GT per frame at each distance threshold.
7. Writes TP, FP, and FN CSVs and crops.

Threshold output:

```text
thr_10.0/
  fps.csv
  tps.csv
  fns.csv
  crops/
```

TP/FP/FN CSV schema:

```text
x,y,t,filepath,confidence
```

Important parameters:

- `DIST_THRESHOLDS_PX`
- `GT_T_OFFSET`
- `STAGE5_CROP_W`
- `STAGE5_CROP_H`
- `STAGE5_ONLY_FIREFLY_ROWS`
- `STAGE5_GT_AREA_THRESHOLD_PX`
- `STAGE5_GT_BRIGHT_MAX_THRESHOLD`
- `STAGE5_MIN_PIXEL_BRIGHTNESS_FOR_AREA_CALC`
- `STAGE5_GT_DEDUPE_DIST_PX`

## GT Filtering

The validator filters GT before matching. It keeps GT points whose centered crop has enough brightness/area according to the configured thresholds.

This matters because raw annotation CSVs can include boxes that are too dim or too small to evaluate consistently.

If a takeover user sees "missing" GT rows, inspect the normalized GT CSV and filtering thresholds before assuming the validator is broken.

## GT Deduplication

Within each frame, nearby GT points are grouped by distance. The validator keeps the heaviest crop in each group.

Day default:

```python
STAGE5_GT_DEDUPE_DIST_PX = 2.0
```

This prevents duplicate labels from inflating false negatives.

## FN Confidence

For false negatives, the validator can run the patch classifier on the FN crop and write a confidence value. It uses:

- `STAGE5_MODEL_PATH`
- `STAGE5_BACKBONE`
- `STAGE5_IMAGENET_NORM`

If weights are missing, FN confidence becomes blank/NaN, but matching can still run.

## Stage 6 - Overlay

File:

```text
stage6_overlay_gt_vs_model.py
```

Output root:

```text
ROOT/stage6 overlay videos
```

Legend:

- GT: green.
- Model: red.
- Pixel overlap: yellow.

It also renders per-threshold TP/FP/FN videos when threshold overlays are enabled.

## Stage 7 - FN Analysis

File:

```text
stage7_fn_analysis.py
```

Output root:

```text
ROOT/stage7 fn analysis/<video_stem>
```

Behavior:

- For each FN, finds the nearest TP in the same frame.
- Writes nearest-distance CSVs.
- Renders full-frame debug images.
- Renders FN-vs-prediction context frames with the FN and all model predictions visible.

Use this when real detections appear close to GT but are outside the match radius, or when `GT_T_OFFSET`/coordinate semantics are suspect.

## Stage 8 - FP Analysis

File:

```text
stage8_fp_analysis.py
```

Output root:

```text
ROOT/stage8 fp analysis/<video_stem>
```

Behavior:

- For each FP, finds the nearest TP in the same frame.
- Writes nearest-distance CSVs.
- Renders full-frame debug images.
- Renders FP-vs-GT context frames.

Use this when the pipeline is over-detecting or when a predicted point may actually be an unannotated firefly.

## Stage 9 - Detection Summary

File:

```text
stage9_detection_summary.py
```

Output root:

```text
ROOT/stage9 detection summary/<video_stem>
```

Behavior:

- Aggregates TP/FP/FN metadata into JSON.
- Pulls crop brightness/area values from crop filenames where available.
- Optionally includes nearest-TP details from Stage 7 and Stage 8.

This is useful for parameter sweeps and AI-assisted tuning.

## Common Validation Mistakes

### Time shift

Symptom:

- Predictions and GT look correct visually, but validation shows many FNs/FPs.

Check:

- `GT_T_OFFSET`
- Whether GT frame numbers are raw video frame numbers or clipped-frame numbers.
- Whether gateway inferred a frame offset.

### Coordinate mismatch

Symptom:

- Detections appear consistently offset by about half a box.

Check:

- Whether the CSV being evaluated uses top-left or center semantics.
- Whether `xy_semantics=center` is present where expected.

### GT filtered too aggressively

Symptom:

- Fewer normalized GT rows than expected.

Check:

- Normalized GT CSV in Stage 5 output.
- GT brightness/area thresholds.
- Raw annotation boxes on rendered video.

