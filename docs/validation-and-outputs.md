# Validation and Output Schemas

## Ground Truth Formats

Validators accept two GT styles.

Annotator-style boxes:

```text
x,y,w,h,frame
```

Normalized points:

```text
x,y,t
```

The manual annotation GUI writes annotator-style boxes where `x,y` are top-left coordinates and `frame` is zero-based.

Validators normalize GT to point detections before matching. They can filter dim/small GT and dedupe nearby same-frame GT.

## Frame Offsets

`GT_T_OFFSET` is route-specific configuration:

- Day: `params.GT_T_OFFSET`
- Night: `pipeline_params.GT_T_OFFSET`

The gateway can infer or override this for subprocesses. If predictions and GT look consistently shifted in time, inspect `GT_T_OFFSET` and the GT CSV's frame convention before tuning model thresholds.

## Matching

Day validation:

```text
Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/stage5_validate.py
```

Night validation:

```text
Pipelines/night_time_pipeline/stage9_validate.py
```

Both use distance thresholds from `DIST_THRESHOLDS_PX`. Current defaults validate at 10 px.

The validation output is organized by threshold:

```text
thr_10.0/
  fps.csv
  tps.csv
  fns.csv
  crops/
```

Exact names may vary by stage and pipeline, but the important records are TP, FP, and FN CSVs plus crops/debug images.

## Day Prediction Schemas

Stage 2 YOLO detections:

```text
x,y,w,h,frame_range,video_name
```

Stage 3 patch classifier positives:

```text
frame_idx,video_name,x,y,w,h,conf,det_id
```

Stage 3.1 selected trajectories:

```text
frame_idx,video_name,x,y,w,h,conf,det_id,traj_id,traj_size,traj_motion_xy,traj_intensity_range,traj_is_selected
```

Stage 3.2 logits:

```text
x,y,t,firefly_logit,background_logit
```

Stage 3.2 reconstruction export:

```text
x,y,t
```

Use Stage 3.2 for final centroid-style point outputs.

## Night Prediction Schemas

Early night detections use box CSVs with frame coordinates.

After Stage 8, the main CSV is rewritten with center semantics:

```text
frame,x,y,w,h,...,xy_semantics
```

Expected final semantics:

- `x,y` are detection centers.
- `w,h` are fixed 10x10 boxes.
- `xy_semantics` is `center`.

Night logits CSV:

```text
x,y,t,background_logit,firefly_logit
```

## Overlays

Day overlay:

```text
stage6 overlay videos/
```

Night overlay:

```text
stage10 overlay videos/
```

Legend convention:

- GT: green.
- Model: red.
- Overlap: yellow.

Threshold-specific TP/FP/FN videos may be emitted under validation/overlay outputs.

## Analysis Stages

Day:

- Stage 7: FN analysis.
- Stage 8: FP analysis.
- Stage 9: detection summary.

Night:

- Stage 11: FN analysis.
- Stage 12: FP analysis.
- Stage 14: detection summary.

These stages are useful for model/parameter tuning. They are not the minimum required for routine inference.

## Coordinate Pitfalls

Do not mix these without conversion:

- Annotator top-left boxes: `x,y,w,h,frame`.
- Day Stage 3 top-left patch detections.
- Day Stage 3.2 center-style `x,y,t`.
- Night final Stage 8 center-style detections.

When in doubt, open the relevant stage code or render an overlay before comparing CSVs numerically.

