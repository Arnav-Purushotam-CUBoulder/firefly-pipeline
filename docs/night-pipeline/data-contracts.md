# Night Pipeline Data Contracts

This page documents night-pipeline files and schemas. The key thing to remember is that the main CSV changes semantics over time.

## Input Video Contract

Supported extensions:

```text
.mp4, .avi, .mov, .mkv
```

Direct runs read:

```text
ROOT/original videos
```

Gateway runs override the orchestrator video iterator and process the source path directly.

## Main CSV Contract

Main CSV path:

```text
ROOT/csv files/<video_stem>.csv
```

The same file is rewritten in place by many stages.

Early schema:

```text
frame,x,y,w,h
```

After CNN classification, extra columns include class/logit/confidence values.

After Stage 8, final expected schema includes:

```text
frame,x,y,w,h,class,background_logit,firefly_logit,firefly_confidence,xy_semantics
```

The exact column set can include additional preserved fields. Do not write consumers that require only the listed columns unless you intentionally filter them.

## Coordinate Semantics Over Time

| Point in pipeline | `x,y` meaning |
| --- | --- |
| Stage 1 output | Box coordinates from detector |
| Stage 2 output | Recentered box coordinates |
| Stage 3 output | Filtered box coordinates |
| Stage 4 output | Classified box coordinates |
| Stage 7 output | Kept row from each duplicate group |
| Stage 8 and later | Center coordinates with `xy_semantics=center` |

After Stage 8:

- `x,y` are center coordinates.
- `w,h` are fixed 10x10 unless Stage 8.7 repairs a large flash with a larger square.
- `xy_semantics` should be `center`.

## Firefly Logits CSV

Path:

```text
ROOT/csv files/<video_stem>_fireflies_logits.csv
```

Built or rebuilt by:

```text
stage8_sync.py
```

Schema:

```text
t,x,y,background_logit,firefly_logit
```

Purpose:

- Slim firefly-only point/logit export.
- Rebuilt after Stage 7, Stage 8, Stage 8.5, Stage 8.6, and Stage 8.7 so it stays in sync with the main CSV.

## Stage 8 Crops

Path:

```text
ROOT/stage8 crops/<video_stem>/
```

Purpose:

- Debug crops around Stage 8 centroid refinement.
- Useful when final centers appear slightly off or when validating Gaussian sigma changes.

## Stage 8.6 Artifacts

Path:

```text
ROOT/stage8.6/<video_stem>/run_XX/
```

Typical contents:

- Blacked-out video or intermediate frame artifacts.
- Rerun CSVs.
- Intermediate stage outputs for detections discovered after masking current detections.

These are debugging outputs and can be large.

## Stage 8.7 Artifacts

Path:

```text
ROOT/stage8.7/<video_stem>/replacements/
```

Purpose:

- Side-by-side crops and replacement records for large-flash BFS repair.
- Useful for confirming whether big flashes were merged correctly or over-expanded.

## GT Contract

Night Stage 9 requires:

```text
x,y,t
```

at:

```text
ROOT/ground truth/gt.csv
```

If GT is still annotation-style:

```text
x,y,w,h,frame
```

Stage 8.9 can overwrite it as `x,y,t`.

Important warning:

- Stage 8.9 uses the `x,y` values from the input GT as the crop center for centroiding.
- If a source annotation tool wrote top-left coordinates, convert or verify the CSV before relying on Stage 8.9.

## Validation Outputs

Stage 9 output root:

```text
ROOT/stage9 validation/<video_stem>
```

Normalized GT:

```text
ROOT/stage9 validation/<video_stem>/<gt_stem>_norm_offset<offset>.csv
ROOT/csv files/<gt_stem>_norm_offset<offset>.csv
```

Threshold output:

```text
thr_10.0/
  fps.csv
  tps.csv
  fns.csv
  crops/
```

TP/FP/FN schema:

```text
x,y,t,filepath,confidence
```

## Render Outputs

Dynamic boxes:

```text
ROOT/original initial output annotated videos/<video_stem>_orig_annotated.mp4
ROOT/BS initial output annotated videos/<video_stem>_bs_annotated.mp4
```

Fixed 10px boxes:

```text
ROOT/original 10px overlay annotated videos/<video_stem>_orig_10px.mp4
```

GT/model overlays:

```text
ROOT/stage10 overlay videos/
```

Legend:

- GT: green.
- Model: red.
- Overlap: yellow.

