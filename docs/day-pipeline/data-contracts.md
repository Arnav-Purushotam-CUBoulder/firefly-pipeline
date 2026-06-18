# Day Pipeline Data Contracts

This page documents the files that stages consume and produce. It is useful when adding wrappers, writing tests, or comparing outputs against night/baseline results.

## Input Video Contract

Supported extensions come from `params.VIDEO_EXTS`:

```text
.mp4, .avi, .mov, .mkv
```

Direct pipeline runs read videos from:

```text
ROOT/original videos
```

Gateway runs override `params.list_videos()` to return the selected source path directly. The source video is not copied into `ROOT/original videos` by the gateway.

## Ground Truth Discovery

The day orchestrator searches `GT_CSV_DIR` in this order:

```text
<video_stem>.csv
<video_stem>_gt.csv
gt_<video_stem>.csv
gt.csv
<video_stem>*.csv
gt_<video_stem>*.csv
```

Then it falls back to `GT_CSV_PATH` if set.

Supported GT schemas:

```text
x,y,t
x,y,w,h,frame
```

For `x,y,w,h,frame`, the validator parses `frame`, applies `GT_T_OFFSET`, and uses the provided `x,y` values as the GT point. It does not reliably infer center from `w,h`. Verify whether a given annotation source stores top-left or center coordinates before validation.

## Stage 1 Filename Contract

Stage 1 writes:

```text
<interval>_<video_stem>_<mode>_<start>-<end>.png
```

Stage 2 parses `<start>-<end>` from this filename to populate `frame_range`.

Do not rename Stage 1 outputs unless the Stage 2 parser is updated.

## Stage 2 CSV

Path:

```text
ROOT/stage2_yolo_detections/<video_stem>/<video_stem>.csv
```

Schema:

```text
x,y,w,h,frame_range,video_name
```

Semantics:

- `x,y,w,h` are YOLO box coordinates in long-exposure image space.
- `frame_range` is the original frame range represented by that long-exposure PNG.
- `video_name` is the source video filename.

## Stage 3 CSV

Path:

```text
ROOT/stage3_patch_classifier/<video_stem>/<video_stem>_patches.csv
```

Schema:

```text
frame_idx,video_name,x,y,w,h,conf,det_id
```

Semantics:

- One row per patch-classifier positive.
- `frame_idx` is the source-video frame index.
- `x,y` are top-left patch coordinates.
- `conf` is the patch classifier positive probability.
- `det_id` ties rows back to the originating Stage 2 detection.

Positive crop path pattern:

```text
ROOT/stage3_patch_classifier/<video_stem>/crops/positives/f_<t>_x<x>_y<y>_w<w>_h<h>_p<p>.png
```

Stage 3.2 uses these filenames to match CSV rows back to crop files.

## Stage 3.1 CSVs

Selected-only path:

```text
ROOT/stage3_patch_classifier/<video_stem>/<video_stem>_patches_motion.csv
```

All trajectories path:

```text
ROOT/stage3_patch_classifier/<video_stem>/<video_stem>_patches_motion_all.csv
```

Schema:

```text
frame_idx,video_name,x,y,w,h,conf,det_id,traj_id,traj_size,traj_motion_xy,traj_intensity_range,traj_is_selected
```

Semantics:

- `traj_is_selected=1` rows are retained for Stage 3.2.
- `traj_is_selected=0` rows are rejected but available for debugging.
- `traj_intensity_range` is the range of intensity over the linked crop sequence.

## Stage 3.2 Logits CSV

Path:

```text
ROOT/stage3_patch_classifier/<video_stem>/stage3_2/<video_stem>_stage3_2_firefly_background_logits.csv
```

Schema:

```text
x,y,t,firefly_logit,background_logit
```

Semantics:

- `x,y` are full-frame refined centroid coordinates.
- `t` is the source-video frame index.
- Logits are derived from Stage 3 patch confidence.

## Stage 3.2 Reconstruction CSV

Path:

```text
ROOT/stage3_2 xyt for 3d reconstruction/<video_stem>.csv
```

Schema:

```text
x,y,t
```

This is the clean day output for downstream reconstruction.

## Stage 5 Prediction CSV

The day orchestrator builds a validator-facing CSV under:

```text
ROOT/stage5 validation/<video_stem>/<video_stem>.csv
```

Schema:

```text
x,y,w,h,t,class,xy_semantics,firefly_logit,background_logit
```

Source:

- Preferred: Stage 3.2 logits.
- Fallback: Stage 3 patch confidence converted to logits and centered from top-left boxes.

This CSV exists for validation/overlay compatibility. It is not the raw model output.

## Coordinate Semantics Summary

| Artifact | `x,y` meaning |
| --- | --- |
| Annotator-style GT `x,y,w,h,frame` | Source-dependent; verify top-left vs center before validation |
| Normalized GT `x,y,t` | Center point |
| Stage 2 CSV | Long-exposure box top-left |
| Stage 3 CSV | Top-left patch coordinates |
| Stage 3.1 CSV | Top-left patch coordinates |
| Stage 3.2 logits | Refined center point |
| Stage 3.2 reconstruction CSV | Refined center point |
| Stage 5 prediction CSV | Center point, with `xy_semantics=center` |

Always check the artifact, not just the column names.
