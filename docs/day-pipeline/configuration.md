# Day Pipeline Configuration

All routine day-pipeline configuration is in:

```text
Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/params.py
```

This file uses module-level constants. Direct pipeline runs import it once at startup. Gateway runs import it inside a subprocess and override selected constants dynamically.

## Root and Inputs

```python
ROOT = "/mnt/Samsung_SSD_2TB/temp to delete/day time pipeline inference output data"
ORIGINAL_VIDEOS_DIR = ROOT / "original videos"
GT_CSV_DIR = ROOT / "ground truth"
```

`ROOT` is the working directory for all intermediate and final outputs.

Direct-run layout:

```text
ROOT/
  original videos/
    video.mp4
  ground truth/
    gt_<video_stem>.csv
```

Gateway runs override `ROOT` and route outputs into:

```text
<gateway_output_root>/day_pipeline_v3
```

## Cleanup

```python
RUN_PRE_RUN_CLEANUP = True
```

When enabled, Stage 0 deletes generated output directories under `ROOT`, preserving only:

- `original videos`
- `ground truth`

Use `True` for clean direct runs on disposable output roots. Use caution when a root contains outputs you still need.

## Frame Bounds

```python
MAX_FRAMES = None
```

`None` processes the full video. Set an integer for smoke tests.

The gateway has `FORCE_ALL_FRAMES=True` by default, so it sets pipeline `MAX_FRAMES=None` unless `--max-frames` is explicitly provided.

## Long Exposure

```python
LONG_EXPOSURE_MODE = "lighten"
INTERVAL_FRAMES = 100
PROGRESS_EVERY = 50
```

Use `lighten` for normal day inference. The output images preserve the brightest signal in each interval, which makes small flashes visible to YOLO.

Change `INTERVAL_FRAMES` when:

- Too many detections are merged in time: reduce interval length.
- Flashes are too faint on long exposure: increase interval length or check exposure mode.
- Runtime or output size is too high: increase interval length.

## YOLO Stage

```python
YOLO_MODEL_WEIGHTS = Path(...)
YOLO_IMG_SIZE = None
YOLO_CONF_THRES = 0.01
YOLO_IOU_THRES = 0.15
YOLO_DEVICE = "auto"
YOLO_BATCH_SIZE = 8
YOLO_HALF_ON_CUDA = True
```

Default global day YOLO checkpoint:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/models/global models/20260414/best_firefly_yolo.pt
```

Tuning guidance:

- Raise `YOLO_CONF_THRES` to reduce candidates and speed Stage 3, at the risk of more FNs.
- Lower `YOLO_CONF_THRES` for recall, at the cost of more Stage 3 work.
- `YOLO_IOU_THRES=0.15` is intentionally strict because duplicate long-exposure boxes can multiply per-frame crops.
- Use a fixed `YOLO_IMG_SIZE` only if auto sizing causes memory or shape issues.

## Patch Classifier Stage

```python
PATCH_MODEL_PATH = DEFAULT_DAY_PATCH_MODEL_PATH
STAGE3_INPUT_SIZE = 10
STAGE3_POSITIVE_THRESHOLD = 0.80
STAGE3_DEVICE = "auto"
STAGE3_USE_AMP = True
PATCH_SIZE_PX = 10
```

Default direct-run patch model currently points into a previous temporary training run. Scaling and gateway runs can override this with model-zoo models.

Tuning guidance:

- Lower `STAGE3_POSITIVE_THRESHOLD` for more recall.
- Raise it for fewer false positives before Stage 3.1.
- Keep `PATCH_SIZE_PX` aligned with training data unless retraining.
- `IMAGENET_NORMALIZE` must match training.

## Stage 3.1 Trajectory Selection

```python
RUN_STAGE3_1_TRAJECTORY_INTENSITY_SELECTOR = True
STAGE3_1_LINK_RADIUS_PX = 15.0
STAGE3_1_MAX_FRAME_GAP = 6
STAGE3_1_MIN_TRACK_POINTS = 3
STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE = 2000
STAGE3_1_HIGHVAR_REQUIRE_HILL_SHAPE = True
```

This stage is a major precision filter. It assumes fireflies produce a time-localized intensity rise/fall.

Tuning symptoms:

- Many one-frame detections: lower `STAGE3_1_MIN_TRACK_POINTS` only if single-frame flashes are valid, otherwise improve linking.
- Real moving flashes rejected: increase `STAGE3_1_LINK_RADIUS_PX`.
- Smooth sustained noise retained: increase range threshold or tighten hill-shape criteria.

## Stage 3.2 Centroids

```python
RUN_STAGE3_2 = True
STAGE3_2_GAUSSIAN_SIGMA = 1.0
STAGE3_2_XYT_EXPORT_DIR = ROOT / "stage3_2 xyt for 3d reconstruction"
```

Use `STAGE3_2_GAUSSIAN_SIGMA=0` for plain intensity centroid. Use a positive sigma to weight the crop center more strongly.

## Rendering

```python
RENDER_CODEC = "mp4v"
RENDER_FPS_HINT = None
RENDER_ENFORCE_SOURCE_RESOLUTION = ENFORCE_SOURCE_VIDEO_RESOLUTION
OVERLAY_BOX_THICKNESS = VIDEO_RENDER_BBOX_THICKNESS_PX
STAGE4_DRAW_STAGE3_1_REJECTED = False
```

`STAGE4_DRAW_STAGE3_1_REJECTED=True` is useful when debugging why Stage 3.1 rejected detections. It draws rejected rows from `*_patches_motion_all.csv`.

## Validation and Test Toggles

```python
RUN_STAGE5_VALIDATE = True
RUN_STAGE6_OVERLAY = True
RUN_STAGE7_FN_ANALYSIS = False
RUN_STAGE8_FP_ANALYSIS = False
RUN_STAGE9_DETECTION_SUMMARY = False
DIST_THRESHOLDS_PX = [10.0]
GT_T_OFFSET = 0
```

Stage 7-9 analysis is slower and more output-heavy, so it defaults off.

Enable it when tuning parameters or preparing handoff evidence:

```python
RUN_STAGE7_FN_ANALYSIS = True
RUN_STAGE8_FP_ANALYSIS = True
RUN_STAGE9_DETECTION_SUMMARY = True
```

## What the Gateway Overrides

For day subprocesses, the gateway can override:

- `ROOT`
- `ORIGINAL_VIDEOS_DIR`
- all stage output dirs
- `GT_CSV_DIR`
- `PATCH_MODEL_PATH`
- `STAGE5_MODEL_PATH`
- `YOLO_MODEL_WEIGHTS`
- `MAX_FRAMES`
- `GT_T_OFFSET`
- `RUN_PRE_RUN_CLEANUP`
- validation/test toggles when `--force-tests` is passed

Because of this, prefer gateway flags for one-off inference rather than editing `params.py`.

