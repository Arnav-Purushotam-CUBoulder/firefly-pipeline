# Day Pipeline Operations and Debugging

## Preferred Run Path

Use the gateway for one-off videos or mixed data folders:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/day_video.mp4" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --route-override day \
  --force-tests \
  --max-concurrent 1
```

Why gateway is preferred:

- It avoids permanent edits to `params.py`.
- It can override model paths.
- It controls cleanup for subprocess/concurrent runs.
- It can infer or pass frame caps from GT.

## Direct Run Path

1. Put source videos here:

```text
/mnt/Samsung_SSD_2TB/temp to delete/day time pipeline inference output data/original videos
```

2. Put optional GT here:

```text
/mnt/Samsung_SSD_2TB/temp to delete/day time pipeline inference output data/ground truth
```

3. Run:

```bash
python3 "Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/orchestrator.py"
```

## Minimal Smoke Test

For a fast direct smoke test:

1. Set `MAX_FRAMES` to a small value such as `300`.
2. Leave `RUN_PRE_RUN_CLEANUP=True`.
3. Use one short video in `original videos`.
4. Run the orchestrator.
5. Confirm these exist:

```text
stage1_long_exposure/<stem>/*.png
stage2_yolo_detections/<stem>/<stem>.csv
stage3_patch_classifier/<stem>/<stem>_patches.csv
stage4_rendering/<stem>/<stem>_patches.mp4
```

If Stage 3.2 is enabled, also confirm:

```text
stage3_2 xyt for 3d reconstruction/<stem>.csv
```

## Output Review Order

When debugging a video, inspect in this order:

1. Stage 1 PNGs: are flashes visible in long exposure?
2. Stage 2 annotated PNGs: did YOLO find the correct broad regions?
3. Stage 3 positive crops: are positive patches real flashes or noise?
4. Stage 3.1 SVG/highvar video: did trajectory selection reject real flashes?
5. Stage 3.2 `x,y,t`: are centroids plausible?
6. Stage 4 render: do boxes track the visible flashes?
7. Stage 5/6 validation overlays: do predictions align with GT?

## Common Failures

### No videos found

Direct run cause:

- `ROOT/original videos` is empty or `ROOT` points to the wrong folder.

Gateway cause:

- `--input` path is wrong or extension is not in `VIDEO_EXTS`.

### Stage 2 cannot load YOLO

Check:

- `ultralytics` installed.
- `YOLO_MODEL_WEIGHTS` exists.
- GPU/CPU device selected correctly.
- Logs mention an apostrophe-safe copied weights path if the original path contains an apostrophe.

### Stage 3 cannot load patch model

Check:

- `PATCH_MODEL_PATH` exists.
- Model architecture matches `resnet18`.
- Normalization matches training.
- Gateway `--day-patch-model` path is correct.

### Stage 3 is slow

Likely causes:

- Stage 2 produced too many YOLO boxes.
- `YOLO_CONF_THRES` is too low.
- `INTERVAL_FRAMES` creates too many intervals.
- Running on CPU.

Actions:

- Raise `YOLO_CONF_THRES` slightly.
- Confirm CUDA is available.
- Reduce frame cap for debugging.
- Inspect Stage 2 annotated images for duplicate boxes.

### Stage 3.1 rejects real flashes

Check:

- `*_patches_motion_all.csv` and `traj_is_selected`.
- Intensity SVGs.
- `STAGE3_1_LINK_RADIUS_PX`.
- `STAGE3_1_MAX_FRAME_GAP`.
- Hill-shape and range thresholds.

If real flashes move quickly, the link radius may be too small.

### Stage 3.2 output is empty

Check:

- Stage 3.1 selected CSV exists and has rows.
- Positive crop filenames match Stage 3.2 expected pattern.
- `RUN_STAGE3_2=True`.

### Validation skipped

The orchestrator skips validation if GT cannot be found. Check accepted names under `GT_CSV_DIR`:

```text
<video_stem>.csv
<video_stem>_gt.csv
gt_<video_stem>.csv
gt.csv
```

### Many FNs and FPs but overlay looks shifted in time

Check:

- `GT_T_OFFSET`.
- Whether GT frame values are raw frame numbers.
- Gateway `--max-frames` and GT inference logs.

### Many FPs that look like true fireflies

Possibilities:

- GT annotation is incomplete.
- Match radius too strict.
- Predictions are center coordinates but GT normalization is wrong.

Use Stage 8 FP analysis and Stage 6 overlays before tightening model thresholds.

## Safe Parameter-Tuning Loop

1. Copy or isolate one representative video.
2. Use gateway with `--route-override day` and `--max-frames` for speed.
3. Make one parameter change at a time.
4. Compare Stage 5 TP/FP/FN counts and Stage 6 overlay.
5. Use Stage 7/8 analysis only after narrowing down the issue.
6. Keep notes of the exact model paths and params used.

