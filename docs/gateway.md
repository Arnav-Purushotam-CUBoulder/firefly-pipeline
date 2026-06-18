# Pipeline Gateway

Path:

```text
Pipelines/Pipeline Gateway/gateway.py
```

## Purpose

The gateway is the preferred runner when you have one video or a folder and want the correct day/night pipeline invoked without manually arranging pipeline roots.

It:

1. Finds videos from `--input`.
2. Routes each video to day or night.
3. Creates route-specific output roots.
4. Points the subprocess at the selected source video by overriding the pipeline's video iterator.
5. Mutates the pipeline config inside a subprocess.
6. Runs the day or night orchestrator.
7. Keeps cleanup from deleting other concurrent jobs when needed.

## Basic Commands

Single day video:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/video.mp4" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --route-override day \
  --force-tests \
  --max-concurrent 1
```

Single night video:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/video.mp4" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --route-override night \
  --force-tests \
  --max-concurrent 1
```

Folder dry run:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/folder" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --recursive \
  --dry-run
```

## Routing

Brightness routing is currently disabled. Routing is based on:

- `--route-override day`
- `--route-override night`
- Day/night tokens in the video path/name

If both day and night tokens are found, the gateway raises an error. If no route token is found and no override is supplied, it raises an error.

For scaling runs, use explicit `--route-override` because the catalog already knows the route.

## Output Layout

Default output base:

```text
/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data
```

Subfolders:

```text
day_pipeline_v3/
night_time_pipeline/
```

Each subfolder is shaped like the direct pipeline `ROOT`, with `original videos`, `ground truth`, stage outputs, renders, and validation outputs.

Gateway runs do not copy the source video into `original videos`. The subprocess processes the original source path directly while writing outputs into the route root.

## Ground Truth Discovery

The gateway looks for GT in the output root to infer frame bounds and offsets:

- `ground truth/gt_<video_stem>.csv`
- `ground truth/gt.csv`
- `gt.csv`

If `--max-frames` is provided, it overrides GT-based frame-bound inference.

## Important Flags

| Flag | Meaning |
| --- | --- |
| `--input` | Video file or folder. Required unless `INPUT_PATH` is edited in the script. |
| `--output-root` | Base folder for day/night output subfolders. |
| `--route-override` | Force all input videos to `day` or `night`. |
| `--day-patch-model` | Override day patch classifier. |
| `--day-yolo-model` | Override day YOLO checkpoint. |
| `--night-cnn-model` | Override night classifier. |
| `--force-tests` | Force validation/test stages on. |
| `--max-concurrent` | Number of videos to process concurrently. |
| `--max-frames` | Explicit frame cap. |
| `--recursive` | Recurse through input folder. |
| `--dry-run` | Print routing decisions without running pipelines. |

Deprecated and ignored:

- `--threshold`
- `--frames`

## Cleanup and Concurrency

The underlying pipelines normally run pre-run cleanup. The gateway disables cleanup after the first route or when concurrency would make cleanup dangerous. This prevents one subprocess from deleting another video's outputs.

For repeatable production-style runs, use `--max-concurrent 1` first. Increase concurrency only after validating output isolation and GPU memory.

## Model Overrides

The gateway dynamically overrides:

Day:

- `ROOT` and stage output dirs.
- `PATCH_MODEL_PATH`.
- `STAGE5_MODEL_PATH`.
- `YOLO_MODEL_WEIGHTS`.
- `MAX_FRAMES`.
- `GT_T_OFFSET`.
- cleanup behavior.

Night:

- `ROOT` and output dirs.
- `CNN_MODEL_PATH`.
- `STAGE9_MODEL_PATH`.
- `MAX_FRAMES`.
- `GT_T_OFFSET`.
- cleanup behavior.

This means gateway runs are usually safer than permanently editing pipeline config files for one-off inference.
