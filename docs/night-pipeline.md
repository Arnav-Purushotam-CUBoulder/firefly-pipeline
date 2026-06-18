# Night-Time Pipeline

Path:

```text
Pipelines/night_time_pipeline
```

Main config:

```text
Pipelines/night_time_pipeline/pipeline_params.py
```

Main runner:

```text
Pipelines/night_time_pipeline/orchestrator.py
```

## Purpose

The night pipeline handles dark-background videos where firefly flashes appear as bright blobs. It detects blobs, recenters and filters them, classifies patches with a CNN, merges duplicates, refines centroids, performs repair passes, renders videos, and validates against GT.

## Input Layout

`pipeline_params.ROOT` must contain:

```text
original videos/
```

Optional:

```text
BS videos/
ground truth/gt.csv
```

Default root:

```text
/mnt/Samsung_SSD_2TB/temp to delete/night time pipeline inference output data
```

## Model

Default CNN model:

```text
/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/tmp_day_night_combo_train_and_infer__20260305__163237/models/night/global_all_species.pt
```

When running through the gateway, override with `--night-cnn-model`.

## Run

From repo root:

```bash
python3 "Pipelines/night_time_pipeline/orchestrator.py"
```

For safer single-video use, prefer the gateway:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/night_video.mp4" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --route-override night \
  --force-tests \
  --max-concurrent 1
```

## Stages

| Stage | File | What it does | Main output |
| --- | --- | --- | --- |
| 0 | `stage0_cleanup.py` | Deletes generated outputs, preserving `ground truth` and `original videos`; can copy newest root CSV to `ground truth/gt.csv`. | Clean working root |
| 1 | `stage1_detect_cucim.py` or variant | Detects bright blobs. | `csv files/<stem>.csv` |
| 2 | `stage2_recenter.py` | Recenters boxes by intensity centroid and drops dim crops. | Updated CSV |
| 3 | `stage3_area_filter.py` | Filters small detections by bright-pixel area. | Updated CSV |
| 4 | `stage4_cnn_filter.py` | Applies ResNet18 firefly/background classifier. | Updated CSV with class/logits/confidence |
| 5 | `stage5_render.py` | Renders initial dynamic boxes. | Annotated videos |
| 6 | `stage6_10px_renderer.py` | Renders fixed 10px boxes. | `original 10px overlay annotated videos/` |
| 7 | `stage7_merge.py` | Union-find duplicate merge by centroid distance. | Updated CSV |
| 8 | `stage8_gaussian_centroid.py` | Refines centroid, fixes 10x10 box semantics, writes logits CSV. | `*_fireflies_logits.csv`, stage8 crops |
| 8.5 | `stage8_5_blob_area_filter.py` | Removes detections with too-small bright connected components. | Updated CSV/logits |
| 8.6 | `stage8_6_neighbor_hunt.py` | Masks known detections, reruns detection on blacked-out video, merges new neighbors. | `stage8.6/<stem>/run_*` |
| 8.7 | `stage8_7_large_flash_bfs.py` | Repairs large flashes by BFS region growth and replacement. | `stage8.7/<stem>/replacements` |
| 8.9 | `stage8_9_gt_gaussian_centroid.py` | Recenters annotator-style GT and overwrites it as `x,y,t`. | `ground truth/gt.csv` |
| 9 | `stage9_validate.py` | Matches predictions to GT. | `stage9 validation/<stem>/thr_*` |
| 10 | `stage10_overlay_gt_vs_model.py` | Renders GT/model/overlap videos. | `stage10 overlay videos/` |
| 11 | `stage11_fn_analysis.py` | False-negative analysis. | FN analysis outputs |
| 12 | `stage12_fp_analysis.py` | False-positive analysis. | FP analysis outputs |
| 14 | `stage14_detection_summary.py` | JSON summary for tuning. | `stage14 detection summaries/` |

## Stage 1 Variants

Configured by `STAGE1_VARIANT`:

- `cucim` - default GPU cuCIM blob detector.
- `cc_cuda` - CUDA connected-components detector.
- `cc_cpu` - CPU connected-components detector.
- SimpleBlobDetector path via `stage1_detect.py`.

If the default cuCIM path fails on a new machine, try a small run with `cc_cpu` or `cc_cuda` before changing thresholds.

## Important Parameters

Common edit points in `pipeline_params.py`:

- `ROOT` - working folder.
- `MAX_FRAMES` - frame cap.
- `RUN_PRE_RUN_CLEANUP` - defaults to `True`.
- `STAGE1_VARIANT`.
- `BRIGHT_MAX_THRESHOLD`.
- `AREA_THRESHOLD_PX`.
- `CNN_MODEL_PATH`, `FIREFLY_CONF_THRESH`, `DROP_BACKGROUND_ROWS`.
- `STAGE7_DIST_THRESHOLD_PX`.
- `STAGE8_GAUSSIAN_SIGMA`.
- `STAGE8_6_RUNS`, `STAGE8_6_DEDUPE_PX`.
- `STAGE8_7_INTENSITY_THR`, `STAGE8_7_MIN_SQUARE_AREA_PX`.
- `GT_CSV_PATH`, `GT_T_OFFSET`, `DIST_THRESHOLDS_PX`.

## Output Semantics

Early night CSVs use frame-level boxes.

After Stage 8, the final CSV is rewritten so:

- `x,y` are center coordinates.
- `w,h` are fixed 10x10 box dimensions.
- `xy_semantics` should be `center`.

The firefly logits CSV contains frame-wise centroid predictions and logits for downstream scoring/reconstruction.

## Operational Risks

- Stage 0 cleanup deletes generated outputs under `ROOT`.
- Stage 0 has special behavior for `ground truth/gt.csv`; inspect GT files before running direct night orchestration.
- Stage 8.9 overwrites GT into normalized `x,y,t` form if the input GT is annotator-style.
- The default Stage 1 requires GPU-side dependencies. Missing CuPy/cuCIM/CUDA support is an environment issue, not necessarily a pipeline logic issue.

