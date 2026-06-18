# Operations Runbook

## Before Running Anything

1. Confirm the SSD is mounted:

```bash
ls "/mnt/Samsung_SSD_2TB"
```

2. Confirm the repo path:

```bash
pwd
```

3. Use `python3`, not `python`.

4. Quote paths.

5. Check cleanup settings before direct pipeline runs:

- Day: `RUN_PRE_RUN_CLEANUP` in `params.py`.
- Night: `RUN_PRE_RUN_CLEANUP` in `pipeline_params.py`.

## Run One Video Through the Gateway

Use an explicit route first.

Day:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/video.mp4" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --route-override day \
  --force-tests \
  --max-concurrent 1
```

Night:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/video.mp4" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --route-override night \
  --force-tests \
  --max-concurrent 1
```

Check outputs under:

```text
/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data/day_pipeline_v3
/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data/night_time_pipeline
```

## Run a Folder Through the Gateway

Dry run first:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/folder" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --recursive \
  --dry-run
```

If routing is ambiguous, use separate commands with `--route-override day` or `--route-override night`.

## Direct Day Pipeline Run

1. Put videos under:

```text
/mnt/Samsung_SSD_2TB/temp to delete/day time pipeline inference output data/original videos
```

2. Put optional GT under:

```text
/mnt/Samsung_SSD_2TB/temp to delete/day time pipeline inference output data/ground truth
```

3. Run:

```bash
python3 "Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/orchestrator.py"
```

4. Inspect:

- `stage4_rendering/`
- `stage5 validation/`
- `stage6 overlay videos/`
- `stage3_2 xyt for 3d reconstruction/`

## Direct Night Pipeline Run

1. Put videos under:

```text
/mnt/Samsung_SSD_2TB/temp to delete/night time pipeline inference output data/original videos
```

2. Put optional GT at:

```text
/mnt/Samsung_SSD_2TB/temp to delete/night time pipeline inference output data/ground truth/gt.csv
```

3. Run:

```bash
python3 "Pipelines/night_time_pipeline/orchestrator.py"
```

4. Inspect:

- `csv files/`
- `original 10px overlay annotated videos/`
- `stage9 validation/`
- `stage10 overlay videos/`
- `stage14 detection summaries/`

## Ingest New Patch Training Data

Dry run:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" --dry-run
```

Restrict to a species:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" \
  --only-species "night_Photinus carolinus" \
  --dry-run
```

If the dry run is correct, remove `--dry-run`.

After ingestion, inspect:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/patch training datasets and pipeline validation data/Integrated_prototype_datasets
/mnt/Samsung_SSD_2TB/integrated prototype data/codex_change_log.jsonl
```

## Run Scaling Experiment

Plan only:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Small smoke:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --max-videos 2
```

Full run behavior depends on constants near the top of the script. Check model training toggles, inference toggles, baseline toggles, YOLO toggles, and species switches before running.

Outputs:

```text
/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/<run_id>
```

Key files:

- `final_results.csv`
- `run_results_manifest.json`
- `run_metadata.json`

## Run YOLO Dataset Helpers

Generate long exposures for configured day species:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py" --dry-run
```

Ingest YOLO export:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py" \
  "/path/to/yolo/source_or_zip" \
  --species-tag "photinus-greeni" \
  --dry-run
```

## Use the Annotation GUI

Run:

```bash
python3 "tools (other)/firefly flash annotation tool v2.py"
```

The GUI writes `x,y,w,h,frame` CSVs. It snaps clicks to the brightest local spot and uses 10x10 boxes.

## Troubleshooting

### No route found in gateway

Pass `--route-override day` or `--route-override night`.

### Day YOLO model path fails

Check the file exists. If the path contains an apostrophe, the code should copy it into `~/.cache/firefly_pipeline/ultralytics_weights`; inspect the Stage 2 log for the safe path.

### Night Stage 1 fails with CUDA/cuCIM errors

This is often environment-related. Try changing `STAGE1_VARIANT` to `cc_cpu` or `cc_cuda` for a smoke test, then fix CuPy/cuCIM/CUDA compatibility.

### Validation has many FNs with a constant time shift

Check `GT_T_OFFSET`, the GT CSV schema, and whether the GT frame index is raw-frame or clipped-frame based.

### Validation has coordinate mismatches

Check coordinate semantics:

- Annotator-style CSVs use `x,y,w,h,frame`, but verify whether `x,y` is top-left or center for the specific source.
- Day Stage 3 uses top-left patch boxes.
- Day Stage 3.2 uses centroid `x,y,t`.
- Night after Stage 8 uses center coordinates.

### Outputs disappeared

Check whether a direct pipeline run started with `RUN_PRE_RUN_CLEANUP = True` and the same `ROOT`.
