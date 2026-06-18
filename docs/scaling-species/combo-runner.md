# Combo Runner

Main script:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

Despite the `tmp_` name, this is the central scaling experiment runner.

## Purpose

The combo runner coordinates:

1. Discovering training sources from integrated patch datasets.
2. Building the raw training/inference catalog.
3. Training route-specific patch classifiers when enabled.
4. Loading model-zoo patch classifiers when training is disabled.
5. Training or resolving day YOLO models when enabled.
6. Selecting held-out inference videos from the catalog.
7. Running gateway inference for global and leaveout patch models.
8. Running lab and Raphael baselines.
9. Writing final result rows, manifests, and metadata.

## Commands

Dry run:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Limit video count for smoke test:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --max-videos 2
```

Override raw root:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" \
  --raw-videos-root "/path/to/raw/root" \
  --dry-run
```

Override runs root:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" \
  --runs-root "/path/to/runs/root" \
  --dry-run
```

## Main Paths

```python
REPO_ROOT = "/home/guest/Desktop/arnav's files/firefly pipeline"
RAW_VIDEOS_ROOT = "/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)"
RUNS_ROOT = "/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run"
INTEGRATED_OUTER_ROOT = "/mnt/Samsung_SSD_2TB/integrated prototype data"
INTEGRATED_INNER_ROOT = INTEGRATED_OUTER_ROOT / "integrated prototype data"
PATCH_DATA_ROOT = INTEGRATED_INNER_ROOT / "patch training datasets and pipeline validation data"
MODEL_ZOO_ROOT = INTEGRATED_INNER_ROOT / "model zoo"
BASELINES_DATA_ROOT = INTEGRATED_OUTER_ROOT / "Baselines data"
```

Run output root:

```text
RUNS_ROOT/tmp_day_night_combo_train_and_infer__<date>__<time>/
```

Important output files:

```text
final_results.csv
run_results_manifest.json
run_metadata.json
logs/
```

## Route Metadata

Canonical species routing is defined in `ROUTE_BY_SPECIES`:

Day:

- `bicellonycha-wickershamorum`
- `photinus-acuminatus`
- `photinus-greeni`
- `photuris-bethaniensis`

Night:

- `forresti`
- `frontalis`
- `photinus-carolinus`
- `photinus-knulli`
- `tremulans`

The runner can also use `ROUTE_BY_VIDEO_STEM` for exact video-stem overrides.

## Main Switch Groups

### Patch model training

```python
RUN_MODEL_TRAINING = False
RUN_DAY_MODEL_TRAINING = False
RUN_NIGHT_MODEL_TRAINING = False
TRAIN_GLOBAL_MODELS = False
TRAIN_LEAVEOUT_MODELS = False
```

If all are false, the runner loads existing model-zoo patch models when pipeline inference is enabled.

### Pipeline inference

```python
RUN_DAY_PIPELINE_INFERENCE = False
RUN_NIGHT_PIPELINE_INFERENCE = False
RUN_GLOBAL_MODEL_INFERENCE = False
RUN_LEAVEOUT_MODEL_INFERENCE = False
```

Global/leaveout inference has per-species switches:

```python
GLOBAL_DAY_INFERENCE_SPECIES_SWITCHES
GLOBAL_NIGHT_INFERENCE_SPECIES_SWITCHES
LEAVEOUT_DAY_INFERENCE_SPECIES_SWITCHES
LEAVEOUT_NIGHT_INFERENCE_SPECIES_SWITCHES
```

### Baselines

```python
RUN_LAB_BASELINE = True
RUN_RAPHAEL_BASELINE = True
LAB_BASELINE_SPECIES_SWITCHES
RAPHAEL_BASELINE_SPECIES_SWITCHES
```

Baseline switches are independent from pipeline inference switches.

### Day YOLO training

```python
RUN_DAY_YOLO_MODEL_TRAINING = False
TRAIN_DAY_YOLO_GLOBAL_MODEL = False
TRAIN_DAY_YOLO_LEAVEOUT_MODELS = False
DAY_YOLO_TRAINING_SPECIES_SWITCHES
```

Day pipeline inference always needs a day YOLO model in addition to the patch classifier.

## Execution Phases

The high-level order in `main()` is:

1. Validate config, routes, and toggles.
2. Discover training sources from integrated patch datasets.
3. Optionally train patch models.
4. Resolve model-zoo patch models if not training.
5. Optionally train day YOLO models.
6. Resolve latest day YOLO global model.
7. Build and write the raw training/inference catalog.
8. Select pipeline inference videos.
9. Select baseline videos.
10. Load/normalize GT for selected videos.
11. Run global model inference through gateway.
12. Run leaveout model inference through gateway.
13. Run baselines.
14. Write baseline registries.
15. Write final results, manifest, and run metadata.

## Gateway Invocation

For each selected pipeline inference video, the runner calls:

```text
Pipelines/Pipeline Gateway/gateway.py
```

It passes:

- `--input`
- `--output-root`
- `--route-override`
- route-specific patch/CNN model path
- day YOLO model path when route is day
- `--max-concurrent`
- optional frame bounds
- optional `--force-tests`

The gateway then runs the day or night orchestrator and validation.

## Final Results CSV

Path:

```text
<run_root>/final_results.csv
```

Columns:

```text
run_id
route
species_name
video_name
eval_type
model_used
results
inference_output_path
lab_results
lab_output_path
raphael_results
raphael_output_path
gt_source
gt_rows
gt_max_t
```

`eval_type` values include:

- `global`
- `leaveout`
- `baseline`

## Manifests

`run_results_manifest.json` is the structured results manifest.

`run_metadata.json` records:

- raw roots,
- run toggles,
- model roots and model sources,
- YOLO settings,
- baseline settings,
- training source summaries,
- catalog summary,
- training metrics,
- baseline GT details.

Use these files to reproduce or audit a run.

## Safety Checks

Important safeguards:

- `REQUIRE_GT_FOR_INFERENCE = True` by default.
- `REUSE_EXISTING_MODELS_IF_PRESENT = False` to avoid silent stale checkpoints.
- Leaveout day inference requires matching leaveout day YOLO checkpoints.
- If nothing is enabled, the runner exits instead of doing a no-op run.

## Practical Advice

Always run `--dry-run` after changing toggles.

For real runs, capture:

- Git diff,
- `run_metadata.json`,
- `run_results_manifest.json`,
- final model paths,
- stdout/stderr logs under the run root.

