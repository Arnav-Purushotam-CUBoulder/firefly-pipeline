# Day YOLO Workflow

Day pipeline inference requires two model families:

1. Day YOLO model for long-exposure candidate detection.
2. Day patch classifier for per-frame crop filtering.

This page covers the YOLO side.

## YOLO Data Root

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data
```

Expected layout:

```text
dataset/
  <species>/
    images/
    labels/
ingest manifests/
models/
  global models/
  leave one out models/
  legacy models/
generated long exposure images/
```

## Current Default Global Checkpoint

The day pipeline default points at:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/models/global models/20260414/best_firefly_yolo.pt
```

The combo runner can auto-discover the newest dated global folder:

```python
AUTO_DISCOVER_LATEST_DAY_YOLO_MODEL = True
```

Manual fallback:

```python
DAY_YOLO_MODEL_WEIGHTS = DAY_YOLO_GLOBAL_MODELS_ROOT / "20260414" / "best_firefly_yolo.pt"
```

## Generate Long-Exposure Training Images

Script:

```text
tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py
```

Dry run:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py" --dry-run
```

Limited run:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py" --limit 2
```

Behavior:

- Loads the raw training/inference catalog.
- Selects training videos only.
- Reuses day pipeline Stage 1 settings.
- Skips inference clips to avoid leakage.
- Writes generated long-exposure images under the YOLO data root.

## Ingest YOLO Labels

Script:

```text
tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py
```

Dry run:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py" \
  "/path/to/yolo/export_or_zip" \
  --species-tag "photinus-greeni" \
  --dry-run
```

Useful flags:

- `--species-tag`
- `--dataset-root`
- `--local-image-root`
- `--extract-root`
- `--manifest-root`
- `--delete-zip`
- `--delete-extracted`
- `--dry-run`

Supported source types:

- extracted folder,
- zip,
- YOLO labels with images,
- labels only plus `--local-image-root`,
- COCO `_annotations.coco.json` source.

The ingestor validates labels, handles collisions, writes per-ingest manifests, and updates the species-grouped dataset.

## YOLO Training from Combo Runner

Combo-runner toggles:

```python
RUN_DAY_YOLO_MODEL_TRAINING = False
TRAIN_DAY_YOLO_GLOBAL_MODEL = False
TRAIN_DAY_YOLO_LEAVEOUT_MODELS = False
DAY_YOLO_TRAINING_SPECIES_SWITCHES = ...
```

Training defaults:

```python
DAY_YOLO_MODEL_WEIGHTS_INIT = "yolov8s.pt"
DAY_YOLO_EPOCHS = 50
DAY_YOLO_IMG_SIZE = None
DAY_YOLO_BATCH_SIZE = 1
DAY_YOLO_DEVICE = 0
DAY_YOLO_WORKERS = 2
DAY_YOLO_PATIENCE = 20
DAY_YOLO_LR0 = 0.01
DAY_YOLO_WEIGHT_DECAY = 0.0005
```

The runner builds a combined YOLO train dataset and writes a `data.yaml` that uses absolute paths so Ultralytics does not rewrite it under its global datasets directory.

## Global vs Leaveout YOLO Models

Global models:

```text
models/global models/<date>/best_firefly_yolo.pt
```

Leaveout models:

```text
models/leave one out models/<date>/<species>/best_firefly_yolo.pt
```

Important:

- Leaveout day patch inference also needs a matching leaveout day YOLO model.
- The combo runner will raise if leaveout day inference is enabled but required leaveout YOLO checkpoints are missing.

## Path Safety

Ultralytics can have trouble with apostrophes in paths. The combo runner and day Stage 2 use an apostrophe-safe cache:

```text
~/.cache/firefly_pipeline/ultralytics_weights
```

If a YOLO load fails and the original path contains an apostrophe, inspect the logs for the cached safe path.

## Recommended YOLO Update Flow

1. Confirm raw catalog marks intended videos as training.
2. Generate long-exposure images for training videos.
3. Label images externally.
4. Ingest labels with `tmp_ingest_daytime_yolo_dataset.py --dry-run`.
5. Run real ingest.
6. Train global or leaveout YOLO models through the combo runner.
7. Run gateway inference on held-out day clips.
8. Inspect Stage 2 annotated long exposures and Stage 5 validation.
9. Promote the checkpoint only with a manifest and dated folder.

