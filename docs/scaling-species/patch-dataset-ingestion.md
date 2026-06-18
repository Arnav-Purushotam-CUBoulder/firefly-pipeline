# Patch Dataset Ingestion

Main script:

```text
tools for scaling species/patch_classification_models_dataset_ingestor.py
```

Purpose:

- Scan raw route-prefixed species folders.
- Match videos to firefly/background annotation CSVs.
- Stage CSVs into canonical names.
- Extract firefly and background patches.
- Update single-species and integrated patch datasets.
- Record changes in an append-only change log.

## Commands

Dry run all species:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" --dry-run
```

Dry run one raw folder:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" \
  --only-species "day_Photinus greeni" \
  --dry-run
```

Override roots:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" \
  --raw-root "/path/to/raw/root" \
  --root "/path/to/integrated inner root" \
  --log-root "/path/to/log root" \
  --dry-run
```

## Important Defaults

```python
RAW_VIDEOS_ROOT = "/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)"
ROOT_PATH = "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data"
LOG_ROOT_PATH = "/mnt/Samsung_SSD_2TB/integrated prototype data"
DATA_SUBDIR = "patch training datasets and pipeline validation data"
CHANGE_LOG_FILENAME = "codex_change_log.jsonl"
```

Dataset root:

```text
ROOT_PATH/DATA_SUBDIR/Integrated_prototype_datasets
```

Change log:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/codex_change_log.jsonl
```

## Catalog-Based Split

Default:

```python
USE_ROOT_VIDEO_CATALOG_IF_PRESENT = True
INGEST_TRAINING_VIDEOS_FROM_ROOT_CATALOG = True
TRAIN_PAIR_FRACTION = 0.8
```

If the root catalog exists, training videos come from the catalog. `TRAIN_PAIR_FRACTION` is only a fallback if catalog usage is disabled or unavailable.

This protects held-out inference videos from being accidentally ingested into training data.

## Dataset Versions

Default:

```python
ONE_DATASET_VERSION_PER_BATCH = True
DATASET_VERSION_COPY_MODE = "hardlink"
```

The ingestor creates one new version per species batch. It copies or hardlinks previous version contents, adds new patches, and rebuilds final train/val/test splits.

Single-species dataset layout:

```text
Integrated_prototype_datasets/
  single species datasets/
    <species>/
      <version>/
        initial dataset/
          firefly/
          background/
        final dataset/
          train/
            firefly/
            background/
          val/
            firefly/
            background/
          test/
            firefly/
            background/
        patch_locations.csv
        patch_locations_background.csv
```

Integrated route dataset layout:

```text
Integrated_prototype_datasets/
  integrated pipeline datasets/
    <version>/
      day_time_dataset/
        initial dataset/
        final dataset/
      night_time_dataset/
        initial dataset/
        final dataset/
```

## Canonical Staging Filenames

The ingestor stages raw CSVs into names expected by the inlined stage1 ingestion core:

```text
<video_name>_<species_name>_<day_time|night_time>_<firefly|background>.csv
```

Why this matters:

- The stage1 core parses identity from underscores.
- Species tokens must not contain underscores.
- Staging catches name collisions early.

Staging area:

```text
DATA_ROOT/batch_exports/ingestor_only_observed_dir_staging/<run_tag>__<raw_species_dir>/
```

Successful ingests remove scratch exports when:

```python
CLEAN_BATCH_EXPORTS_AFTER_SUCCESS = True
```

## Auto Background Patches

Default:

```python
AUTO_GENERATE_BACKGROUND_PATCHES = True
AUTO_BACKGROUND_TO_FIREFLY_RATIO = 10.0
AUTO_BACKGROUND_PATCH_SIZE_PX = 10
AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME = 10
AUTO_BACKGROUND_MAX_FRAME_SAMPLES = 5000
AUTO_BACKGROUND_SEED = 1337
AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS = True
```

Behavior:

- Uses training firefly annotations to avoid firefly frames.
- Samples background frames from non-firefly ranges.
- Attempts blob-like background sampling.
- Falls back to random centers if blob detection becomes too slow or unavailable.

Slow-frame guard:

```python
AUTO_BACKGROUND_BLOB_DETECT_SLOW_FRAME_SECONDS = 2.0
AUTO_BACKGROUND_BLOB_DETECT_DISABLE_AFTER_SLOW_FRAMES = 1
```

This prevents background generation from stalling on high-resolution videos.

## Change Log

Default:

```python
ENABLE_CODEX_CHANGE_LOG = True
```

The ingestor snapshots relevant folders before and after ingestion and writes append-only JSONL records to:

```text
codex_change_log.jsonl
```

Patch image folders are treated as bulk records to keep the log size reasonable.

Use the change log to answer:

- which species/video was ingested,
- which dataset version was created,
- which files changed,
- what config toggles were active.

## Ingestion Safety Checks

The ingestor stops when:

- a raw species folder is not route-prefixed,
- no valid video/firefly CSV pairs are found,
- multiple firefly CSVs match the same video,
- catalog references a training video that is not present as a discovered pair,
- root/log/data paths are missing,
- species token is invalid for canonical parsing.

## Recommended Workflow

1. Add raw videos and CSVs to the correct `day_` or `night_` folder.
2. Update/regenerate the root catalog.
3. Run ingestor dry run restricted to the species.
4. Inspect planned training count and version paths.
5. Run real ingestion.
6. Inspect new single-species and integrated dataset versions.
7. Check `codex_change_log.jsonl`.
8. Only then train or run scaling evaluation.

