# Scaling Species

Path:

```text
tools for scaling species
```

For detailed takeover docs, start with:

```text
docs/scaling-species/README.md
```

Detailed pages:

- [scaling-species/raw-data-and-catalog.md](scaling-species/raw-data-and-catalog.md)
- [scaling-species/patch-dataset-ingestion.md](scaling-species/patch-dataset-ingestion.md)
- [scaling-species/combo-runner.md](scaling-species/combo-runner.md)
- [scaling-species/model-zoo-and-training.md](scaling-species/model-zoo-and-training.md)
- [scaling-species/day-yolo-workflow.md](scaling-species/day-yolo-workflow.md)
- [scaling-species/baselines-and-evaluation.md](scaling-species/baselines-and-evaluation.md)
- [scaling-species/add-species-runbook.md](scaling-species/add-species-runbook.md)
- [scaling-species/worked-example-new-day-species.md](scaling-species/worked-example-new-day-species.md)

This folder contains the tooling that turns raw annotated videos into scalable datasets, trains or selects models, runs inference through the gateway, evaluates against GT, and compares against legacy baselines.

## Scaling Mental Model

The two application pipelines are fixed targets:

- Day route: long exposure + YOLO + patch classifier.
- Night route: blob detection + CNN classifier + repair passes.

The scaling layer manages everything around those targets:

1. Discover raw species folders and annotations.
2. Split clips into training and held-out inference using the raw catalog.
3. Build patch datasets.
4. Train or select global and leaveout patch classifiers.
5. Train or select day YOLO models.
6. Run gateway inference on held-out clips.
7. Run baselines for comparison.
8. Write final result manifests.

## Patch Dataset Ingestor

Script:

```text
tools for scaling species/patch_classification_models_dataset_ingestor.py
```

Dry run:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" --dry-run
```

Restrict to one raw species folder:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" \
  --only-species "day_Photinus greeni" \
  --dry-run
```

Important defaults:

- Raw root: `/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)`
- Integrated root: `/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data`
- Log root: `/mnt/Samsung_SSD_2TB/integrated prototype data`
- Uses the raw catalog if present.
- Ingests only training videos from the catalog.
- Auto-generates background patches by default.
- Uses append-only `codex_change_log.jsonl` to know what was already ingested.

Expected raw folder naming:

```text
day_<species display name>
night_<species display name>
```

Species tokens are normalized to kebab case. Avoid underscores in species tokens because staged canonical filenames are parsed by underscores.

## Combo Training and Inference Runner

Script:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

Dry run:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Smoke test with first N discovered videos:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --max-videos 2
```

This runner orchestrates:

- Raw catalog creation/update.
- Route-specific patch model training if enabled.
- Global and leaveout patch models if enabled.
- Day YOLO global and leaveout training if enabled.
- Gateway inference for global and leaveout model evaluation.
- Lab/Nolan and Raphael baseline evaluation.
- Final results CSV and run manifest writing.

Outputs are written under:

```text
/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/<run_id>
```

Permanent model/baseline artifacts may be exported to the integrated data root or baseline root depending on toggles.

Important current behavior:

- Patch model training toggles default off.
- Day YOLO training toggles default off.
- Global and leaveout inference toggles are available.
- Baseline toggles are controlled near the top of the script.
- `REQUIRE_GT_FOR_INFERENCE = True` protects evaluation from clips without GT.
- Day YOLO inference auto-discovers the newest dated global YOLO checkpoint unless configured otherwise.

## Model Zoo

Patch model zoo:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/model zoo
```

Expected global models:

```text
global models/daytime global/best.pt
global models/night time global/best.pt
```

Expected leaveout models:

```text
leave out models/<species>/best.pt
```

Each model directory should also include `training_manifest.json`.

## Day YOLO Tools

Folder:

```text
tools for scaling species/YOLO model dataset creation, ingestion, and training scripts
```

Generate day-pipeline long-exposure training images:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py" --dry-run
```

Ingest a YOLO source folder or zip:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py" \
  "/path/to/yolo/export_or_zip" \
  --species-tag "photinus-greeni" \
  --dry-run
```

The YOLO dataset root is:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/dataset
```

Expected species-grouped layout:

```text
<species>/
  images/
  labels/
```

## Legacy Baselines

Folder:

```text
tools for scaling species/legacy_baselines
```

Main scripts:

- `nolan_mp4_to_predcsv.py` - lab baseline from video to Stage-5-style predictions CSV.
- `raphael_oorb_detect_and_gauss.py` - Python port of Raphael OOrb detection plus Gaussian centroid refinement.
- `match_predictions_to_processed_gt.py` - baseline prediction/GT matcher.
- `render_baseline_predictions.py` - renders baseline predictions onto video.

Baseline outputs are stored under:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/Baselines data
```

Treat this as durable data, not scratch output.

## Annotation Corrections

Script:

```text
tools for scaling species/tmp_species_annotation_corrections_automation.py
```

This is a manual merge tool. Edit the hardcoded paths at the top of the script:

- `VIDEO_PATH`
- `ORIGINAL_ANNOTATIONS_CSV_PATH`
- `CORRECTED_ANNOTATIONS_CSV_PATH`

It backs up CSVs, appends corrections, sorts, exact-dedupes, same-frame distance-dedupes, and writes a JSON report.

## Adding a New Species

High-level process:

1. Add raw videos and annotator CSVs under a route-prefixed folder: `day_<name>` or `night_<name>`.
2. Ensure CSVs use `x,y,w,h,frame` or another schema the ingestor can normalize.
3. Update or regenerate the raw catalog so training and inference clips are explicit.
4. Run the patch ingestor with `--dry-run`.
5. Run the real ingestor after confirming planned files.
6. Train or update patch classifiers if needed.
7. For day species, generate long exposures and ingest YOLO labels if the YOLO model needs coverage.
8. Run gateway inference on held-out clips.
9. Compare against validation outputs and baselines.
10. Export durable models/manifests only after validating results.

## Current Route Species

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
