# Add or Scale a Species Runbook

Use this checklist when adding a new species or adding more videos to an existing species.

## 1. Decide the Route

Choose exactly one:

- day route,
- night route.

Use day route when long-exposure YOLO detection is required because the background is bright or structured.

Use night route when flashes are bright blobs against a dark background.

## 2. Add Raw Files

Create or use a route-prefixed folder:

```text
day_<species display name>
night_<species display name>
```

Add:

```text
<video>.mp4
<matching_annotation>.csv
```

If using separate class CSVs:

```text
<stem>_firefly.csv
<stem>_background.csv
```

If using one unlabeled CSV, it is treated as firefly annotations.

## 3. Verify Annotation Semantics

Before ingestion or validation, answer:

- Are `x,y` top-left coordinates or center coordinates?
- Is `frame` a zero-based integer, raw frame filename, or clipped-frame index?
- Are all rows within video frame bounds?
- Are boxes roughly 10x10 or variable size?

Render a few annotations manually or through the pipeline overlays. Do not assume semantics from column names alone.

## 4. Update Route Metadata

If this is a new species, update route metadata in:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

Specifically:

```python
ROUTE_BY_SPECIES
```

Then update any relevant species switches:

- global inference switches,
- leaveout inference switches,
- baseline switches,
- day YOLO training switches for day species.

## 5. Update or Regenerate Catalog

Run the combo runner dry run:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Inspect:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/tmp_scaling_species_training_inference_catalog.json
```

Confirm each new video has:

```text
species_name
route
category
```

Category should be:

- `training` for videos intended for dataset ingestion,
- `inference` for held-out evaluation.

## 6. Ingest Patch Training Data

Dry run the ingestor:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" \
  --only-species "<raw folder name>" \
  --dry-run
```

Confirm:

- expected number of discovered pairs,
- expected training count from the catalog,
- expected dataset version paths,
- no missing catalog references,
- species token is correct and contains no underscores.

Run real ingestion:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" \
  --only-species "<raw folder name>"
```

Inspect:

```text
single species datasets/<species>/<version>
integrated pipeline datasets/<version>
codex_change_log.jsonl
```

## 7. Update Day YOLO Data If Route Is Day

For a day species, patch data alone is not enough. The day pipeline also needs YOLO coverage on long-exposure images.

Generate long exposures:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py" --dry-run
```

Label the generated images externally.

Ingest labels:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py" \
  "/path/to/yolo/export_or_zip" \
  --species-tag "<species-token>" \
  --dry-run
```

Then run real ingest after validation.

## 8. Train or Select Models

Patch models:

- global route model if expanding the production route model,
- leaveout model if evaluating generalization to the species.

Day YOLO:

- global model if adding coverage generally,
- leaveout YOLO model if doing leaveout day inference.

Always keep model manifests with promoted checkpoints.

## 9. Run Inference

Use the combo runner for catalog-selected held-out evaluation:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Then run real evaluation after switches are correct.

For quick single-video testing, use the gateway:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/video.mp4" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --route-override day \
  --force-tests \
  --max-concurrent 1
```

Change `--route-override night` for night species.

## 10. Review Outputs

For day:

- Stage 2 annotated long exposures.
- Stage 3 crops.
- Stage 3.1 trajectory SVG/video.
- Stage 3.2 `x,y,t`.
- Stage 5 validation.
- Stage 6 overlays.

For night:

- Stage 1/4/8 row counts.
- Stage 6 fixed boxes.
- Stage 8 crops.
- Stage 8.6/8.7 artifacts.
- Stage 9 validation.
- Stage 10 overlays.

For scaling run:

- `final_results.csv`
- `run_results_manifest.json`
- `run_metadata.json`
- gateway logs
- baseline `results.json`

## 11. Promote Artifacts

Promote only after visual and metric review:

- patch model `best.pt` plus `training_manifest.json`,
- day YOLO `best_firefly_yolo.pt` plus manifest,
- updated catalog,
- updated baseline registry if baselines were run.

Do not promote a checkpoint without the manifest and exact data-version context.

## 12. Handoff Notes to Capture

For every added species, record:

- route,
- raw folder path,
- catalog category split,
- dataset version created,
- model paths used,
- YOLO checkpoint used if day route,
- validation thresholds,
- known GT offset or coordinate quirks,
- baseline result locations,
- any manually corrected annotations.

