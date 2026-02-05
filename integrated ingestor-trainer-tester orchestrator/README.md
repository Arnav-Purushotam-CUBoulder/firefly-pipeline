# Integrated ingestor → trainer → tester (gateway) orchestrator

This folder contains the “automation orchestrator” that ties together:
- `test1/species scaler/species_scaler.py` (ingest new annotation batches + version datasets)
- `test1/tools/model training tools/training resnet18 model.py` (train CNN/patch classifier)
- `test1/integrated pipeline/gateway.py` (route videos to day/night pipelines + run validation)

## Local scaffold created here

For convenience, this repo includes a **local** scaffold you can point the orchestrator at:
- `model zoo/` (models + run history)
- `inference outputs/` (pipeline outputs per run)

Note: the repo’s `test1/.gitignore` ignores `.pt`, `.json`, `.jsonl`, `.csv`, images, and videos, so trained models and run logs will remain local-only by default.

## Recommended usage (single root)

Pass a single `--root` path. The orchestrator will create/use:

- `<root>/patch training datasets and pipeline validation data/` (datasets + validation CSV versions)
- `<root>/model zoo/` (trained models + `results_history.jsonl` + `video_registry.json`)
- `<root>/inference outputs/` (per-run gateway/pipeline outputs)

You also pass an `--observed-dir` folder that contains **multiple** videos and annotator CSVs. The orchestrator:

- matches `*.csv` → `*.mp4` by filename prefix (CSV stem must start with the video stem)
- splits matched video/csv pairs **by video**:
  - first half → training ingestion
  - second half → validation/testing (if odd, validation gets the larger half)
- uses `--type-of-video day|night` (not filenames) for dataset routing + model-zoo selection

CSV naming expectation (flexible):

- Firefly CSV (required per video):
  - `<video_stem>.csv` (defaults to firefly), or
  - `<video_stem>_firefly.csv`
- Background CSV (optional, used only for training videos):
  - `<video_stem>_background.csv`

Any trailing `*_day_time` / `*_night_time` tokens in CSV filenames are ignored (the run uses `--type-of-video`).

Any species tokens in CSV filenames are ignored — pass the species name explicitly via `--species-name` (or set `SPECIES_NAME` at the top of the orchestrator file).

Example:

```bash
python3 "test1/integrated ingestor-trainer-tester orchestrator/orchestrator.py" \
  --root "/Volumes/DL Project SSD/integrated prototype data" \
  --species-name "Photinus_pyralis" \
  --observed-dir "/path/to/observed folder" \
  --type-of-video night
```

Each run creates a new folder under `<root>/inference outputs/` with a timestamp + batch details in the name.
