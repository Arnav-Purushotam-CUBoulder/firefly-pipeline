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

Instead of passing `--data-root`, `--model-zoo-root`, and `--inference-output-root` separately, you can pass a
single `--root` path. The orchestrator will create/use:

- `<root>/patch training datasets and pipeline validation data/` (datasets + validation CSV versions)
- `<root>/model zoo/` (trained models + `results_history.jsonl` + `video_registry.json`)
- `<root>/inference outputs/` (per-run gateway/pipeline outputs)

Example:

```bash
python3 "test1/integrated ingestor-trainer-tester orchestrator/orchestrator.py" \
  --annotations-csv "/path/to/<video>_<species>_<day_time|night_time>_firefly.csv" \
  --video "/path/to/video.mp4" \
  --root "/Volumes/DL Project SSD/integrated prototype data" \
  --train-fraction 0.8
```

Each run creates a new folder under `<root>/inference outputs/` with a timestamp + batch details in the name.
