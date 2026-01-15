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

