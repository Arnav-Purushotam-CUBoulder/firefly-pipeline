# Firefly Pipeline

This repository contains the application pipelines and scaling tools for firefly flash detection across day-time and night-time videos.

The important mental model is:

1. `Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)` is the day-time application pipeline.
2. `Pipelines/night_time_pipeline` is the night-time application pipeline.
3. `Pipelines/Pipeline Gateway/gateway.py` is the router that runs the correct pipeline for one video or a folder of videos.
4. `tools for scaling species` contains the dataset ingestion, model training, baseline, and experiment tooling needed to scale the system to many species.
5. `tools (other)` is a grab bag of on-demand utilities, including the manual annotation GUI.

Most scripts currently assume the Samsung SSD is mounted at `/mnt/Samsung_SSD_2TB`. See [docs/data-roots.md](docs/data-roots.md) before moving data or changing paths.

## Start Here

- New takeover overview: [docs/README.md](docs/README.md)
- Repository map: [docs/repository-map.md](docs/repository-map.md)
- Environment and dependencies: [docs/environment.md](docs/environment.md)
- External SSD data layout: [docs/data-roots.md](docs/data-roots.md)
- Day pipeline overview: [docs/day-pipeline.md](docs/day-pipeline.md)
- Day pipeline deep docs: [docs/day-pipeline/README.md](docs/day-pipeline/README.md)
- Night pipeline overview: [docs/night-pipeline.md](docs/night-pipeline.md)
- Night pipeline deep docs: [docs/night-pipeline/README.md](docs/night-pipeline/README.md)
- Gateway runner: [docs/gateway.md](docs/gateway.md)
- Species scaling overview: [docs/scaling-species.md](docs/scaling-species.md)
- Species scaling deep docs: [docs/scaling-species/README.md](docs/scaling-species/README.md)
- Validation and output schemas: [docs/validation-and-outputs.md](docs/validation-and-outputs.md)
- Operations runbook: [docs/operations-runbook.md](docs/operations-runbook.md)

## Quick Commands

Run the day pipeline directly after putting input videos under the configured `ROOT/original videos` folder:

```bash
python3 "Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/orchestrator.py"
```

Run the night pipeline directly after putting input videos under the configured `ROOT/original videos` folder:

```bash
python3 "Pipelines/night_time_pipeline/orchestrator.py"
```

Run a single video through the gateway with an explicit route:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/video.mp4" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --route-override day \
  --force-tests \
  --max-concurrent 1
```

Plan a scaling run without training or inference side effects:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Plan patch dataset ingestion:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" --dry-run
```

## Safety Notes

- The day and night pipelines both have pre-run cleanup enabled by default. Cleanup deletes generated outputs under the configured `ROOT`.
- Do not point `ROOT` at a raw-data or model-zoo folder.
- The night cleanup stage keeps only `ground truth` and `original videos`, and has special behavior around `ground truth/gt.csv`.
- Many paths contain spaces and apostrophes. Quote paths in shell commands.
- Day and night stage numbers do not match. Day validation is Stage 5; night validation is Stage 9.
- Coordinate semantics change by stage. Raw annotations are usually top-left boxes; final centroid exports are center `x,y,t`.

## Core External Roots

Raw videos and annotator CSVs:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)
```

Integrated datasets, model zoo, YOLO data, and baselines:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data
```

Temporary run outputs:

```text
/mnt/Samsung_SSD_2TB/temp to delete
```
