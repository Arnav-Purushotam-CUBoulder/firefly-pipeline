# Repository Map

## Top Level

```text
Pipelines/
  Pipeline Gateway/
  day time pipeline v3 (yolo + patch classifier ensemble)/
  night_time_pipeline/
tools for scaling species/
  YOLO model dataset creation, ingestion, and training scripts/
  legacy_baselines/
tools (other)/
video_rendering_defaults.py
```

## Core Application Code

### Day pipeline

Path:

```text
Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)
```

This is the day-time detector. It generates long-exposure images, runs a YOLO model, expands detections back over frames, filters with a patch classifier, selects flash-like trajectories, exports centroids/logits, renders videos, and optionally validates against ground truth.

Important files:

- `params.py` - central configuration and default paths.
- `orchestrator.py` - stage runner.
- `stage0_cleanup.py` through `stage9_detection_summary.py` - pipeline stages and test-suite stages.

### Night pipeline

Path:

```text
Pipelines/night_time_pipeline
```

This is the night-time detector. It starts from bright blob detection, recenters detections, filters by area and CNN classification, merges duplicates, applies centroid refinement, runs repair passes, renders outputs, and validates against ground truth.

Important files:

- `pipeline_params.py` - central configuration and default paths.
- `orchestrator.py` - stage runner.
- `stage1_detect_*.py` - alternative Stage 1 detectors.
- `stage8_*` files - centroiding, neighbor hunt, large-flash repair, GT recentering.
- `stage9_validate.py`, `stage10_overlay_gt_vs_model.py`, `stage11_fn_analysis.py`, `stage12_fp_analysis.py`, `stage14_detection_summary.py` - validation and analysis.

### Pipeline Gateway

Path:

```text
Pipelines/Pipeline Gateway/gateway.py
```

The gateway accepts a video or folder, decides whether each video should run through the day or night route, creates route-specific output roots, mutates pipeline configuration inside subprocesses, and launches the appropriate orchestrator.

Use this when processing mixed folders or when you want one command to run a single video without manually moving files into pipeline `ROOT/original videos`.

## Scaling and Research Tooling

### `tools for scaling species`

This folder contains the scaling system around the two core pipelines.

Main files:

- `patch_classification_models_dataset_ingestor.py` - ingests raw annotated videos into patch-classification datasets.
- `tmp_day_night_combo_train_and_infer.py` - main experiment runner for global/leaveout models, gateway inference, baselines, and result manifests.
- `tmp_species_annotation_corrections_automation.py` - merge corrected annotations back into raw annotation CSVs.

Subfolders:

- `YOLO model dataset creation, ingestion, and training scripts/` - day-time YOLO long-exposure generation and YOLO dataset ingestion.
- `legacy_baselines/` - Nolan/lab baseline, Raphael baseline, baseline validation, and baseline rendering.

The script names still say `tmp` in several places, but this folder is the current scaling layer.

### `tools (other)`

Use this only when needed.

Important files:

- `firefly flash annotation tool v2.py` - Tkinter annotation GUI. Writes CSVs with `x,y,w,h,frame`.
- `general file mover (src to dest).py` - manual file moving helper.
- `count_files_in_folder.py` - simple utility.
- `nolan's val code/val.py` - older exploratory validation code with hardcoded paths.

## Shared Defaults

`video_rendering_defaults.py` centralizes rendering defaults used by both pipelines:

- `VIDEO_RENDER_BBOX_THICKNESS_PX = 1`
- `ENFORCE_SOURCE_VIDEO_RESOLUTION = True`

This keeps rendered videos consistent across day, night, gateway, and baseline outputs.

## Configuration Style

Most code uses editable module-level constants rather than config files. The normal pattern is:

1. Edit `params.py`, `pipeline_params.py`, or the top of a scaling script.
2. Run the orchestrator or script with `python3`.
3. For gateway runs, pass overrides such as model paths, route, output root, and frame cap on the command line.

Many paths are absolute. Treat path changes as operational changes, not simple refactors.

