# Baselines and Evaluation

Baseline code:

```text
tools for scaling species/legacy_baselines
```

Permanent baseline data root:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/Baselines data
```

The baselines are independent of pipeline inference toggles. The combo runner can run baselines even if day/night pipeline inference is disabled.

## Baseline Methods

### Lab/Nolan baseline

Script:

```text
legacy_baselines/nolan_mp4_to_predcsv.py
```

Purpose:

- Uses green-channel rolling background subtraction.
- Applies blur/threshold/connected components.
- Recenters detections by intensity.
- Writes Stage-5-style prediction CSV.

Key flags:

- `--video`
- `--out-csv`
- `--threshold`
- `--blur-sigma`
- `--bkgr-window-sec`
- `--max-frames`

Combo defaults:

```python
LAB_BASELINE_THRESHOLD = 0.12
LAB_BASELINE_BLUR_SIGMA = 1.0
LAB_BASELINE_BKGR_WINDOW_SEC = 2.0
```

### Raphael baseline

Script:

```text
legacy_baselines/raphael_oorb_detect_and_gauss.py
```

Purpose:

- Python port of Raphael OOrb detection.
- Uses EWMA background.
- Uses connected components and TorchScript ffnet classification.
- Optionally applies Gaussian/intensity centroid refinement.
- Writes Stage-5-style prediction CSV.

Model:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/Baselines data/Raphael's model/ffnet_best.pth
```

Combo defaults:

```python
RAPHAEL_BW_THR = 0.2
RAPHAEL_CLASSIFY_THR = 0.98
RAPHAEL_BKGR_WINDOW_SEC = 2.0
RAPHAEL_PATCH_SIZE_PX = 33
RAPHAEL_BATCH_SIZE = 1000
RAPHAEL_GAUSS_CROP_SIZE = 10
RAPHAEL_DEVICE = "auto"
```

## Baseline Validation

Script:

```text
legacy_baselines/match_predictions_to_processed_gt.py
```

Purpose:

- Matches baseline predictions against normalized GT.
- Writes `fps.csv`, `tps.csv`, and `fns.csv` into threshold folders.

Combo default:

```python
BASELINE_DIST_THRESHOLDS_PX = [10.0]
BASELINE_VALIDATE_CROP_W = 10
BASELINE_VALIDATE_CROP_H = 10
```

## Baseline Rendering

Script:

```text
legacy_baselines/render_baseline_predictions.py
```

Purpose:

- Draws baseline predictions onto source video.

Combo output filename:

```python
BASELINE_RENDERED_VIDEO_FILENAME = "predictions_overlay.mp4"
```

## Baseline GT Processing

The combo runner normalizes GT internally using route-aware validator logic:

- day GT processing uses day Stage 5 helpers,
- night GT processing uses night Stage 9 helpers.

This keeps baseline scoring aligned with pipeline scoring.

Special case:

```python
BASELINE_GT_T_SHIFT_BY_VIDEO_STEM = {
    "val_9k-14k_frontalis_clip": 19,
}
```

This shifts GT later only for baseline scoring of that clip variant. It does not modify the shared raw annotation file or pipeline evaluation.

## Baseline Output Layout

Lab baseline:

```text
Baselines data/lab_baseline/<species>/
  data/
  results.json
  _logs/
```

Raphael baseline:

```text
Baselines data/Raphael's method/<species>/
  data/
  results.json
  _logs/
```

Registry:

```text
Baselines data/baseline_results_registry.json
```

Treat these as durable artifacts.

## Combo Runner Baseline Switches

```python
RUN_LAB_BASELINE = True
RUN_RAPHAEL_BASELINE = True
LAB_BASELINE_SPECIES_SWITCHES = {...}
RAPHAEL_BASELINE_SPECIES_SWITCHES = {...}
```

The species switches are independent from:

- global pipeline inference species switches,
- leaveout pipeline inference species switches,
- model training species.

## Final Result Records

The combo runner writes baseline metrics into:

```text
final_results.csv
run_results_manifest.json
run_metadata.json
```

Pipeline rows can also include baseline metrics for the same video so comparisons are visible in one row.

## Recommended Evaluation Flow

1. Ensure the raw catalog marks intended clips as inference.
2. Enable baseline species switches.
3. Run `--dry-run`.
4. Run baselines.
5. Inspect per-species `results.json`.
6. Inspect rendered prediction overlays.
7. Compare against pipeline `global` and `leaveout` rows in `final_results.csv`.

