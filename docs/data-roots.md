# Data Roots and Persistent Artifacts

The repo code is small compared with the external SSD data. The takeover should treat the SSD layout as part of the system.

## Raw Videos and Annotations

Root:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)
```

This root contains species folders with source videos and sibling annotation CSVs.

Current top-level species/data folders include:

```text
Photinus marginellus
Photinus obscurellus
Photinus pyralis
Pyractomena angulata
day_Bicellonycha wickershamorum
day_Photinus acuminatus
day_Photinus greeni
day_Photuris bethaniensis
night_Forresti
night_Frontalis
night_Photinus Knulli
night_Photinus carolinus
night_Tremulans
riverbend Photuris species
```

The scaling ingestor currently expects route-prefixed raw folders:

- `day_<species display name>`
- `night_<species display name>`

Unprefixed folders are not part of the current automatic ingestion route unless code or folder naming is updated.

## Raw Catalog

Catalog:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/tmp_scaling_species_training_inference_catalog.json
```

This catalog connects raw videos to species, route, and category. It is the source of truth for which clips are training clips and which clips are held-out inference clips.

Current catalog summary:

| Species | Route | Training videos | Inference videos |
| --- | --- | ---: | ---: |
| `bicellonycha-wickershamorum` | day | 8 | 2 |
| `photinus-acuminatus` | day | 3 | 1 |
| `photinus-greeni` | day | 4 | 4 |
| `photuris-bethaniensis` | day | 4 | 1 |
| `photinus-carolinus` | night | 7 | 2 |
| `photinus-knulli` | night | 5 | 2 |
| `forresti` | night | 0 | 1 |
| `frontalis` | night | 0 | 1 |
| `tremulans` | night | 0 | 1 |

Overall summary:

- 46 total videos.
- 31 training videos.
- 15 inference videos.
- Day: 19 training, 8 inference.
- Night: 12 training, 7 inference.

## Integrated Prototype Data

Root:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data
```

Important subtrees:

```text
Baselines data/
integrated prototype data/
v3 daytime YOLO model data/
```

### Baselines

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/Baselines data
```

Contains:

- `lab_baseline/` - permanent outputs from the Nolan/lab baseline.
- `Raphael's method/` - permanent outputs from Raphael baseline runs.
- `Raphael's model/ffnet_best.pth` - model used by the Raphael baseline.
- `baseline_results_registry.json` - registry of baseline results.
- `photinus-knulli comparison videos/` - comparison videos.

These are not temporary per-run outputs. Be careful before deleting or overwriting.

### Inner Integrated Root

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data
```

Contains:

- `model zoo/`
- `patch training datasets and pipeline validation data/`

The repeated folder name is intentional in the current scripts.

### Patch Model Zoo

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/model zoo
```

Expected layout:

```text
global models/
  daytime global/
    best.pt
    training_manifest.json
  night time global/
    best.pt
    training_manifest.json
leave out models/
  <species>/
    best.pt
    training_manifest.json
single species models/
```

The combo runner can train/export to this zoo, or use the existing global and leaveout models for inference.

### Patch Training Datasets

Root:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/patch training datasets and pipeline validation data/Integrated_prototype_datasets
```

Expected layout:

```text
single species datasets/
  <species>/
    <version>/
      initial dataset/
      final dataset/
integrated pipeline datasets/
  <version>/
    day_time_dataset/
    night_time_dataset/
```

The ingestor writes firefly/background patches and patch-location CSVs here.

### Daytime YOLO Data

Root:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data
```

Important subtrees:

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

The current day pipeline default YOLO checkpoint points at:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/models/global models/20260414/best_firefly_yolo.pt
```

## Temporary Outputs

Most direct pipeline and experiment outputs live under:

```text
/mnt/Samsung_SSD_2TB/temp to delete
```

Examples:

- `day time pipeline inference output data`
- `night time pipeline inference output data`
- `pipeline gateway inference output data`
- `firefly_patch_training_local_run`

The name means outputs are disposable, but still inspect before deleting if an experiment is in progress.

## Common CSV Schemas

Annotator CSV:

```text
x,y,w,h,frame
```

Normalized GT CSV:

```text
x,y,t
```

Day Stage 3 patch predictions:

```text
frame_idx,video_name,x,y,w,h,conf,det_id
```

Day Stage 3.2 reconstruction export:

```text
x,y,t
```

Night final CSV after Stage 8:

```text
frame,x,y,w,h,...,xy_semantics
```

For night after Stage 8, `x,y` are center coordinates and `xy_semantics` should be `center`.

