# Raw Data and Catalog

Raw root:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)
```

This root is the source of truth for raw videos and annotation CSVs. The scaling scripts expect route-specific species folders and one or more `.mp4` plus annotation `.csv` pairs inside each folder.

## Expected Folder Naming

Automatic ingestion expects:

```text
day_<species display name>
night_<species display name>
```

Examples:

```text
day_Photinus greeni
night_Photinus carolinus
```

The route prefix matters:

- `day_` maps to day-time pipeline and day-time integrated patch dataset.
- `night_` maps to night-time pipeline and night-time integrated patch dataset.

Unprefixed raw species folders currently exist, but the main ingestor skips them unless folder naming or override logic is updated.

## Species Tokens

Scaling scripts normalize species into kebab-case tokens, for example:

```text
Photinus greeni -> photinus-greeni
Photuris bethaniensis -> photuris-bethaniensis
```

Avoid underscores in species tokens. The patch ingestor stages canonical CSV filenames that are parsed by splitting on underscores.

If needed, use `SPECIES_NAME_OVERRIDES` in:

```text
tools for scaling species/patch_classification_models_dataset_ingestor.py
```

Overrides can be keyed by full folder name or base species name.

## Raw Video and CSV Pairing

Each raw species folder should contain videos and matching annotation CSVs.

The ingestor looks for:

- one video file,
- a firefly annotation CSV,
- optionally a background annotation CSV.

CSV label detection:

- Suffix `_firefly.csv` means firefly annotations.
- Suffix `_background.csv` means background annotations.
- Unlabeled CSVs default to firefly annotations.

If multiple firefly CSVs match one video, ingestion stops instead of guessing.

## Annotation Schema

The scaling tools are built around annotation rows that can be normalized into:

```text
x,y,w,h,t
```

or:

```text
x,y,w,h,frame
```

Be careful with coordinate semantics. Some tools or docs refer to `x,y` as top-left; some downstream centroiding code treats `x,y` as the point center. Before using a new annotation source, verify by rendering a few frames.

## Root Catalog

Catalog path:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/tmp_scaling_species_training_inference_catalog.json
```

The catalog records:

- raw videos root,
- sources,
- summary counts,
- per-species training/inference lists,
- per-video route/species/category records.

Important keys:

```text
generated_at
raw_videos_root
catalog_path
sources
summary
by_species
videos
```

## Why the Catalog Matters

The catalog prevents accidental mixing of:

- training clips,
- held-out inference clips,
- species routes,
- legacy source clips,
- baseline-only evaluation videos.

Patch ingestion uses the catalog when:

```python
USE_ROOT_VIDEO_CATALOG_IF_PRESENT = True
INGEST_TRAINING_VIDEOS_FROM_ROOT_CATALOG = True
```

In this mode, only catalog `category=training` videos are ingested into training datasets.

Gateway inference and baselines use catalog `category=inference` videos.

## Current Catalog Summary

Current catalog totals:

- 46 total videos.
- 31 training videos.
- 15 inference videos.
- Day: 19 training, 8 inference.
- Night: 12 training, 7 inference.

Current species counts:

| Species | Route | Training | Inference |
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

## Catalog Generation

The combo runner builds and writes the catalog:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

Relevant functions:

- `_discover_training_sources_by_route`
- `_collect_training_video_stems_for_source`
- `_build_training_inference_catalog`
- `_write_training_inference_catalog`

The catalog is built by comparing:

- training video stems present in integrated patch datasets,
- raw videos under the raw root,
- route/species inference rules.

## Practical Rule

Do not ingest or evaluate a new raw video until it appears in the catalog with the expected:

```text
species_name
route
category
```

If the catalog is wrong, fix the catalog/source mapping first. Do not work around it by manually copying clips into dataset folders.

