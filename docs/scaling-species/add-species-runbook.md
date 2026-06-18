# Add or Scale a Species Runbook

This is the practical, exact runbook for taking a new species from annotator output to scaled training/evaluation.

The example species used throughout is fictitious:

```text
Photinus exampleii
```

Example route:

```text
day
```

Canonical species token:

```text
photinus-exampleii
```

Raw folder:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/day_Photinus exampleii
```

Replace these names with the real species when you do the actual work.

## What You Edit

For a new species, you normally edit these files:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

Required for route registration, train/infer switches, baselines, and model-training toggles.

```text
tools for scaling species/patch_classification_models_dataset_ingestor.py
```

Only required if you need a species-name override or need to bootstrap first ingestion by temporarily bypassing the root catalog.

For a day species, you may also edit:

```text
tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py
```

Required when generating long-exposure images for a specific new day species.

For day YOLO label ingestion, you usually do not need to edit the file if you pass `--species-tag`, but you may edit aliases if the folder naming does not normalize correctly:

```text
tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py
```

## Step 1: Decide the Route

Choose exactly one route.

Use day route when:

- videos have bright or structured backgrounds,
- fireflies are not simple isolated bright blobs,
- the day pipeline's long-exposure YOLO stage is needed.

Use night route when:

- flashes are bright blobs against a dark background,
- per-frame blob detection is a good first candidate generator.

For this example:

```text
Photinus exampleii -> day
```

## Step 2: Create the Raw Annotator Folder

Create:

```bash
mkdir -p "/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/day_Photinus exampleii"
```

Put videos and CSVs inside:

```text
day_Photinus exampleii/
  PE_001.mp4
  PE_001.csv
  PE_002.mp4
  PE_002.csv
  PE_003.mp4
  PE_003.csv
  PE_004.mp4
  PE_004.csv
  PE_005.mp4
  PE_005.csv
```

Supported raw folder naming:

```text
day_<species display name>
night_<species display name>
```

The route prefix is required by the patch ingestor.

CSV convention:

```text
x,y,w,h,frame
```

If you have separate class CSVs, use:

```text
PE_001_firefly.csv
PE_001_background.csv
```

If there is only one CSV per video and it has no class suffix, the ingestor treats it as firefly annotations.

## Step 3: Verify Annotation Semantics

Before training or validation, verify:

- whether `x,y` are top-left or center coordinates,
- whether `frame` is zero-based, raw frame number, or a filename containing a frame number,
- whether rows are inside video bounds,
- whether boxes are 10x10 or variable size.

This matters because downstream code does not always infer center from `w,h`. Some validation and GT-recentering paths use the provided `x,y` as the point.

Minimum check:

1. Open one CSV.
2. Pick 5 rows from different frames.
3. Open those frames in the video.
4. Confirm the annotation lands on the flash.

Do this before ingestion. A model trained on shifted patches will produce confusing results later.

## Step 4: Choose or Override the Species Token

The default token for:

```text
day_Photinus exampleii
```

is:

```text
photinus-exampleii
```

If the raw folder name should map to a different token, edit:

```text
tools for scaling species/patch_classification_models_dataset_ingestor.py
```

Add to `SPECIES_NAME_OVERRIDES`:

```python
SPECIES_NAME_OVERRIDES: Dict[str, str] = {
    "Photinus exampleii": "photinus-exampleii",
}
```

Only add an override if the automatic token is wrong.

Rules:

- use lowercase,
- use hyphens,
- do not use underscores,
- keep the same token in every scaling script.

## Step 5: Register the Species Route

Edit:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

Find `ROUTE_BY_SPECIES` and add:

```python
ROUTE_BY_SPECIES: Dict[str, str] = {
    "bicellonycha-wickershamorum": "day",
    "photinus-acuminatus": "day",
    "photinus-greeni": "day",
    "photuris-bethaniensis": "day",
    "photinus-exampleii": "day",
    "forresti": "night",
    "frontalis": "night",
    "photinus-carolinus": "night",
    "photinus-knulli": "night",
    "tremulans": "night",
}
```

This is required. It lets the combo runner infer species and route from raw paths.

If a specific video stem must override route inference, add to `ROUTE_BY_VIDEO_STEM`:

```python
ROUTE_BY_VIDEO_STEM: Dict[str, str] = {
    "PE_001": "day",
}
```

Only use this for unusual clips. Normal species route registration should go in `ROUTE_BY_SPECIES`.

## Step 6: Add the Species to Inference Switches

Still in:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

For this day species, edit `GLOBAL_DAY_INFERENCE_SPECIES_SWITCHES`:

```python
GLOBAL_DAY_INFERENCE_SPECIES_SWITCHES: Dict[str, bool] = {
    "bicellonycha-wickershamorum": True,
    "photinus-acuminatus": True,
    "photinus-greeni": True,
    "photuris-bethaniensis": True,
    "photinus-exampleii": True,
}
```

Then edit `LEAVEOUT_DAY_INFERENCE_SPECIES_SWITCHES`:

```python
LEAVEOUT_DAY_INFERENCE_SPECIES_SWITCHES: Dict[str, bool] = {
    "bicellonycha-wickershamorum": False,
    "photinus-acuminatus": False,
    "photinus-greeni": False,
    "photuris-bethaniensis": False,
    "photinus-exampleii": False,
}
```

Keep leaveout day inference off until you have both:

- leaveout day patch model,
- leaveout day YOLO model.

For a night species, you would instead add to:

```python
GLOBAL_NIGHT_INFERENCE_SPECIES_SWITCHES
LEAVEOUT_NIGHT_INFERENCE_SPECIES_SWITCHES
```

## Step 7: Add the Species to Baseline Switches

Still in:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

Add the species key to both baseline switch dictionaries:

```python
LAB_BASELINE_SPECIES_SWITCHES: Dict[str, bool] = {
    "bicellonycha-wickershamorum": False,
    "photinus-acuminatus": True,
    "photinus-greeni": False,
    "photuris-bethaniensis": False,
    "photinus-exampleii": False,
    "forresti": False,
    "frontalis": True,
    "photinus-carolinus": False,
    "photinus-knulli": False,
    "tremulans": False,
}
```

```python
RAPHAEL_BASELINE_SPECIES_SWITCHES: Dict[str, bool] = {
    "bicellonycha-wickershamorum": False,
    "photinus-acuminatus": True,
    "photinus-greeni": False,
    "photuris-bethaniensis": False,
    "photinus-exampleii": False,
    "forresti": False,
    "frontalis": True,
    "photinus-carolinus": False,
    "photinus-knulli": False,
    "tremulans": False,
}
```

Set the value to `True` only if you want that baseline to run for the new species.

## Step 8: Bootstrap the New Species Training Split

This is the most important first-time-new-species caveat.

The root catalog is:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/tmp_scaling_species_training_inference_catalog.json
```

The current catalog generator marks videos as `training` when their stems already exist in an integrated patch dataset. A brand-new species has no integrated patch dataset yet, so its videos may initially appear as `inference`.

You need one bootstrap path.

### Option A: Temporarily Bypass the Catalog for First Ingestion

This is the simplest path for the first ingest.

Edit:

```text
tools for scaling species/patch_classification_models_dataset_ingestor.py
```

Temporarily set:

```python
USE_ROOT_VIDEO_CATALOG_IF_PRESENT: bool = False
TRAIN_PAIR_FRACTION: float = 0.8
```

For the example folder, this sorted split means:

```text
training:  PE_001, PE_002, PE_003, PE_004
held out:  PE_005
```

Run the ingestor for the species only. After it succeeds, restore:

```python
USE_ROOT_VIDEO_CATALOG_IF_PRESENT: bool = True
```

### Option B: Manually Add Catalog Entries

Use this if you need exact control over the split.

Edit the root catalog JSON and add entries like:

```json
{
  "video_path": "/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/day_Photinus exampleii/PE_001.mp4",
  "video_name": "PE_001.mp4",
  "video_stem": "PE_001",
  "species_name": "photinus-exampleii",
  "route": "day",
  "category": "training"
}
```

Held-out example:

```json
{
  "video_path": "/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/day_Photinus exampleii/PE_005.mp4",
  "video_name": "PE_005.mp4",
  "video_stem": "PE_005",
  "species_name": "photinus-exampleii",
  "route": "day",
  "category": "inference"
}
```

The ingestor primarily needs the `videos` entries to choose training clips. Keep `summary` and `by_species` consistent if you manually edit them, or regenerate the catalog after first ingestion.

Important: `tmp_day_night_combo_train_and_infer.py --dry-run` does not write the catalog. It only prints/plans.

## Step 9: Dry-Run Patch Dataset Ingestion

Run:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" \
  --only-species "day_Photinus exampleii" \
  --dry-run
```

Confirm the output says:

```text
day_Photinus exampleii -> photinus-exampleii (day_time)
```

Confirm the expected video counts.

If using catalog mode, confirm it says something like:

```text
root catalog split: train=4 val=0
```

If it skips the species, common causes are:

- raw folder does not start with `day_` or `night_`,
- CSV filenames do not match video stems,
- catalog has wrong `species_name`,
- catalog has wrong `route`,
- catalog marks every video as inference,
- species was already ingested according to `codex_change_log.jsonl`.

## Step 10: Run Patch Dataset Ingestion

Run:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" \
  --only-species "day_Photinus exampleii"
```

The script creates integrated folders automatically. Do not manually create these.

Expected single-species dataset:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/patch training datasets and pipeline validation data/Integrated_prototype_datasets/single species datasets/photinus-exampleii/<version>
```

Expected integrated route dataset:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/patch training datasets and pipeline validation data/Integrated_prototype_datasets/integrated pipeline datasets/<version>/day_time_dataset
```

Check the change log:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/codex_change_log.jsonl
```

After bootstrap ingestion, restore catalog mode if you temporarily disabled it:

```python
USE_ROOT_VIDEO_CATALOG_IF_PRESENT: bool = True
```

## Step 11: Refresh and Check the Catalog

Run a combo-runner dry run to check planned species detection:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Remember: dry run does not write the catalog.

To inspect the current on-disk catalog:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/tmp_scaling_species_training_inference_catalog.json")
d = json.loads(p.read_text())

for v in d.get("videos", []):
    if v.get("species_name") == "photinus-exampleii":
        print(v["category"], v["route"], v["video_name"])
PY
```

Expected after catalog is properly refreshed:

```text
training day PE_001.mp4
training day PE_002.mp4
training day PE_003.mp4
training day PE_004.mp4
inference day PE_005.mp4
```

If all entries remain `inference`, the catalog did not see the new training stems in the integrated patch dataset.

## Step 12: Day Species Only - Generate Long-Exposure YOLO Images

Skip this section for night species.

For a day species, patch data alone is not enough. The day pipeline still needs YOLO to propose long-exposure boxes.

Edit:

```text
tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py
```

Change:

```python
OUTPUT_ROOT: Path = Path(
    "/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/generated long exposure images/day_Photinus exampleii"
)
SPECIES_TOKEN: str = "photinus-exampleii"
ROUTE_NAME: str = "day"
```

Dry run:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py" --dry-run
```

If the plan is correct, run:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py"
```

Label the generated long-exposure images externally.

## Step 13: Day Species Only - Ingest YOLO Labels

Skip this section for night species.

If your labeling tool produced a YOLO folder or zip, ingest it:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py" \
  "/path/to/yolo/export_or_zip" \
  --species-tag "photinus-exampleii" \
  --dry-run
```

If the label export contains labels only and images are elsewhere:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py" \
  "/path/to/yolo/labels_or_zip" \
  --species-tag "photinus-exampleii" \
  --local-image-root "/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/generated long exposure images/day_Photinus exampleii" \
  --dry-run
```

Then run without `--dry-run`.

Expected output:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/dataset/photinus-exampleii/images
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/dataset/photinus-exampleii/labels
```

If the species tag does not normalize correctly, edit `SPECIES_FOLDER_ALIASES` in:

```text
tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py
```

Add:

```python
SPECIES_FOLDER_ALIASES: Dict[str, str] = {
    ...
    "day-photinus-exampleii": "photinus-exampleii",
    "photinus-exampleii": "photinus-exampleii",
}
```

## Step 14: Configure Patch Model Training

Edit:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

For a day species global model update:

```python
RUN_MODEL_TRAINING: bool = True
RUN_DAY_MODEL_TRAINING: bool = True
RUN_NIGHT_MODEL_TRAINING: bool = False
TRAIN_GLOBAL_MODELS: bool = True
TRAIN_LEAVEOUT_MODELS: bool = False
```

For day global plus leaveout patch models:

```python
RUN_MODEL_TRAINING: bool = True
RUN_DAY_MODEL_TRAINING: bool = True
RUN_NIGHT_MODEL_TRAINING: bool = False
TRAIN_GLOBAL_MODELS: bool = True
TRAIN_LEAVEOUT_MODELS: bool = True
```

For a night species:

```python
RUN_MODEL_TRAINING: bool = True
RUN_DAY_MODEL_TRAINING: bool = False
RUN_NIGHT_MODEL_TRAINING: bool = True
TRAIN_GLOBAL_MODELS: bool = True
TRAIN_LEAVEOUT_MODELS: bool = True
```

Keep:

```python
TRAIN_RESNET: str = "resnet18"
```

The inference loaders expect ResNet18 checkpoints.

## Step 15: Configure Day YOLO Training

Skip this section for night species.

In:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

For global YOLO training:

```python
RUN_DAY_YOLO_MODEL_TRAINING: bool = True
TRAIN_DAY_YOLO_GLOBAL_MODEL: bool = True
TRAIN_DAY_YOLO_LEAVEOUT_MODELS: bool = False
```

For global plus leaveout YOLO training:

```python
RUN_DAY_YOLO_MODEL_TRAINING: bool = True
TRAIN_DAY_YOLO_GLOBAL_MODEL: bool = True
TRAIN_DAY_YOLO_LEAVEOUT_MODELS: bool = True
```

The species should be included automatically in:

```python
DAY_YOLO_TRAINING_SPECIES_SWITCHES
```

because it is built from `DAY_ROUTE_SPECIES`, which comes from `ROUTE_BY_SPECIES`.

If the YOLO dataset folder uses a different name, add an alias in `DAY_YOLO_SPECIES_ALIASES`:

```python
DAY_YOLO_SPECIES_ALIASES: Dict[str, str] = {
    ...
    "photinus-exampleii": "photinus-exampleii",
}
```

## Step 16: Configure Pipeline Inference

Still in:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

For day global inference:

```python
RUN_DAY_PIPELINE_INFERENCE: bool = True
RUN_NIGHT_PIPELINE_INFERENCE: bool = False
RUN_GLOBAL_MODEL_INFERENCE: bool = True
RUN_LEAVEOUT_MODEL_INFERENCE: bool = False
```

For day leaveout inference, only after leaveout patch and leaveout YOLO models exist:

```python
RUN_DAY_PIPELINE_INFERENCE: bool = True
RUN_GLOBAL_MODEL_INFERENCE: bool = True
RUN_LEAVEOUT_MODEL_INFERENCE: bool = True

LEAVEOUT_DAY_INFERENCE_SPECIES_SWITCHES["photinus-exampleii"] = True
```

For night global/leaveout inference:

```python
RUN_DAY_PIPELINE_INFERENCE: bool = False
RUN_NIGHT_PIPELINE_INFERENCE: bool = True
RUN_GLOBAL_MODEL_INFERENCE: bool = True
RUN_LEAVEOUT_MODEL_INFERENCE: bool = True
```

## Step 17: Dry-Run the Combo Runner

Run:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Check:

- new species appears under the correct route,
- selected inference videos are held-out videos only,
- model training toggles match your intent,
- day YOLO training is enabled only if you want it,
- baseline species selection is intentional,
- leaveout day inference is not enabled without leaveout day YOLO models.

## Step 18: Run Training and Evaluation

When the dry run is correct:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py"
```

Outputs go under:

```text
/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/<run_id>
```

Key files:

```text
final_results.csv
run_results_manifest.json
run_metadata.json
logs/
```

Model zoo outputs, if training/export happens:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/model zoo/global models/daytime global/best.pt
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/model zoo/global models/daytime global/training_manifest.json
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/model zoo/leave out models/photinus-exampleii/best.pt
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/model zoo/leave out models/photinus-exampleii/training_manifest.json
```

Day YOLO outputs, if enabled:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/models/global models/<date>/best_firefly_yolo.pt
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/models/leave one out models/<date>/photinus-exampleii/best_firefly_yolo.pt
```

## Step 19: Review Outputs

For day route, inspect:

```text
stage1_long_exposure/
stage2_yolo_detections/
stage3_patch_classifier/
stage3_2 xyt for 3d reconstruction/
stage4_rendering/
stage5 validation/
stage6 overlay videos/
```

Review order:

1. Are flashes visible in Stage 1 long exposures?
2. Did YOLO find the right regions in Stage 2?
3. Are Stage 3 positive crops real flashes?
4. Did Stage 3.1 reject real flashes?
5. Are Stage 3.2 centroids plausible?
6. Do Stage 6 overlays align with GT?

For night route, inspect:

```text
csv files/
original 10px overlay annotated videos/
stage8 crops/
stage8.6/
stage8.7/
stage9 validation/
stage10 overlay videos/
stage14 detection summaries/
```

Review order:

1. Did Stage 1 find candidates?
2. Did Stage 4 classify true flashes as firefly?
3. Did Stage 8 center detections correctly?
4. Did Stage 8.6 add valid neighbors or noise?
5. Did Stage 8.7 repair large flashes correctly?
6. Does Stage 10 overlay align with GT?

## Step 20: Promote or Reject Artifacts

Promote only after both metrics and overlays look correct.

Promote:

- patch model `best.pt`,
- patch model `training_manifest.json`,
- day YOLO `best_firefly_yolo.pt` if day route,
- day YOLO manifest if available,
- updated raw catalog,
- run metadata and result manifest,
- baseline registry if baselines were run.

Do not promote:

- checkpoints without manifests,
- models trained with possible train/inference leakage,
- models with good aggregate metrics but bad overlays,
- models whose exact data version is unclear.

## Step 21: Handoff Notes to Capture

For the species, write down:

- species display name,
- species token,
- route,
- raw folder path,
- raw video names,
- training vs inference split,
- annotation coordinate semantics,
- dataset versions created,
- patch model paths,
- YOLO dataset/model paths if day route,
- combo-runner toggles used,
- validation thresholds,
- known GT offset quirks,
- baseline paths if run,
- issues found in overlays.

## Night Species Example Differences

If the new species were night route instead:

```text
night_Photinus exampleii
photinus-exampleii
```

Use:

```python
ROUTE_BY_SPECIES["photinus-exampleii"] = "night"
```

Add to:

```python
GLOBAL_NIGHT_INFERENCE_SPECIES_SWITCHES
LEAVEOUT_NIGHT_INFERENCE_SPECIES_SWITCHES
LAB_BASELINE_SPECIES_SWITCHES
RAPHAEL_BASELINE_SPECIES_SWITCHES
```

Do not run day YOLO steps.

Use training toggles:

```python
RUN_MODEL_TRAINING: bool = True
RUN_DAY_MODEL_TRAINING: bool = False
RUN_NIGHT_MODEL_TRAINING: bool = True
TRAIN_GLOBAL_MODELS: bool = True
TRAIN_LEAVEOUT_MODELS: bool = True
```

Use inference toggles:

```python
RUN_DAY_PIPELINE_INFERENCE: bool = False
RUN_NIGHT_PIPELINE_INFERENCE: bool = True
RUN_GLOBAL_MODEL_INFERENCE: bool = True
RUN_LEAVEOUT_MODEL_INFERENCE: bool = True
```

Night species do not need:

- long-exposure generation,
- day YOLO label ingestion,
- day YOLO training,
- leaveout day YOLO checkpoints.

## Common Failure Modes

### The raw species folder is skipped

Likely causes:

- folder does not start with `day_` or `night_`,
- folder is empty,
- CSV/video stems do not match,
- multiple firefly CSVs match one video.

### The ingestor says catalog failed

Likely causes:

- new species not in catalog,
- catalog species token does not match,
- catalog route is wrong,
- catalog references missing video stems,
- catalog marks all new videos as inference.

For first ingest, use the bootstrap flow in Step 8.

### Combo runner does not find the species

Likely causes:

- missing `ROUTE_BY_SPECIES` entry,
- species token typo,
- raw folder name does not slug-match token,
- inference species switch missing or false.

### Day pipeline inference runs but misses everything

Likely causes:

- patch classifier trained but day YOLO was not updated,
- YOLO dataset has no new species coverage,
- wrong YOLO checkpoint selected,
- Stage 2 confidence threshold too high.

### Leaveout day inference fails

Likely cause:

- leaveout patch model exists but leaveout day YOLO model does not.

Fix:

- train leaveout day YOLO, or keep leaveout day inference disabled.

### Validation is shifted

Likely causes:

- wrong `GT_T_OFFSET`,
- annotation `x,y` semantics wrong,
- `frame` values are raw frame numbers but treated as clipped frame numbers,
- GT conversion/recentering was applied incorrectly.

Always inspect overlays before trusting metrics.

