# Worked Example: Add a New Day Species

This is a concrete example of adding a new species from annotator output and scaling it through dataset ingestion, model training, and inference.

Example species:

```text
Photinus exampleii
```

Route:

```text
day
```

Canonical species token:

```text
photinus-exampleii
```

Raw species folder:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/day_Photinus exampleii
```

Use this as a template for a real species. Replace the example names and paths with the actual annotator folder and species token.

## Goal

By the end, you should have:

1. Raw videos and annotator CSVs in the correct raw-data folder.
2. The species registered in the scaling runner.
3. Training/inference split bootstrapped into the raw catalog.
4. Patch-classification datasets created under the integrated data root.
5. Day YOLO data added if this day species needs YOLO coverage.
6. Global and optionally leaveout models trained.
7. Gateway inference and validation run on held-out videos.
8. Results written into a scaling run folder.

## Important Paths

Raw annotator data root:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)
```

Integrated data root:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data
```

Patch dataset root:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/patch training datasets and pipeline validation data/Integrated_prototype_datasets
```

Patch model zoo:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/model zoo
```

Day YOLO data root:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data
```

Temporary scaling runs:

```text
/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run
```

## Step 1: Create the Raw Species Folder

Create:

```bash
mkdir -p "/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/day_Photinus exampleii"
```

Put annotator output inside:

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

The CSVs should be compatible with the ingestor/validators, usually:

```text
x,y,w,h,frame
```

Before ingesting, verify the annotation coordinate semantics:

- Is `x,y` the box top-left?
- Is `x,y` the box center?
- Is `frame` a zero-based integer, raw frame filename, or raw frame number?

Do not skip this. Bad coordinate assumptions create confusing validation failures later.

## Step 2: Register the Species in the Combo Runner

Edit:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

Add the species to `ROUTE_BY_SPECIES`:

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

Then add it to the day inference switches.

For global day inference:

```python
GLOBAL_DAY_INFERENCE_SPECIES_SWITCHES: Dict[str, bool] = {
    "bicellonycha-wickershamorum": True,
    "photinus-acuminatus": True,
    "photinus-greeni": True,
    "photuris-bethaniensis": True,
    "photinus-exampleii": True,
}
```

For leaveout day inference:

```python
LEAVEOUT_DAY_INFERENCE_SPECIES_SWITCHES: Dict[str, bool] = {
    "bicellonycha-wickershamorum": False,
    "photinus-acuminatus": False,
    "photinus-greeni": False,
    "photuris-bethaniensis": False,
    "photinus-exampleii": False,
}
```

Keep leaveout day inference `False` until you also have leaveout day YOLO checkpoints. Day leaveout inference needs both:

- leaveout patch classifier,
- leaveout day YOLO model.

If you want baselines for this species, add keys here too:

```python
LAB_BASELINE_SPECIES_SWITCHES: Dict[str, bool] = {
    ...
    "photinus-exampleii": False,
}

RAPHAEL_BASELINE_SPECIES_SWITCHES: Dict[str, bool] = {
    ...
    "photinus-exampleii": False,
}
```

Set those to `True` only when you intentionally want to run those baselines.

## Step 3: Bootstrap the Training/Inference Split

For an existing species, the raw catalog already knows which videos are training and which are inference.

For a brand-new species, the catalog generator may initially mark every raw video as `inference`, because no patch dataset exists yet for that species. You need to bootstrap the split once.

Catalog path:

```text
/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)/tmp_scaling_species_training_inference_catalog.json
```

There are two workable bootstrap methods.

### Bootstrap Method A: Temporarily Bypass the Catalog

This is the simplest current-code path.

Temporarily edit:

```text
tools for scaling species/patch_classification_models_dataset_ingestor.py
```

Set:

```python
USE_ROOT_VIDEO_CATALOG_IF_PRESENT = False
TRAIN_PAIR_FRACTION = 0.8
```

Then the ingestor uses a deterministic sorted 80/20 split for that raw folder. It ingests the training side only and leaves the rest held out.

After the first successful ingestion, restore:

```python
USE_ROOT_VIDEO_CATALOG_IF_PRESENT = True
```

Then the integrated patch dataset contains training stems for the new species. The next real combo-runner execution can rebuild the catalog with those stems marked as `training`.

Important: `tmp_day_night_combo_train_and_infer.py --dry-run` builds the catalog in memory but does not write it to disk. A non-dry-run combo execution writes the catalog, but it may also run whichever training/inference/baseline toggles are enabled. Check toggles before using a real run just to refresh the catalog.

### Bootstrap Method B: Manually Add Catalog Entries

Use this when you need exact control over which clips are training and which clips are held out before first ingestion.

1. Decide which videos are training and which are held-out inference.
2. Add entries for the new species to the catalog's `videos` list.
3. Use `category: "training"` for training clips.
4. Use `category: "inference"` for held-out clips.

Example catalog entries:

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

For a small first pass, a reasonable split might be:

```text
training:  PE_001, PE_002, PE_003, PE_004
inference: PE_005
```

If you manually edit the catalog, keep `summary` and `by_species` consistent or regenerate the catalog with a real combo run after the first ingestion. The ingestor primarily uses the `videos` entries for the explicit split.

## Step 4: Dry-Run Patch Dataset Ingestion

Run:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" \
  --only-species "day_Photinus exampleii" \
  --dry-run
```

Check the output carefully. You want to see:

```text
day_Photinus exampleii -> photinus-exampleii (day_time)
```

and the expected training count.

If using the catalog bootstrap, it should print something like:

```text
root catalog split: train=4 val=0
```

If it says all videos are already ingested, the change log thinks this species/video set already exists.

If it says catalog failed, inspect the catalog entries for:

- exact species token,
- exact route,
- exact video names/stems,
- valid JSON syntax.

## Step 5: Run Patch Dataset Ingestion

After the dry run looks right:

```bash
python3 "tools for scaling species/patch_classification_models_dataset_ingestor.py" \
  --only-species "day_Photinus exampleii"
```

The ingestor creates or updates:

```text
single species datasets/photinus-exampleii/<version>/
integrated pipeline datasets/<version>/day_time_dataset/
```

Do not manually create these integrated dataset folders. Let the ingestor do it.

After ingestion, check:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/codex_change_log.jsonl
```

and verify the new dataset version exists under:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/patch training datasets and pipeline validation data/Integrated_prototype_datasets
```

## Step 6: Regenerate or Check the Catalog

Run a combo-runner dry run to check the plan:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Remember: dry run does not write the catalog.

If you used Bootstrap Method A, the on-disk catalog will not necessarily update until a real combo-runner execution writes it. Before running a real combo run, check the toggles at the top of `tmp_day_night_combo_train_and_infer.py` so you do not accidentally train, infer, or run baselines you did not intend.

Inspect the current on-disk catalog:

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

Expected:

```text
training day PE_001.mp4
training day PE_002.mp4
training day PE_003.mp4
training day PE_004.mp4
inference day PE_005.mp4
```

If all entries are still inference after a real catalog-writing run, the training clips did not make it into the integrated dataset or the catalog did not regenerate from the expected training stems.

## Step 7: Add Day YOLO Data

Because this is a day species, patch data is not enough. The day pipeline also needs YOLO coverage on long-exposure images.

Generate long-exposure training images:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_generate_daytime_training_long_exposures.py" --dry-run
```

If the generator script is currently hardcoded to another species, edit its species selection to point at:

```text
photinus-exampleii
```

Then run it for real.

Label the generated long-exposure images externally.

Then ingest the YOLO labels:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py" \
  "/path/to/yolo/export_or_zip" \
  --species-tag "photinus-exampleii" \
  --dry-run
```

If the dry run is correct:

```bash
python3 "tools for scaling species/YOLO model dataset creation, ingestion, and training scripts/tmp_ingest_daytime_yolo_dataset.py" \
  "/path/to/yolo/export_or_zip" \
  --species-tag "photinus-exampleii"
```

Expected YOLO dataset folder:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/dataset/photinus-exampleii
```

with:

```text
images/
labels/
```

## Step 8: Train Patch Models

Edit:

```text
tools for scaling species/tmp_day_night_combo_train_and_infer.py
```

For patch model training:

```python
RUN_MODEL_TRAINING = True
RUN_DAY_MODEL_TRAINING = True
RUN_NIGHT_MODEL_TRAINING = False
TRAIN_GLOBAL_MODELS = True
TRAIN_LEAVEOUT_MODELS = True
```

Keep:

```python
TRAIN_RESNET = "resnet18"
```

The inference loaders expect ResNet18 checkpoints.

Dry run:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Then run for real when the plan is correct.

Expected promoted model-zoo locations:

```text
model zoo/global models/daytime global/best.pt
model zoo/global models/daytime global/training_manifest.json
model zoo/leave out models/photinus-exampleii/best.pt
model zoo/leave out models/photinus-exampleii/training_manifest.json
```

## Step 9: Train Day YOLO Models

If you added YOLO labels for this species, enable day YOLO training:

```python
RUN_DAY_YOLO_MODEL_TRAINING = True
TRAIN_DAY_YOLO_GLOBAL_MODEL = True
TRAIN_DAY_YOLO_LEAVEOUT_MODELS = False
```

Use leaveout YOLO only if you want leaveout day inference:

```python
TRAIN_DAY_YOLO_LEAVEOUT_MODELS = True
```

Because `photinus-exampleii` was added to `ROUTE_BY_SPECIES`, it becomes part of `DAY_ROUTE_SPECIES`. The current `DAY_YOLO_TRAINING_SPECIES_SWITCHES` is built from `DAY_ROUTE_SPECIES`, so verify that the new species is included as intended.

Dry run first:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Expected YOLO model outputs:

```text
v3 daytime YOLO model data/models/global models/<date>/best_firefly_yolo.pt
v3 daytime YOLO model data/models/leave one out models/<date>/photinus-exampleii/best_firefly_yolo.pt
```

## Step 10: Run Pipeline Inference on Held-Out Videos

For global model evaluation, set:

```python
RUN_DAY_PIPELINE_INFERENCE = True
RUN_NIGHT_PIPELINE_INFERENCE = False
RUN_GLOBAL_MODEL_INFERENCE = True
RUN_LEAVEOUT_MODEL_INFERENCE = False
```

For leaveout evaluation, only enable this after both leaveout patch and leaveout YOLO models exist:

```python
RUN_LEAVEOUT_MODEL_INFERENCE = True
LEAVEOUT_DAY_INFERENCE_SPECIES_SWITCHES["photinus-exampleii"] = True
```

Run dry first:

```bash
python3 "tools for scaling species/tmp_day_night_combo_train_and_infer.py" --dry-run
```

Then run for real.

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

## Step 11: Review Day Pipeline Outputs

For each held-out `Photinus exampleii` inference video, inspect:

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

1. Are flashes visible in Stage 1 long-exposure images?
2. Did YOLO detect the right long-exposure regions in Stage 2?
3. Are Stage 3 positive crops real fireflies?
4. Did Stage 3.1 reject real flash trajectories?
5. Are Stage 3.2 centroids reasonable?
6. Do Stage 6 overlays align with GT?
7. Are FPs likely model errors or missing annotations?

## Step 12: Promote or Roll Back

Promote only if metrics and overlays look good.

Promote:

- global day patch model,
- leaveout patch model if useful,
- global day YOLO model,
- leaveout day YOLO model if useful,
- updated catalog,
- run metadata/manifests.

Do not promote:

- checkpoints without manifests,
- models trained with unclear split leakage,
- models whose overlays look bad even if aggregate metrics look acceptable.

If the run is bad, keep the run folder for debugging but do not copy its models into the durable model zoo.

## Night Species Differences

If this example were a night species instead, for example:

```text
night_Photinus exampleii
photinus-exampleii
```

the differences would be:

1. Register route as `"night"` in `ROUTE_BY_SPECIES`.
2. Add the species to `GLOBAL_NIGHT_INFERENCE_SPECIES_SWITCHES`.
3. Add the species to `LEAVEOUT_NIGHT_INFERENCE_SPECIES_SWITCHES` if needed.
4. Skip all day YOLO steps.
5. Enable:

```python
RUN_NIGHT_MODEL_TRAINING = True
RUN_DAY_MODEL_TRAINING = False
RUN_NIGHT_PIPELINE_INFERENCE = True
RUN_DAY_PIPELINE_INFERENCE = False
```

Night model outputs go through the night global/leaveout patch model zoo and night pipeline validation.

## Common Mistakes

### Forgetting `ROUTE_BY_SPECIES`

Symptom:

- Catalog entries have missing or wrong `species_name`.
- Gateway cannot route cleanly.

Fix:

- Add the species token to `ROUTE_BY_SPECIES`.

### Letting held-out clips enter training

Symptom:

- Evaluation metrics look too good.

Fix:

- Bootstrap the catalog explicitly.
- Verify catalog categories before ingestion.

### Skipping day YOLO work for a day species

Symptom:

- Patch classifier is trained, but day pipeline still misses the species because YOLO never proposes the right long-exposure boxes.

Fix:

- Generate, label, ingest, and train day YOLO data for the species.

### Enabling leaveout day inference without leaveout YOLO

Symptom:

- Combo runner errors about missing leaveout day YOLO model.

Fix:

- Train leaveout day YOLO or keep leaveout day inference disabled.

### Assuming annotation `x,y` semantics

Symptom:

- Validation overlays show consistent spatial shift.

Fix:

- Verify whether annotator `x,y` is top-left or center.
- Convert annotations or adjust validation preprocessing before trusting metrics.
