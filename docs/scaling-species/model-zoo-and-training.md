# Model Zoo and Patch Model Training

Patch model zoo root:

```text
/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/model zoo
```

The model zoo stores patch classifiers used by the gateway/pipelines.

## Expected Layout

```text
model zoo/
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

## Model Types

### Global route model

One model per route:

- day global model trained on all day-route training species.
- night global model trained on all night-route training species.

Model key:

```text
global_all_species
```

### Leaveout model

One model per held-out species:

- trained on route species excluding the named species.
- evaluated on the left-out species' held-out inference videos.

Model key:

```text
leaveout_<species>
```

Example:

```text
leaveout_photinus-carolinus
```

## Training Sources

The combo runner discovers latest integrated patch dataset versions:

```text
Integrated_prototype_datasets/integrated pipeline datasets/<version>/day_time_dataset/final dataset
Integrated_prototype_datasets/integrated pipeline datasets/<version>/night_time_dataset/final dataset
```

It requires trainable datasets with:

```text
train/firefly
train/background
```

## Training Hyperparameters

Current combo runner defaults:

```python
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 128
TRAIN_LR = 3e-4
TRAIN_NUM_WORKERS = 2
TRAIN_RESNET = "resnet18"
TRAIN_SEED = 1337
```

Inference loaders expect ResNet18 checkpoints. The runner exits if `TRAIN_RESNET` is not `resnet18`.

## Dataset Sanitizer

Before training, datasets can be sanitized:

```python
SANITIZE_DATASET_IMAGES = True
SANITIZE_DATASET_MODE = "quarantine"
SANITIZE_DATASET_VERIFY_WITH_PIL = True
SANITIZE_DATASET_REPORT_MAX = 20
```

This helps catch corrupt images before training.

## Exported Manifests

Global manifest:

```text
global models/<route global>/training_manifest.json
```

Leaveout manifest:

```text
leave out models/<species>/training_manifest.json
```

Manifests record:

- manifest version,
- route,
- model scope,
- model key,
- left-out species where applicable,
- trained-on species,
- best checkpoint path,
- training metrics path,
- hyperparameters,
- sanitizer settings.

## Loading Models for Inference

When training is disabled and inference is enabled, the combo runner loads model specs from the model zoo.

Global route:

```text
global models/daytime global/best.pt
global models/night time global/best.pt
```

Leaveout:

```text
leave out models/<species>/best.pt
```

If required files are missing, the runner raises an error rather than silently skipping.

## Recommended Model Update Flow

1. Ingest new training data.
2. Confirm integrated dataset versions and patch counts.
3. Enable model training toggles in the combo runner.
4. Run `--dry-run`.
5. Run a small training/evaluation if possible.
6. Inspect training metrics.
7. Export to model zoo only if metrics and inference overlays look reasonable.
8. Keep `training_manifest.json` with every exported `best.pt`.

