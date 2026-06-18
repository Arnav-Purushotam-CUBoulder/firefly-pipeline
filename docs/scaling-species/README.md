# Scaling Species Deep Dive

This folder documents the scaling layer around the two core pipelines.

Source folder:

```text
tools for scaling species
```

Read these pages in order:

1. [raw-data-and-catalog.md](raw-data-and-catalog.md) - raw root structure, species naming, and train/inference catalog.
2. [patch-dataset-ingestion.md](patch-dataset-ingestion.md) - how annotated videos become patch datasets.
3. [combo-runner.md](combo-runner.md) - the main train/infer/baseline experiment runner.
4. [model-zoo-and-training.md](model-zoo-and-training.md) - global and leaveout patch models.
5. [day-yolo-workflow.md](day-yolo-workflow.md) - daytime YOLO long-exposure data and checkpoints.
6. [baselines-and-evaluation.md](baselines-and-evaluation.md) - lab/Raphael baselines and final result records.
7. [add-species-runbook.md](add-species-runbook.md) - end-to-end checklist for adding or scaling a species.
8. [worked-example-new-day-species.md](worked-example-new-day-species.md) - concrete `Photinus exampleii` day-species example with code edits and commands.

## Short Mental Model

The scaling code does not replace the day or night pipelines. It wraps them.

The scaling layer owns:

- raw video and annotation discovery,
- train/inference split cataloging,
- patch dataset construction,
- global and leaveout classifier training,
- day YOLO dataset/model maintenance,
- gateway-driven inference,
- baseline execution,
- final metrics aggregation.

The two application pipelines remain the execution targets:

- Day route: long exposure, YOLO, patch classifier, trajectory/intensity filter, centroids.
- Night route: blob detection, CNN classifier, merge/centroid/repair, validation.
