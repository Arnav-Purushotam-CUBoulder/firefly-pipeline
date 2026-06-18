# Day Pipeline Deep Dive

This folder documents the day-time pipeline in more operational detail than the top-level overview.

Source folder:

```text
Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)
```

Read these pages in order:

1. [stage-flow.md](stage-flow.md) - exact pipeline stages, inputs, outputs, and why each stage exists.
2. [configuration.md](configuration.md) - the important `params.py` knobs and when to change them.
3. [data-contracts.md](data-contracts.md) - CSV schemas, coordinate semantics, naming conventions, and output paths.
4. [validation-and-analysis.md](validation-and-analysis.md) - Stage 5 through Stage 9 validation behavior.
5. [operations-and-debugging.md](operations-and-debugging.md) - direct runs, gateway runs, smoke tests, and common failure modes.

## Short Mental Model

The day pipeline is not a per-frame blob detector. It first compresses video intervals into long-exposure images, detects streak/candidate regions with YOLO, then expands each long-exposure box back into per-frame patches and uses a patch classifier to decide whether the crop is a firefly. Later stages keep only flash-like trajectories and convert box detections into centroid-style outputs.

The normal final point output is Stage 3.2:

```text
ROOT/stage3_2 xyt for 3d reconstruction/<video_stem>.csv
```

with columns:

```text
x,y,t
```

Do not treat raw Stage 3 `x,y` as final centroid coordinates. Stage 3 `x,y` are top-left patch coordinates.

