# Night Pipeline Deep Dive

This folder documents the night-time pipeline in operational detail.

Source folder:

```text
Pipelines/night_time_pipeline
```

Read these pages in order:

1. [stage-flow.md](stage-flow.md) - exact stage sequence and what each stage changes.
2. [configuration.md](configuration.md) - `pipeline_params.py` knobs and how to tune them.
3. [data-contracts.md](data-contracts.md) - CSV schemas, coordinate semantics, and output paths.
4. [repair-and-validation.md](repair-and-validation.md) - Stage 8.5/8.6/8.7/8.9 and Stage 9-14 details.
5. [operations-and-debugging.md](operations-and-debugging.md) - run commands, smoke tests, and common failures.

## Short Mental Model

The night pipeline starts from bright pixels instead of long exposures. It detects candidate blobs in each frame, recenters them, removes tiny or dim objects, classifies patches with a CNN, merges duplicate boxes, converts the final detections to center semantics, applies repair passes, then validates against normalized GT.

The main prediction CSV is:

```text
ROOT/csv files/<video_stem>.csv
```

After Stage 8, this CSV should have center semantics:

```text
x,y are center coordinates
w,h are fixed 10x10
xy_semantics is center
```

The night validation stage expects GT as `x,y,t`. If GT is still `x,y,w,h,frame`, Stage 8.9 is responsible for converting/recentering it before Stage 9.

