# Environment and Dependencies

## Expected Machine

The current repo assumes a Linux workstation with:

- Python 3 available as `python3`.
- The Samsung SSD mounted at `/mnt/Samsung_SSD_2TB`.
- Enough disk space under `/mnt/Samsung_SSD_2TB/temp to delete` for intermediate crops, CSVs, rendered videos, and experiment outputs.
- CUDA-capable GPU for the fastest night Stage 1 path and for model inference/training, although some CPU fallbacks exist.

`python` was not present in this checkout during inspection. Use `python3` in commands.

## Path Quoting

The repo path and many data paths contain spaces and apostrophes. Always quote paths:

```bash
python3 "Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/orchestrator.py"
```

Avoid building shell commands that rely on unquoted paths.

## Python Packages

There is no checked-in `requirements.txt`, `pyproject.toml`, or environment file in this repo. The code imports the following major packages:

- `opencv-python` / `cv2`
- `numpy`
- `torch`
- `torchvision`
- `Pillow`
- `ultralytics`
- `pandas`
- `tqdm`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `tkinter` for the annotation GUI
- `cupy` and `cucim` for the default night Stage 1 GPU detector

The exact installed versions should be captured from the working machine before rebuilding an environment.

Useful environment capture commands:

```bash
python3 -m pip freeze > environment.freeze.txt
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
PY
```

## GPU Notes

Day pipeline:

- Uses Ultralytics YOLO for Stage 2.
- Uses Torch/ResNet18 patch classification for Stage 3.
- `params.py` supports `YOLO_DEVICE`, `YOLO_BATCH_SIZE`, `STAGE3_DEVICE`, mixed precision, and TF32 toggles.

Night pipeline:

- Default `STAGE1_VARIANT = 'cucim'`.
- The cuCIM path requires compatible CUDA, CuPy, cuCIM, and NVRTC/Jitify behavior.
- Alternative Stage 1 implementations exist: `cc_cuda`, `cc_cpu`, and the older SimpleBlobDetector path.

If cuCIM fails on a new machine, first try documenting the error and running a small night smoke test with another Stage 1 variant before changing algorithm parameters.

## Model File Expectations

Patch classifiers are ResNet18 binary classifiers where class index `1` is firefly. Day YOLO uses Ultralytics `.pt` weights.

The day pipeline has a workaround for Ultralytics path parsing when the weights path contains an apostrophe: it copies weights into `~/.cache/firefly_pipeline/ultralytics_weights` and runs from the apostrophe-safe path.

## External Tools

`jq` was not available during inspection. Prefer Python one-liners for JSON inspection unless the environment is updated.

