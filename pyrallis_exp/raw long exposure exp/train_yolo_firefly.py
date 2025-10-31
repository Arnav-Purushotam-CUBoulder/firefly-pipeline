#!/usr/bin/env python3
"""
Train a YOLO model to detect firefly streaks in long‑exposure images.

Dataset format: YOLOv8 (data.yaml + images/labels in YOLO format).
Edit the globals below, then run this file from your IDE or terminal.

Requires: pip install ultralytics>=8.0.0
"""

from __future__ import annotations
from pathlib import Path
import shutil
import sys
from datetime import datetime
import cv2
import torch
import os

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Ultralytics is required. Install with: pip install ultralytics\n"
        f"Import error: {e}"
    )


# ===================== Globals (edit these) =====================

# Path to dataset root OR YAML (YOLOv8 format). If DATA_YAML is None,
# the script expects to find 'data.yaml' inside DATASET_ROOT.
DATASET_ROOT: Path | None = Path("/Users/arnavps/Downloads/streaks.v3i.yolov8")
DATA_YAML: Path | None = None  # e.g., Path("/path/to/dataset/data.yaml")

# Choose model family and size. Examples:
#   YOLOv8:  'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
#   YOLOv11: 'yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt', 'yolov11l.pt', 'yolov11x.pt'
#   Custom  : path to your own checkpoint '.pt'
MODEL_WEIGHTS = "yolov8s.pt"

# Training hyperparameters
EPOCHS = 50
IMG_SIZE: int | None = None  # None → auto from dataset image size (max(H,W))
BATCH = 1
DEVICE: str | int | None = "cpu"  # 'auto'|'cpu'|'mps'|CUDA index or list, e.g., 0
WORKERS = 2
PATIENCE = 20
LR0 = 0.01  # initial learning rate; None to use default
WEIGHT_DECAY = 0.0005

# Where Ultralytics will put the run (project/name)
PROJECT_DIR = Path("/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/test1/pyrallis_exp/raw long exposure exp/yolo train output data/runs_firefly")
RUN_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Export a copy of the best weights to this path after training
EXPORT_BEST_TO = Path("/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/test1/pyrallis_exp/raw long exposure exp/yolo train output data/best models folder/v1_best_firefly_yolo.pt")

# If True, delete previous run directory with same RUN_NAME before training
CLEAR_EXISTING_RUN = False

# On Apple Silicon (MPS), some ops (e.g., torchvision NMS) are not implemented.
# Enable CPU fallback for unsupported ops to avoid runtime errors during
# validation. Disable if you prefer strict MPS-only execution.
MPS_CPU_FALLBACK = True


# ===================== Helpers =====================

def _ensure_valid_paths():
    # Resolve YAML path
    global DATA_YAML
    if DATA_YAML is None:
        if DATASET_ROOT is None:
            raise FileNotFoundError("Provide either DATA_YAML or DATASET_ROOT")
        DATA_YAML = Path(DATASET_ROOT) / "data.yaml"
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"DATA_YAML not found: {DATA_YAML}")
    pd = Path(PROJECT_DIR)
    pd.mkdir(parents=True, exist_ok=True)


def _maybe_clear_existing_run():
    run_dir = Path(PROJECT_DIR) / RUN_NAME
    if CLEAR_EXISTING_RUN and run_dir.exists():
        shutil.rmtree(run_dir)


def _find_best_weight_after_train(model: YOLO) -> Path:
    # Prefer trainer.best if available
    best = None
    try:
        best_attr = getattr(model.trainer, "best", None)
        if best_attr:
            p = Path(best_attr)
            if p.exists():
                best = p
    except Exception:
        best = None
    if best and best.exists():
        return best
    # Fallback: search in project/name/weights/best.pt
    save_dir = getattr(model, "trainer", None) and getattr(model.trainer, "save_dir", None)
    if save_dir:
        p = Path(save_dir) / "weights" / "best.pt"
        if p.exists():
            return p
    # Last resort: glob
    proj = Path(PROJECT_DIR)
    for p in proj.rglob("best.pt"):
        return p
    raise FileNotFoundError("Could not locate best.pt after training")


# ==== Dataset discovery + auto image size ====

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _read_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # Fallback: very small parser for 'path', 'train'
        data = {}
        root = None
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("path:"):
                    root = line.split(":", 1)[1].strip().strip("'\"")
                    data["path"] = root
                elif line.startswith("train:"):
                    train = line.split(":", 1)[1].strip().strip("'\"")
                    data["train"] = train
        return data


def _discover_split_dir(root: Path, split_names: list[str]) -> Path | None:
    """Return a valid images directory under root matching split names.

    Tries root/<name>/images then root/images/<name>, then root/<name> directly.
    """
    for name in split_names:
        p = root / name / "images"
        if p.exists():
            return p.resolve()
        p = root / "images" / name
        if p.exists():
            return p.resolve()
        p = root / name
        if p.exists() and p.is_dir():
            # If this folder contains images directly, accept
            any_img = next((x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS), None)
            if any_img is not None:
                return p.resolve()
    return None


def _auto_imgsz_from_dataset(yaml_path: Path, dataset_root: Path | None = None) -> int | None:
    # Prefer dataset_root if provided
    train_dir = None
    if dataset_root is not None:
        train_dir = _discover_split_dir(Path(dataset_root), ["train", "training"])  # typical names

    if train_dir is None:
        info = _read_yaml(yaml_path)
        base = Path(info.get("path", ""))
        train_entry = info.get("train")
        if train_entry:
            train_path = Path(train_entry)
            if not train_path.is_absolute() and base:
                train_dir = (yaml_path.parent / base / train_path).resolve()
            elif not train_path.is_absolute():
                train_dir = (yaml_path.parent / train_path).resolve()
            else:
                train_dir = train_path

    if not train_dir or not train_dir.exists():
        return None
    for p in train_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            h, w = im.shape[:2]
            size = max(h, w)
            print(f"Auto-detected dataset image size: {w}x{h} → imgsz={size}")
            return int(size)
    return None


# ==== Device selection (CPU / MPS / CUDA) ====

def _select_device(user_choice: str | int | None):
    # Respect explicit non-auto choices first
    if isinstance(user_choice, int):
        return user_choice
    if isinstance(user_choice, str) and user_choice.lower() not in {"auto", "", "none"}:
        return user_choice

    # Auto detection
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return 0
    return "cpu"


# ==== Fix/standardize dataset YAML ====

def _prepare_data_yaml(orig_yaml: Path, out_dir: Path, dataset_root: Path | None) -> Path:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("pyyaml is required for YAML rewriting: pip install pyyaml") from e

    with open(orig_yaml, "r") as f:
        data = yaml.safe_load(f) or {}

    # Decide dataset root, prefer explicit DATASET_ROOT
    base_root = Path(dataset_root) if dataset_root is not None else Path(data.get("path") or orig_yaml.parent)

    # Discover split folders directly under dataset root
    train_dir = _discover_split_dir(base_root, ["train", "training"])
    val_dir = _discover_split_dir(base_root, ["val", "valid", "validation"]) or train_dir
    test_dir = _discover_split_dir(base_root, ["test"]) if _discover_split_dir(base_root, ["test"]) else None

    if not train_dir or not train_dir.exists():
        raise FileNotFoundError(f"Could not locate train images under {base_root}. Expecting '<root>/train/images'.")

    new_data = dict(data)  # copy keep nc/names, etc.
    new_data["path"] = str(base_root)
    new_data["train"] = str(train_dir)
    new_data["val"] = str(val_dir)
    if test_dir:
        new_data["test"] = str(test_dir)
    new_data.pop("valid", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_yaml = out_dir / "_data_fixed.yaml"
    with open(fixed_yaml, "w") as f:
        yaml.safe_dump(new_data, f, sort_keys=False)
    print(f"Using standardized dataset YAML → {fixed_yaml}")
    print(f"  train: {new_data['train']}")
    print(f"  val:   {new_data['val']}")
    if new_data.get('test'):
        print(f"  test:  {new_data['test']}")
    return fixed_yaml


# ===================== Main =====================

def main():
    _ensure_valid_paths()
    _maybe_clear_existing_run()

    model = YOLO(MODEL_WEIGHTS)

    # Derive image size if requested
    imgsz = int(IMG_SIZE) if IMG_SIZE is not None else (_auto_imgsz_from_dataset(Path(DATA_YAML), DATASET_ROOT) or 640)
    device = _select_device(DEVICE)
    if str(device).lower() == "mps" and MPS_CPU_FALLBACK:
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("Enabled PYTORCH_ENABLE_MPS_FALLBACK=1 for unsupported MPS ops (e.g., NMS)")

    # Prepare a standardized data.yaml under the run directory
    run_dir = Path(PROJECT_DIR) / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    data_yaml_fixed = _prepare_data_yaml(Path(DATA_YAML), run_dir, DATASET_ROOT)

    train_args = dict(
        data=str(data_yaml_fixed),
        epochs=int(EPOCHS),
        imgsz=imgsz,
        batch=int(BATCH),
        device=device,
        workers=int(WORKERS),
        project=str(PROJECT_DIR),
        name=str(RUN_NAME),
        patience=int(PATIENCE),
    )
    if LR0 is not None:
        train_args["lr0"] = float(LR0)
    if WEIGHT_DECAY is not None:
        train_args["weight_decay"] = float(WEIGHT_DECAY)

    # No additional low-memory overrides; use the provided hyperparameters

    print("Starting training with args:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")

    results = model.train(**train_args)
    # Access to results is not guaranteed across versions; rely on files
    best_pt = _find_best_weight_after_train(model)
    print(f"Best weights found at: {best_pt}")

    # Export a copy
    dest = Path(EXPORT_BEST_TO)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, dest)
    print(f"Copied best weights → {dest}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
