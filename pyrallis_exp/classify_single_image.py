#!/usr/bin/env python3
"""
Quick single-image classifier for the Pyrallis ResNet18 binary model.

Edit IMAGE_PATH and MODEL_PATH below, then run this file.
Prints softmax probabilities and predicted class index.
"""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


# ======= USER SETTINGS =======
# Path to the input image to classify (PNG/JPG etc.)
IMAGE_PATH = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/daytime pipeline v1 inference output data/stage2_patch_classifier/20240606_cam1_GS010064/crops/negatives/t_000304_x1684_y992_w10_h10_p0.479.png')

# Path to the trained model checkpoint (.pt) saved by your training script
MODEL_PATH = Path("/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/models and other data/pyrallis gopro models resnet18/resnet18_pyrallis_gopro_best_model.pt")

# Model/input config
INPUT_SIZE = 10                 # match training (10x10 patches)
POSITIVE_CLASS_INDEX = 1        # 0 or 1 depending on training label ordering
IMAGENET_NORMALIZE = False      # keep False to mirror training script


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))


def _build_transform():
    from torchvision import transforms as T
    tfms = [
        T.ToPILImage(),
        T.Resize((int(INPUT_SIZE), int(INPUT_SIZE)), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ]
    if IMAGENET_NORMALIZE:
        tfms.append(T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return T.Compose(tfms)


def _load_model(model_path: Path):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    assert model_path.exists(), f"Model not found: {model_path}"
    dev = _device()

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(str(model_path), map_location=dev)
    # Training script saved under 'model', but be permissive
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    else:
        state = ckpt

    # Strip DistributedDataParallel prefix if present
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)
    model.eval().to(dev)
    return model, dev


def classify_image(image_path: Path, model_path: Path) -> None:
    import torch

    model, dev = _load_model(model_path)
    tfm = _build_transform()

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert img_bgr is not None, f"Could not read image: {image_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = tfm(img_rgb).unsqueeze(0).to(dev)

    with torch.inference_mode():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pos_idx = int(POSITIVE_CLASS_INDEX)
    pos_conf = float(prob[pos_idx])
    pred_idx = int(prob.argmax())

    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Probs: negative={prob[1-pos_idx]:.4f}, positive={prob[pos_idx]:.4f}")
    print(f"Predicted class index: {pred_idx}  (positive_index={pos_idx})")


def main():
    img = Path(IMAGE_PATH)
    mdl = Path(MODEL_PATH)
    if not img.exists():
        sys.exit(f"IMAGE_PATH not found: {img}")
    if not mdl.exists():
        sys.exit(f"MODEL_PATH not found: {mdl}")
    classify_image(img, mdl)


if __name__ == "__main__":
    main()

