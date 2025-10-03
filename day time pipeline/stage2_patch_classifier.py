#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import numpy as np

import params
from utils import open_video, center_crop_with_pad, progress


_MODEL_CACHE = {"model": None, "device": None, "warmed": False}


def _device():
    try:
        import torch
        pref = str(getattr(params, 'STAGE2_DEVICE', 'auto')).lower()
        if pref == 'cpu':
            return torch.device('cpu')
        if pref == 'cuda':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if pref == 'mps':
            if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        # auto: prefer mps → cuda → cpu
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    except Exception as e:
        raise RuntimeError("PyTorch is required for Stage 2. Install torch/torchvision.") from e


def _to_tensor_batch(np_list, img_size: int):
    import torch
    # np_list: list of RGB uint8 arrays HxWx3
    # If resize is requested and different from current, use OpenCV for speed
    out = []
    do_norm = bool(getattr(params, 'IMAGENET_NORMALIZE', False))
    for arr in np_list:
        if arr.shape[0] != img_size or arr.shape[1] != img_size:
            arr = cv2.resize(arr, (int(img_size), int(img_size)), interpolation=cv2.INTER_LINEAR)
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0)
        if do_norm:
            mean = torch.tensor(params.IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(params.IMAGENET_STD).view(3, 1, 1)
            ten = (ten - mean) / std
        out.append(ten)
    return torch.stack(out, dim=0)


def _load_model(model_path: Path):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    if not model_path.exists():
        raise FileNotFoundError(f"Patch model not found: {model_path}. Set PATCH_MODEL_PATH in params.py")

    # Cache model across videos for speed
    if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["device"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["device"]

    dev = _device()
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    ckpt = torch.load(str(model_path), map_location=dev)

    # Match the training script: prefer 'model' key; then 'state_dict'; then raw
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    else:
        state = ckpt

    # Strip possible 'module.' prefix
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = m.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        print(f"Stage2  Warning: load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    m.eval().to(dev)
    # Warm-up once to reduce first-batch latency (esp. on MPS)
    if not _MODEL_CACHE["warmed"]:
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, int(getattr(params, 'STAGE2_INPUT_SIZE', 10)), int(getattr(params, 'STAGE2_INPUT_SIZE', 10)), device=dev)
            _ = m(dummy)
        _MODEL_CACHE["warmed"] = True
    _MODEL_CACHE["model"] = m
    _MODEL_CACHE["device"] = dev
    return m, dev


def _load_stage1_rows(stem: str) -> dict[int, dict]:
    s1_csv = (params.STAGE1_DIR / stem) / f"{stem}_trajectories.csv"
    out: dict[int, dict] = {}
    with s1_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            gid = int(row["global_id"])  # unique id assigned in stage1
            out[gid] = row
    return out


def _all_gids_from_stage1(rows: dict[int, dict]) -> list[int]:
    return sorted(int(g) for g in rows.keys())


def run_for_video(video_path: Path) -> Path:
    stem = video_path.stem
    s1_rows = _load_stage1_rows(stem)
    # Use ALL Stage 1 candidates
    selected = set(_all_gids_from_stage1(s1_rows))

    out_root = (params.STAGE2_DIR / stem)
    pos_dir = out_root / "crops" / "positives"
    neg_dir = out_root / "crops" / "negatives"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    cap, W, H, fps, total = open_video(video_path)

    # Index: frame -> list[(gid, cx, cy)]
    by_frame = defaultdict(list)
    for gid in selected:
        row = s1_rows.get(gid)
        if not row:
            continue
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["w"])
        h = int(row["h"])
        t = int(row["first_t"])
        cx = x + w / 2.0
        cy = y + h / 2.0
        by_frame[int(t)].append((gid, cx, cy))

    # Load model + transform
    model, dev = _load_model(params.PATCH_MODEL_PATH)
    import torch

    out_csv = out_root / f"{stem}_patches.csv"
    total_patches = 0
    pos_count = 0
    neg_count = 0
    # Debug summary of probabilities to validate preprocessing matches training
    pos_probs_all: list[float] = []

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        # As requested: x,y,w,h,frame_number
        writer.writerow(["x", "y", "w", "h", "frame_number"])

        frames_sorted = sorted(by_frame.keys())
        # Respect global frame cap if provided
        if params.MAX_FRAMES is not None:
            max_idx = max(0, int(params.MAX_FRAMES) - 1)
            frames_sorted = [t for t in frames_sorted if t <= max_idx]

        if not frames_sorted:
            print("Stage2  NOTE: No frames to process (possibly due to MAX_FRAMES).")
            print(f"Stage2  Candidates from Stage1: {len(selected)}; frames_in_range=0")
            # leave only header in CSV and return
            cap.release()
            print("Stage2  Summary: patches=0, positives=0, negatives=0")
            print(f"Stage2  Wrote patches CSV → {out_csv}")
            return out_csv
        for idx, t in enumerate(frames_sorted):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(t))
            ok, frame = cap.read()
            if not ok:
                continue

            metas: List[tuple[int, float, float]] = by_frame[t]
            np_patches = []
            for gid, cx, cy in metas:
                patch, x0, y0 = center_crop_with_pad(frame, cx, cy, params.PATCH_SIZE_PX, params.PATCH_SIZE_PX)
                rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                np_patches.append(rgb)
            if not np_patches:
                continue

            x = _to_tensor_batch(np_patches, int(getattr(params, 'STAGE2_INPUT_SIZE', 10))).to(dev)
            with torch.inference_mode():
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[:, int(getattr(params, 'STAGE2_POSITIVE_CLASS_INDEX', 1))]
            thr = float(getattr(params, 'STAGE2_POSITIVE_THRESHOLD', 0.5))
            preds = (prob >= thr).cpu().numpy().astype(bool)
            probs_np = prob.detach().cpu().numpy()
            # accumulate for overall summary
            pos_probs_all.extend(probs_np.tolist())

            total_patches += len(metas)
            for (gid, cx, cy), is_pos, p_conf in zip(metas, preds, probs_np):
                patch, _, _ = center_crop_with_pad(frame, cx, cy, params.PATCH_SIZE_PX, params.PATCH_SIZE_PX)
                out_dir = pos_dir if is_pos else neg_dir
                # Filename: frame number + x_y_w_h + model confidence
                out_name = (
                    f"t_{int(t):06d}_x{int(round(cx))}_y{int(round(cy))}_"
                    f"w{int(params.PATCH_SIZE_PX)}_h{int(params.PATCH_SIZE_PX)}_p{float(p_conf):.3f}.png"
                )
                cv2.imwrite(str(out_dir / out_name), patch)
                if is_pos:
                    writer.writerow([float(cx), float(cy), int(params.PATCH_SIZE_PX), int(params.PATCH_SIZE_PX), int(t)])
                    pos_count += 1
                else:
                    neg_count += 1

            if idx % 10 == 0:
                progress(idx + 1, max(1, len(frames_sorted)), "Stage2 classify")
        progress(len(frames_sorted), max(1, len(frames_sorted)), "Stage2 classify done")

    cap.release()
    # Optional: print preprocessing config to ensure parity with training
    print(
        f"Stage2  Config: input_size={getattr(params,'STAGE2_INPUT_SIZE',None)} "
        f"normalize={getattr(params,'IMAGENET_NORMALIZE',None)} pos_idx={getattr(params,'STAGE2_POSITIVE_CLASS_INDEX',None)} "
        f"thr={getattr(params,'STAGE2_POSITIVE_THRESHOLD',None)}"
    )
    if pos_probs_all:
        import numpy as _np
        arr = _np.asarray(pos_probs_all, dtype=_np.float32)
        print(
            f"Stage2  Probs summary (positive class): min={arr.min():.3f} "
            f"p25={_np.percentile(arr,25):.3f} mean={arr.mean():.3f} p75={_np.percentile(arr,75):.3f} max={arr.max():.3f}"
        )
    print(f"Stage2  Summary: patches={total_patches}, positives={pos_count}, negatives={neg_count}")
    print(f"Stage2  Wrote patches CSV → {out_csv}")
    return out_csv


__all__ = ["run_for_video"]
