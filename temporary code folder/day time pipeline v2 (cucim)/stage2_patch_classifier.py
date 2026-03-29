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


def _read_stage1_boxes(stem: str) -> list[dict]:
    """Read Stage 1 CSV in cuCIM format: frame,x,y,w,h."""
    s1_csv = (params.STAGE1_DIR / stem) / f"{stem}.csv"
    rows: list[dict] = []
    with s1_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _by_frame_from_stage1(rows: list[dict]) -> dict[int, list[tuple[float, float]]]:
    """Group Stage 1 boxes by frame as centers (cx, cy)."""
    out: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        t = int(row["frame"])  # 0-based
        x = int(row["x"])      # top-left
        y = int(row["y"])      # top-left
        w = int(row["w"])      # width
        h = int(row["h"])      # height
        cx = x + w / 2.0
        cy = y + h / 2.0
        out.setdefault(t, []).append((cx, cy))
    return out


def run_for_video(video_path: Path) -> Path:
    stem = video_path.stem
    s1_rows = _read_stage1_boxes(stem)

    out_root = (params.STAGE2_DIR / stem)
    pos_dir = out_root / "crops" / "positives"
    pos_dir.mkdir(parents=True, exist_ok=True)

    cap, W, H, fps, total = open_video(video_path)

    # Index: frame -> list[(cx, cy)]
    by_frame = _by_frame_from_stage1(s1_rows)

    # Load model + transform
    model, dev = _load_model(params.PATCH_MODEL_PATH)
    import torch

    out_csv = out_root / f"{stem}_patches.csv"
    total_patches = 0
    pos_count = 0
    neg_count = 0
    # Debug summary of probabilities to validate preprocessing matches training
    pos_probs_all: list[float] = []
    # Per-frame stats
    total_by_t = {t: len(by_frame[t]) for t in by_frame.keys()}
    pos_by_t: dict[int, int] = {t: 0 for t in by_frame.keys()}

    import time as _time
    _t0 = _time.perf_counter()
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "w", "h", "frame_number"])

        frames_sorted = sorted(by_frame.keys())
        # Respect global frame cap if provided
        if params.MAX_FRAMES is not None:
            max_idx = max(0, int(params.MAX_FRAMES) - 1)
            frames_sorted = [t for t in frames_sorted if t <= max_idx]

        total_candidates = sum(len(by_frame[t]) for t in frames_sorted)
        if not frames_sorted or total_candidates == 0:
            print("Stage2  NOTE: No frames to process (possibly due to MAX_FRAMES).")
            print(f"Stage2  Candidates from Stage1: {total_candidates}; frames_in_range=0")
            cap.release()
            print("Stage2  Summary: patches=0, positives=0, negatives=0")
            print(f"Stage2  Wrote patches CSV → {out_csv}")
            return out_csv

        # Micro-batch across frames
        batch_size = int(getattr(params, 'STAGE2_BATCH_SIZE', 64))
        cur_batch: list[tuple[int, float, float, np.ndarray]] = []  # (t, cx, cy, RGB patch)
        processed = 0

        def _flush_batch():
            nonlocal cur_batch, pos_count, neg_count, total_patches, processed, pos_probs_all, pos_by_t
            if not cur_batch:
                return
            imgs = [b[3] for b in cur_batch]
            x = _to_tensor_batch(imgs, int(getattr(params, 'STAGE2_INPUT_SIZE', 10))).to(dev)
            with torch.inference_mode():
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[:, int(getattr(params, 'STAGE2_POSITIVE_CLASS_INDEX', 1))]
            thr = float(getattr(params, 'STAGE2_POSITIVE_THRESHOLD', 0.5))
            preds = (prob >= thr).cpu().numpy().astype(bool)
            probs_np = prob.detach().cpu().numpy()
            pos_probs_all.extend(probs_np.tolist())

            for (t, cx, cy, patch), is_pos, p_conf in zip(cur_batch, preds, probs_np):
                if is_pos:
                    out_name = (
                        f"t_{int(t):06d}_x{int(round(cx))}_y{int(round(cy))}_"
                        f"w{int(params.PATCH_SIZE_PX)}_h{int(params.PATCH_SIZE_PX)}_p{float(p_conf):.3f}.png"
                    )
                    # patch is RGB (for model); convert back to BGR for OpenCV write
                    cv2.imwrite(str(pos_dir / out_name), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                    writer.writerow([float(cx), float(cy), int(params.PATCH_SIZE_PX), int(params.PATCH_SIZE_PX), int(t)])
                    pos_count += 1
                    pos_by_t[int(t)] = pos_by_t.get(int(t), 0) + 1
                else:
                    neg_count += 1
            processed += len(cur_batch)
            total_patches += len(cur_batch)
            cur_batch = []
            progress(processed, max(1, total_candidates), "Stage2 classify")

        for t in frames_sorted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(t))
            ok, frame = cap.read()
            if not ok:
                continue
            for (cx, cy) in by_frame[t]:
                patch, _, _ = center_crop_with_pad(frame, cx, cy, params.PATCH_SIZE_PX, params.PATCH_SIZE_PX)
                rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                cur_batch.append((t, cx, cy, rgb))
                if len(cur_batch) >= batch_size:
                    _flush_batch()

        # Flush any remaining
        _flush_batch()
        progress(total_patches, max(1, total_candidates), "Stage2 classify done")

    cap.release()
    _elapsed = _time.perf_counter() - _t0
    frames_processed = len(sorted(by_frame.keys())) if params.MAX_FRAMES is None else len([t for t in by_frame.keys() if t <= max(0, int(params.MAX_FRAMES) - 1)])
    pos_rate = (pos_count / total_patches) if total_patches else 0.0
    avg_patches_per_frame = (total_patches / frames_processed) if frames_processed else 0.0
    avg_pos_per_frame = (pos_count / frames_processed) if frames_processed else 0.0
    throughput = (total_patches / _elapsed) if _elapsed > 0 else 0.0
    # Optional: print preprocessing config to ensure parity with training
    print(
        f"Stage2  Config: input_size={getattr(params,'STAGE2_INPUT_SIZE',None)} "
        f"normalize={getattr(params,'IMAGENET_NORMALIZE',None)} pos_idx={getattr(params,'STAGE2_POSITIVE_CLASS_INDEX',None)} "
        f"thr={getattr(params,'STAGE2_POSITIVE_THRESHOLD',None)} batch={getattr(params,'STAGE2_BATCH_SIZE',None)}"
    )
    if pos_probs_all:
        import numpy as _np
        arr = _np.asarray(pos_probs_all, dtype=_np.float32)
        print(
            f"Stage2  Probs summary (positive class): min={arr.min():.3f} "
            f"p25={_np.percentile(arr,25):.3f} mean={arr.mean():.3f} p75={_np.percentile(arr,75):.3f} max={arr.max():.3f}"
        )
    print(f"Stage2  Candidates: {total_patches}")
    print(f"Stage2  Frames processed: {frames_processed}")
    print(f"Stage2  Time: {_elapsed:.2f}s; Throughput: {throughput:.1f} patches/s")
    print(f"Stage2  Summary: patches={total_patches}, positives={pos_count}, negatives={neg_count}")
    print(f"Stage2  Rates: pos_rate={pos_rate:.3f}, avg_patches/frame={avg_patches_per_frame:.1f}, avg_pos/frame={avg_pos_per_frame:.2f}")

    # Per-frame stats summaries (include zeros for frames in range)
    try:
        import numpy as _np
        # Consider only frames that were in frames_sorted (respecting MAX_FRAMES)
        frames_sorted = sorted([t for t in total_by_t.keys() if (params.MAX_FRAMES is None or t <= max(0, int(params.MAX_FRAMES) - 1))])
        cand_counts = _np.asarray([int(total_by_t.get(t, 0)) for t in frames_sorted], dtype=_np.int32)
        pos_counts = _np.asarray([int(pos_by_t.get(t, 0)) for t in frames_sorted], dtype=_np.int32)
        if cand_counts.size:
            c_p25 = float(_np.percentile(cand_counts, 25))
            c_p50 = float(_np.percentile(cand_counts, 50))
            c_p75 = float(_np.percentile(cand_counts, 75))
            print(
                f"Stage2  Candidates/frame: min={int(cand_counts.min())} p25={c_p25:.1f} p50={c_p50:.1f} "
                f"mean={cand_counts.mean():.2f} p75={c_p75:.1f} max={int(cand_counts.max())}"
            )
            nz = [(t, int(c)) for t, c in zip(frames_sorted, cand_counts.tolist()) if c > 0]
            nz.sort(key=lambda x: (-x[1], x[0]))
            top = nz[:5]
            if top:
                top_s = ", ".join([f"t={t}:{c}" for (t, c) in top])
                print(f"Stage2  Top frames by candidates: {top_s}")
        if pos_counts.size:
            p_p25 = float(_np.percentile(pos_counts, 25))
            p_p50 = float(_np.percentile(pos_counts, 50))
            p_p75 = float(_np.percentile(pos_counts, 75))
            print(
                f"Stage2  Positives/frame: min={int(pos_counts.min())} p25={p_p25:.1f} p50={p_p50:.1f} "
                f"mean={pos_counts.mean():.2f} p75={p_p75:.1f} max={int(pos_counts.max())}"
            )
            nzp = [(t, int(c)) for t, c in zip(frames_sorted, pos_counts.tolist()) if c > 0]
            nzp.sort(key=lambda x: (-x[1], x[0]))
            top_p = nzp[:5]
            if top_p:
                top_ps = ", ".join([f"t={t}:{c}" for (t, c) in top_p])
                print(f"Stage2  Top frames by positives: {top_ps}")
        # Frames with any positives/none
        frames_with_pos = int((_np.asarray([pos_by_t.get(t, 0) for t in frames_sorted]) > 0).sum()) if frames_sorted else 0
        frames_without_pos = max(0, len(frames_sorted) - frames_with_pos)
        print(f"Stage2  Frames with positives: {frames_with_pos}; without: {frames_without_pos}")
    except Exception:
        pass
    print(f"Stage2  Wrote patches CSV → {out_csv}")
    return out_csv


__all__ = ["run_for_video"]
