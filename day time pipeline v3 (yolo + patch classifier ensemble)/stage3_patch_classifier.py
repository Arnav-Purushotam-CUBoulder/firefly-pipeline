#!/usr/bin/env python3
from __future__ import annotations

"""
Stage 3: patch classifier over YOLO streak candidates.

Reads Stage 2 CSV for a video (x,y,w,h,frame_range,video_name). For each row,
it iterates over all frames in [start,end] from the original video, finds the
brightest pixel inside the (x,y,w,h) ROI, centers a PATCH_SIZE_PX × PATCH_SIZE_PX
crop there (with zero padding at borders), classifies all crops with a binary
patch model, and writes positives to a CSV under STAGE3_DIR:

  frame_idx, video_name, x, y, w, h, conf, det_id

Additionally saves each positive crop as a small image under:
  STAGE3_DIR/<video_stem>/crops/positives/
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

import params


_MODEL_CACHE: dict[str, object] = {"model": None, "device": None, "warmed": False}


def _open_video(path: Path):
    assert path.exists(), f"Input not found: {path}"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, w, h, fps, count


def _progress(i: int, total: int, tag: str = "") -> None:
    total = max(int(total or 1), 1)
    i = min(int(i), total)
    bar_w = 36
    frac = i / total
    fill = int(frac * bar_w)
    bar = "█" * fill + "·" * (bar_w - fill)
    print(f"\r[{bar}] {i}/{total} {tag}", end="")
    if i == total:
        print("")


def _device():
    try:
        import torch

        pref = str(getattr(params, "STAGE3_DEVICE", "auto")).lower()
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pref == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        # auto: prefer CUDA → MPS → CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for Stage 3. Install torch/torchvision.") from e


def _to_tensor_batch(np_list: List[np.ndarray], img_size: int):
    """Convert list of RGB uint8 HxWx3 arrays to a torch batch."""
    import torch

    out = []
    do_norm = bool(getattr(params, "IMAGENET_NORMALIZE", False))
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

    if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["device"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["device"]

    dev = _device()
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)

    ckpt = torch.load(str(model_path), map_location=dev)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = m.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        print(f"Stage3  Warning: load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    m.eval().to(dev)

    # Warm-up once to reduce first-batch latency
    if not _MODEL_CACHE["warmed"]:
        with torch.inference_mode():
            dummy = torch.zeros(
                1,
                3,
                int(getattr(params, "STAGE3_INPUT_SIZE", 10)),
                int(getattr(params, "STAGE3_INPUT_SIZE", 10)),
                device=dev,
            )
            _ = m(dummy)
        _MODEL_CACHE["warmed"] = True

    _MODEL_CACHE["model"] = m
    _MODEL_CACHE["device"] = dev
    return m, dev


def _read_stage2_boxes(stem: str) -> List[dict]:
    """Read Stage 2 CSV: x,y,w,h,frame_range,video_name."""
    s2_csv = (params.STAGE2_DIR / stem) / f"{stem}.csv"
    rows: List[dict] = []
    with s2_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _by_frame_from_stage2(rows: List[dict]) -> Dict[int, List[Tuple[int, int, int, int, int]]]:
    """Group Stage 2 boxes by frame index.

    Returns mapping: frame_idx -> list of (det_id, x, y, w, h).
    Each Stage 2 record contributes the same ROI to all frames in its frame_range.
    """
    out: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    for det_id, row in enumerate(rows):
        try:
            x = int(float(row["x"]))
            y = int(float(row["y"]))
            w = int(float(row["w"]))
            h = int(float(row["h"]))
            fr = str(row["frame_range"])
            if "-" not in fr:
                continue
            s_str, e_str = fr.split("-", 1)
            start = int(s_str)
            end = int(e_str)
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        if end < start:
            start, end = end, start
        for t in range(start, end + 1):
            out.setdefault(t, []).append((det_id, x, y, w, h))
    return out


def _center_crop_with_pad(
    img: np.ndarray, cx: float, cy: float, size_w: int, size_h: int
) -> Tuple[np.ndarray, int, int]:
    """Crop size_w x size_h around (cx,cy) with black padding at borders."""
    H, W = img.shape[:2]
    w = max(1, int(size_w))
    h = max(1, int(size_h))
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x1 = x0 + w
    y1 = y0 + h

    vx0 = max(0, x0)
    vy0 = max(0, y0)
    vx1 = min(W, x1)
    vy1 = min(H, y1)

    px0 = vx0 - x0
    py0 = vy0 - y0

    if img.ndim == 3:
        patch = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
        patch[py0 : py0 + (vy1 - vy0), px0 : px0 + (vx1 - vx0), :] = img[vy0:vy1, vx0:vx1, :]
    else:
        patch = np.zeros((h, w), dtype=img.dtype)
        patch[py0 : py0 + (vy1 - vy0), px0 : px0 + (vx1 - vx0)] = img[vy0:vy1, vx0:vx1]
    return patch, x0, y0


def run_for_video(video_path: Path) -> Path:
    stem = video_path.stem
    s2_rows = _read_stage2_boxes(stem)

    out_root = params.STAGE3_DIR / stem
    pos_dir = out_root / "crops" / "positives"
    pos_dir.mkdir(parents=True, exist_ok=True)

    cap, W, H, fps, total = _open_video(video_path)

    by_frame = _by_frame_from_stage2(s2_rows)
    if not by_frame:
        print("Stage3  NOTE: No Stage 2 candidates found.")
        out_csv = out_root / f"{stem}_patches.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame_idx", "video_name", "x", "y", "w", "h", "conf", "det_id"],
            )
            writer.writeheader()
        cap.release()
        return out_csv

    # Load model
    model, dev = _load_model(params.PATCH_MODEL_PATH)
    import torch

    out_csv = out_root / f"{stem}_patches.csv"
    total_patches = 0
    pos_count = 0
    neg_count = 0
    pos_probs_all: List[float] = []

    total_by_t = {t: len(by_frame[t]) for t in by_frame.keys()}
    pos_by_t: Dict[int, int] = {t: 0 for t in by_frame.keys()}

    frames_sorted = sorted(by_frame.keys())
    if params.MAX_FRAMES is not None:
        max_idx = max(0, int(params.MAX_FRAMES) - 1)
        frames_sorted = [t for t in frames_sorted if t <= max_idx]
    # Clamp to actual video length
    frames_sorted = [t for t in frames_sorted if t < total]

    total_candidates = sum(len(by_frame[t]) for t in frames_sorted)
    if not frames_sorted or total_candidates == 0:
        print("Stage3  NOTE: No frames/candidates to process (possibly due to MAX_FRAMES).")
        print(f"Stage3  Candidates from Stage2: {total_candidates}; frames_in_range=0")
        out_csv = out_root / f"{stem}_patches.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame_idx", "video_name", "x", "y", "w", "h", "conf", "det_id"],
            )
            writer.writeheader()
        cap.release()
        print("Stage3  Summary: patches=0, positives=0, negatives=0")
        print(f"Stage3  Wrote patches CSV → {out_csv}")
        return out_csv

    batch_gpu = int(getattr(params, "STAGE3_BATCH_SIZE_GPU", 4096))
    batch_cpu = int(getattr(params, "STAGE3_BATCH_SIZE_CPU", 512))
    batch_size = batch_gpu if dev.type in {"cuda", "mps"} else batch_cpu

    cur_batch: List[Tuple[int, int, int, int, int, np.ndarray, int]] = []  # (t,x,y,w,h,RGB,det_id)
    processed = 0

    import time as _time

    _t0 = _time.perf_counter()
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_idx", "video_name", "x", "y", "w", "h", "conf", "det_id"],
        )
        writer.writeheader()

        def _flush_batch():
            nonlocal cur_batch, pos_count, neg_count, total_patches, processed, pos_probs_all, pos_by_t
            if not cur_batch:
                return
            imgs = [b[5] for b in cur_batch]
            x_t = _to_tensor_batch(imgs, int(getattr(params, "STAGE3_INPUT_SIZE", 10))).to(dev)
            with torch.inference_mode():
                logits = model(x_t)
                prob = torch.softmax(
                    logits, dim=1
                )[:, int(getattr(params, "STAGE3_POSITIVE_CLASS_INDEX", 1))]
            thr = float(getattr(params, "STAGE3_POSITIVE_THRESHOLD", 0.5))
            preds = (prob >= thr).cpu().numpy().astype(bool)
            probs_np = prob.detach().cpu().numpy()
            pos_probs_all.extend(probs_np.tolist())

            for (t, x0, y0, w, h, patch_rgb, det_id), is_pos, p_conf in zip(
                cur_batch, preds, probs_np
            ):
                if is_pos:
                    out_name = (
                        f"f_{int(t):06d}_x{int(x0)}_y{int(y0)}_"
                        f"w{int(w)}_h{int(h)}_p{float(p_conf):.3f}.png"
                    )
                    # patch_rgb is RGB for the model; convert back to BGR for writing
                    cv2.imwrite(str(pos_dir / out_name), cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
                    writer.writerow(
                        {
                            "frame_idx": int(t),
                            "video_name": video_path.name,
                            "x": int(x0),
                            "y": int(y0),
                            "w": int(w),
                            "h": int(h),
                            "conf": float(p_conf),
                            "det_id": int(det_id),
                        }
                    )
                    pos_count += 1
                    pos_by_t[int(t)] = pos_by_t.get(int(t), 0) + 1
                else:
                    neg_count += 1
            processed += len(cur_batch)
            total_patches += len(cur_batch)
            cur_batch.clear()
            _progress(processed, max(1, total_candidates), "Stage3 classify")

        for t in frames_sorted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(t))
            ok, frame = cap.read()
            if not ok:
                continue
            Hf, Wf = frame.shape[:2]
            for (det_id, x, y, w, h) in by_frame[t]:
                x0 = max(0, int(x))
                y0 = max(0, int(y))
                x1 = min(Wf, int(x + w))
                y1 = min(Hf, int(y + h))
                if x1 <= x0 or y1 <= y0:
                    continue
                roi = frame[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                mean_roi = roi.mean(axis=2)
                flat_idx = int(np.argmax(mean_roi))
                roi_w = x1 - x0
                bx = int(flat_idx % roi_w)
                by = int(flat_idx // roi_w)
                cx = x0 + bx
                cy = y0 + by
                patch_bgr, px0, py0 = _center_crop_with_pad(
                    frame, cx, cy, int(getattr(params, "PATCH_SIZE_PX", 10)), int(getattr(params, "PATCH_SIZE_PX", 10))
                )
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                patch_rgb = np.ascontiguousarray(patch_rgb)
                cur_batch.append(
                    (int(t), int(px0), int(py0), int(patch_bgr.shape[1]), int(patch_bgr.shape[0]), patch_rgb, int(det_id))
                )
                if len(cur_batch) >= batch_size:
                    _flush_batch()

        _flush_batch()
        _progress(total_patches, max(1, total_candidates), "Stage3 classify done")

    cap.release()
    elapsed = _time.perf_counter() - _t0
    frames_processed = len(frames_sorted)
    pos_rate = (pos_count / total_patches) if total_patches else 0.0
    avg_patches_per_frame = (total_patches / frames_processed) if frames_processed else 0.0
    avg_pos_per_frame = (pos_count / frames_processed) if frames_processed else 0.0
    throughput = (total_patches / elapsed) if elapsed > 0 else 0.0

    print(
        f"Stage3  Config: input_size={getattr(params,'STAGE3_INPUT_SIZE',None)} "
        f"normalize={getattr(params,'IMAGENET_NORMALIZE',None)} "
        f"pos_idx={getattr(params,'STAGE3_POSITIVE_CLASS_INDEX',None)} "
        f"thr={getattr(params,'STAGE3_POSITIVE_THRESHOLD',None)} "
        f"batch_gpu={getattr(params,'STAGE3_BATCH_SIZE_GPU',None)} "
        f"batch_cpu={getattr(params,'STAGE3_BATCH_SIZE_CPU',None)}"
    )
    if pos_probs_all:
        import numpy as _np

        arr = _np.asarray(pos_probs_all, dtype=_np.float32)
        print(
            f"Stage3  Probs summary (positive class): min={arr.min():.3f} "
            f"p25={_np.percentile(arr,25):.3f} mean={arr.mean():.3f} "
            f"p75={_np.percentile(arr,75):.3f} max={arr.max():.3f}"
        )
    print(f"Stage3  Candidates (patches): {total_patches}")
    print(f"Stage3  Frames processed: {frames_processed}")
    print(f"Stage3  Time: {elapsed:.2f}s; Throughput: {throughput:.1f} patches/s")
    print(
        f"Stage3  Summary: patches={total_patches}, positives={pos_count}, negatives={neg_count}"
    )
    print(
        f"Stage3  Rates: pos_rate={pos_rate:.3f}, "
        f"avg_patches/frame={avg_patches_per_frame:.1f}, "
        f"avg_pos/frame={avg_pos_per_frame:.2f}"
    )

    # Per-frame stats
    try:
        import numpy as _np

        frames_stats = sorted(
            [t for t in total_by_t.keys() if (params.MAX_FRAMES is None or t <= max(0, int(params.MAX_FRAMES) - 1))]
        )
        cand_counts = _np.asarray(
            [int(total_by_t.get(t, 0)) for t in frames_stats], dtype=_np.int32
        )
        pos_counts = _np.asarray(
            [int(pos_by_t.get(t, 0)) for t in frames_stats], dtype=_np.int32
        )
        if cand_counts.size:
            c_p25 = float(_np.percentile(cand_counts, 25))
            c_p50 = float(_np.percentile(cand_counts, 50))
            c_p75 = float(_np.percentile(cand_counts, 75))
            print(
                f"Stage3  Candidates/frame: min={int(cand_counts.min())} "
                f"p25={c_p25:.1f} p50={c_p50:.1f} mean={cand_counts.mean():.2f} "
                f"p75={c_p75:.1f} max={int(cand_counts.max())}"
            )
        if pos_counts.size:
            p_p25 = float(_np.percentile(pos_counts, 25))
            p_p50 = float(_np.percentile(pos_counts, 50))
            p_p75 = float(_np.percentile(pos_counts, 75))
            print(
                f"Stage3  Positives/frame: min={int(pos_counts.min())} "
                f"p25={p_p25:.1f} p50={p_p50:.1f} mean={pos_counts.mean():.2f} "
                f"p75={p_p75:.1f} max={int(pos_counts.max())}"
            )
            nzp = [(t, int(c)) for t, c in zip(frames_stats, pos_counts.tolist()) if c > 0]
            nzp.sort(key=lambda x: (-x[1], x[0]))
            top_p = nzp[:5]
            if top_p:
                top_ps = ", ".join([f"t={t}:{c}" for (t, c) in top_p])
                print(f"Stage3  Top frames by positives: {top_ps}")
        frames_with_pos = int(
            (_np.asarray([pos_by_t.get(t, 0) for t in frames_stats]) > 0).sum()
        ) if frames_stats else 0
        frames_without_pos = max(0, len(frames_stats) - frames_with_pos)
        print(f"Stage3  Frames with positives: {frames_with_pos}; without: {frames_without_pos}")
    except Exception:
        pass

    print(f"Stage3  Wrote patches CSV → {out_csv}")
    return out_csv


__all__ = ["run_for_video"]

