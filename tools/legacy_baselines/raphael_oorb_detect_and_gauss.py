#!/usr/bin/env python3
"""
raphael_oorb_detect_and_gauss.py
--------------------------------
Python port of Raphael Sarfati's MATLAB OOrb "tracking" detector (trk/fffabcnet.m),
limited to the *detection* portion only:

  - Green-channel EWMA background compensation
  - Foreground thresholding + connected components
  - Patch extraction with 360° horizontal wrap-around
  - Patch classification with Raphael's ffnet (TorchScript)
  - Output firefly detections as x,y,w,h,t (center semantics)
  - Optional Gaussian/intensity-weighted centroid refinement on a small crop

This does NOT perform calibration/triangulation/trajectory steps.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _log_prob(p: float, eps: float = 1e-8) -> float:
    p = float(p)
    p = min(1.0 - eps, max(eps, p))
    return float(math.log(p))


def _intensity_weighted_centroid(gray: np.ndarray) -> Tuple[float, float]:
    """Return (cx, cy) in pixel coords within the array, using intensity weights."""
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0, 0.0
    g = gray.astype(np.float32, copy=False)
    total = float(g.sum())
    if total <= 0.0:
        return float(w) / 2.0, float(h) / 2.0
    ys, xs = np.indices((h, w))
    cx = float((xs * g).sum() / total)
    cy = float((ys * g).sum() / total)
    return cx, cy


def _extract_patch_wrap_x(
    frame_bgr: np.ndarray,
    *,
    cx: int,
    cy: int,
    patch_size: int,
) -> np.ndarray:
    """Extract a square patch centered on (cx,cy). Wrap horizontally, clamp vertically."""
    h, w = frame_bgr.shape[:2]
    s = int(patch_size)
    if s <= 0:
        raise ValueError("patch_size must be > 0")

    x0 = int(cx) - (s // 2)
    y0 = int(cy) - (s // 2)

    cols = (np.arange(x0, x0 + s) % max(1, w)).astype(np.int64)
    rows = np.clip(np.arange(y0, y0 + s), 0, max(0, h - 1)).astype(np.int64)
    patch = frame_bgr[rows[:, None], cols[None, :], :]
    return patch


def _torch_load_model(path: Path):
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to run Raphael's ffnet TorchScript model. "
            "Install torch (and optionally torchvision)."
        ) from e

    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return model


def _torch_device(device: str):
    import torch  # type: ignore

    device = (device or "").strip().lower()
    if device in {"", "auto"}:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _classify_patches(
    model,
    device,
    patches_bgr: List[np.ndarray],
) -> np.ndarray:
    """
    Returns flash probabilities for each patch, assuming model outputs 2-class logits
    with ordering [background, flash] (as in firefl-eye-net).
    """
    import torch  # type: ignore

    tens = []
    for p in patches_bgr:
        rgb = p[:, :, ::-1]  # BGR -> RGB
        rgb = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1)  # 3,H,W
        t = t * 2.0 - 1.0  # Normalize like (x-0.5)/0.5
        tens.append(t)

    batch = torch.stack(tens, dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        flash = probs[:, 1].detach().cpu().numpy()
    return flash.astype(np.float32, copy=False)


def detect_fireflies_raphael(
    *,
    video_path: Path,
    model_path: Path,
    out_pred_csv: Path,
    out_raw_csv: Path | None,
    out_gauss_csv: Path | None,
    bw_thr: float,
    classify_thr: float,
    bkgr_window_sec: float,
    blur_sigma: float,
    patch_size: int,
    batch_size: int,
    gauss_crop_size: int,
    max_frames: int | None,
    device: str,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = total_frames
    if max_frames is not None:
        limit = min(limit, int(max_frames))

    bkgr_stack_frames = max(1, int(round(float(bkgr_window_sec) * fps)))
    alpha = 1.0 / float(bkgr_stack_frames)

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError(f"Could not read any frames from: {video_path}")

    bkgr = frame0[:, :, 1].astype(np.float32, copy=True)  # green channel
    frame_idx = 1

    thr_val = int(round(float(bw_thr) * 255.0)) if float(bw_thr) <= 1.0 else int(round(float(bw_thr)))
    thr_val = max(0, min(255, thr_val))

    model = _torch_load_model(model_path)
    dev = _torch_device(device)
    try:
        import torch  # type: ignore
        model = model.to(dev)
    except Exception:
        pass

    raw_dets: List[Dict[str, float]] = []
    batch_patches: List[np.ndarray] = []
    batch_meta: List[Tuple[int, int, int]] = []  # (t, cx, cy)

    t_start = time.time()
    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        g = frame[:, :, 1].astype(np.float32, copy=False)
        diff = g - bkgr
        diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)

        # Update EWMA background (Raphael: bkgr = (1-a)*bkgr + a*newFrame)
        bkgr = (1.0 - alpha) * bkgr + alpha * g

        if float(blur_sigma) > 0.2:
            diff_u8 = cv2.GaussianBlur(diff_u8, (0, 0), float(blur_sigma))

        _, bw = cv2.threshold(diff_u8, thr_val, 255, cv2.THRESH_BINARY)
        num, _, _, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)

        if num > 1:
            # cents: (num,2) float64, index 0 is background
            for cx_f, cy_f in cents[1:]:
                cx = int(round(float(cx_f)))
                cy = int(round(float(cy_f)))
                patch = _extract_patch_wrap_x(frame, cx=cx, cy=cy, patch_size=int(patch_size))
                batch_patches.append(patch)
                batch_meta.append((frame_idx, cx, cy))

        # Classify in batches (MATLAB patchBatch=1000)
        if len(batch_patches) >= int(batch_size):
            flash_probs = _classify_patches(model, dev, batch_patches)
            for (t, cx, cy), p in zip(batch_meta, flash_probs):
                if float(p) >= float(classify_thr):
                    raw_dets.append({"t": float(t), "x": float(cx), "y": float(cy), "conf": float(p)})
            batch_patches.clear()
            batch_meta.clear()

        frame_idx += 1

    cap.release()

    # Flush any remaining patches
    if batch_patches:
        flash_probs = _classify_patches(model, dev, batch_patches)
        for (t, cx, cy), p in zip(batch_meta, flash_probs):
            if float(p) >= float(classify_thr):
                raw_dets.append({"t": float(t), "x": float(cx), "y": float(cy), "conf": float(p)})
        batch_patches.clear()
        batch_meta.clear()

    # Second pass: Gaussian/intensity-weighted centroid refinement
    dets_by_t: Dict[int, List[int]] = defaultdict(list)
    for i, d in enumerate(raw_dets):
        dets_by_t[int(d["t"])].append(i)

    gauss_dets: List[Dict[str, float]] = []
    if raw_dets:
        cap2 = cv2.VideoCapture(str(video_path))
        if not cap2.isOpened():
            raise RuntimeError(f"Could not re-open video for centroid refinement: {video_path}")

        crop = int(gauss_crop_size)
        crop = max(3, crop)
        if crop % 2 != 0:
            # Make even to match historical 10×10 centroid crops.
            crop += 1
        half = crop // 2

        fr = 0
        while fr < limit:
            ok, frame = cap2.read()
            if not ok or frame is None:
                break

            idxs = dets_by_t.get(fr, [])
            if idxs:
                H, W = frame.shape[:2]
                for i in idxs:
                    d = raw_dets[i]
                    cx = int(round(float(d["x"])))
                    cy = int(round(float(d["y"])))

                    x0 = max(0, min(cx - half, W - crop))
                    y0 = max(0, min(cy - half, H - crop))
                    patch = frame[y0 : y0 + crop, x0 : x0 + crop]
                    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    cx_rel, cy_rel = _intensity_weighted_centroid(gray)
                    gx = int(round(float(x0) + float(cx_rel)))
                    gy = int(round(float(y0) + float(cy_rel)))
                    gauss_dets.append({"t": float(fr), "x": float(gx), "y": float(gy), "conf": float(d["conf"])})

            fr += 1

        cap2.release()

    # Write outputs
    out_pred_csv = Path(out_pred_csv)
    out_pred_csv.parent.mkdir(parents=True, exist_ok=True)

    # Optional debug CSVs
    if out_raw_csv is not None:
        out_raw_csv = Path(out_raw_csv)
        out_raw_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_raw_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["x", "y", "w", "h", "t", "firefly_confidence"])
            w.writeheader()
            for d in raw_dets:
                w.writerow(
                    {
                        "x": int(round(d["x"])),
                        "y": int(round(d["y"])),
                        "w": int(gauss_crop_size),
                        "h": int(gauss_crop_size),
                        "t": int(round(d["t"])),
                        "firefly_confidence": float(d["conf"]),
                    }
                )

    if out_gauss_csv is not None:
        out_gauss_csv = Path(out_gauss_csv)
        out_gauss_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_gauss_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["x", "y", "t", "firefly_confidence"])
            w.writeheader()
            for d in gauss_dets:
                w.writerow(
                    {
                        "x": int(round(d["x"])),
                        "y": int(round(d["y"])),
                        "t": int(round(d["t"])),
                        "firefly_confidence": float(d["conf"]),
                    }
                )

    # Final predictions CSV for Stage-5 validator
    fieldnames = [
        "x",
        "y",
        "w",
        "h",
        "t",
        "class",
        "xy_semantics",
        "firefly_logit",
        "background_logit",
        "firefly_confidence",
    ]
    with out_pred_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in gauss_dets:
            conf = float(d["conf"])
            w.writerow(
                {
                    "x": float(d["x"]),
                    "y": float(d["y"]),
                    "w": int(gauss_crop_size),
                    "h": int(gauss_crop_size),
                    "t": int(round(d["t"])),
                    "class": "firefly",
                    "xy_semantics": "center",
                    "firefly_logit": _log_prob(conf),
                    "background_logit": _log_prob(1.0 - conf),
                    "firefly_confidence": conf,
                }
            )

    dt = time.time() - t_start
    print(f"[raphael] Video: {video_path}")
    print(f"[raphael] Frames processed: {frame_idx}/{limit}  bkgr_stack_frames={bkgr_stack_frames}  thr={thr_val}")
    print(f"[raphael] Raw detections: {len(raw_dets)}  Gauss detections: {len(gauss_dets)}  dt={dt:.1f}s")
    print(f"[raphael] Predictions CSV → {out_pred_csv}")
    if out_raw_csv is not None:
        print(f"[raphael] Raw detections CSV → {out_raw_csv}")
    if out_gauss_csv is not None:
        print(f"[raphael] Gauss centroids CSV → {out_gauss_csv}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raphael OOrb detection (Python port) + gaussian centroids.")
    p.add_argument("--video", type=str, required=True, help="Input video (mp4/avi/mov/mkv).")
    p.add_argument("--model", type=str, required=True, help="TorchScript ffnet model (.pth/.pt).")
    p.add_argument("--out-csv", type=str, required=True, help="Output predictions CSV (Stage-5 schema).")
    p.add_argument("--raw-csv", type=str, default="", help="Optional raw (pre-gauss) detections CSV.")
    p.add_argument("--gauss-csv", type=str, default="", help="Optional gaussian-centroid CSV.")

    # Parameters based on oorb/+prm/set.m defaults
    p.add_argument("--bw-thr", type=float, default=0.2, help="Foreground binarization threshold (0..1).")
    p.add_argument("--classify-thr", type=float, default=0.98, help="Minimum flash probability to keep.")
    p.add_argument("--bkgr-window-sec", type=float, default=2.0, help="EWMA background time constant (sec).")
    p.add_argument("--blur-sigma", type=float, default=0.0, help="Gaussian blur sigma on foreground (px).")
    p.add_argument("--patch-size", type=int, default=33, help="Patch size (px) for ffnet input.")
    p.add_argument("--batch-size", type=int, default=1000, help="Batch size for model inference.")
    p.add_argument("--gauss-crop-size", type=int, default=10, help="Crop size (px) for centroid refinement.")
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames processed.")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    return p.parse_args()


def main() -> int:
    a = _parse_args()
    video = Path(a.video).expanduser().resolve()
    model = Path(a.model).expanduser().resolve()
    out_csv = Path(a.out_csv).expanduser().resolve()
    raw_csv = Path(a.raw_csv).expanduser().resolve() if str(a.raw_csv).strip() else None
    gauss_csv = Path(a.gauss_csv).expanduser().resolve() if str(a.gauss_csv).strip() else None

    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    detect_fireflies_raphael(
        video_path=video,
        model_path=model,
        out_pred_csv=out_csv,
        out_raw_csv=raw_csv,
        out_gauss_csv=gauss_csv,
        bw_thr=float(a.bw_thr),
        classify_thr=float(a.classify_thr),
        bkgr_window_sec=float(a.bkgr_window_sec),
        blur_sigma=float(a.blur_sigma),
        patch_size=int(a.patch_size),
        batch_size=int(a.batch_size),
        gauss_crop_size=int(a.gauss_crop_size),
        max_frames=int(a.max_frames) if a.max_frames is not None else None,
        device=str(a.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
