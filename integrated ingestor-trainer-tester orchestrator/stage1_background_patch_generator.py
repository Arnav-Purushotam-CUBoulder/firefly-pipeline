#!/usr/bin/env python3
"""
Stage 1 (Background Patch Generator)
====================================

This module is a programmatic adaptation of:
  tools/v3 daytime pipeline negative patch generator.py

It provides helpers for:
  - preprocessing a frame (grayscale) for blob detection
  - detecting candidate "background" blob centers (hard negatives)
  - selecting a bounded number of patch centers per frame

It does NOT write patches to disk directly; the caller (e.g. species_scaler)
should turn the centers into fixed-size crops using its own crop logic so the
resulting filenames and patch_locations CSVs stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2  # type: ignore
import numpy as np


@dataclass(frozen=True)
class BlobDetectorConfig:
    # Mirror defaults used by tools/v3 daytime pipeline negative patch generator.py
    min_area_px: float = 0.5
    max_area_scale: float = 1.0  # fraction of frame area allowed (<=0 => no upper bound)
    min_dist_px: float = 0.25
    min_repeat: int = 1
    # Performance knobs:
    # SimpleBlobDetector thresholds from min_threshold..max_threshold in steps of threshold_step.
    # Smaller steps are slower on high-res frames.
    min_threshold: int = 0
    max_threshold: int = 255
    threshold_step: int = 10


@dataclass(frozen=True)
class PreprocessConfig:
    # Mirror defaults used by tools/v3 daytime pipeline negative patch generator.py
    use_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_tile: Tuple[int, int] = (8, 8)

    use_tophat: bool = False
    tophat_ksize: int = 7

    use_dog: bool = False
    dog_sigma1: float = 0.8
    dog_sigma2: float = 1.6
    # Downscale frames before blob detection (speeds up large videos a lot).
    # None disables downscaling.
    downscale_max_dim: int | None = 960


# Best-effort caches (avoid re-creating OpenCV objects per frame)
_DETECTOR_CACHE: dict[tuple, object] = {}
_CLAHE_CACHE: dict[tuple, object] = {}


def _make_blob_detector(*, cfg: BlobDetectorConfig, frame_w: int, frame_h: int):
    area = float(max(1, int(frame_w) * int(frame_h)))
    max_area = float(area * cfg.max_area_scale) if float(cfg.max_area_scale) > 0.0 else float(area)

    p = cv2.SimpleBlobDetector_Params()

    p.filterByColor = False

    p.filterByArea = True
    p.minArea = float(cfg.min_area_px)
    p.maxArea = float(max_area)

    p.filterByCircularity = False
    p.filterByConvexity = False
    p.filterByInertia = False

    p.minThreshold = int(cfg.min_threshold)
    p.maxThreshold = int(cfg.max_threshold)
    p.thresholdStep = max(1, int(cfg.threshold_step))
    p.minRepeatability = int(cfg.min_repeat)
    p.minDistBetweenBlobs = float(cfg.min_dist_px)
    return cv2.SimpleBlobDetector_create(p)


def _preprocess(gray_u8: np.ndarray, *, cfg: PreprocessConfig) -> np.ndarray:
    inp = gray_u8
    if cfg.use_clahe:
        key = (float(cfg.clahe_clip), int(cfg.clahe_tile[0]), int(cfg.clahe_tile[1]))
        clahe = _CLAHE_CACHE.get(key)
        if clahe is None:
            clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip), tileGridSize=tuple(cfg.clahe_tile))
            _CLAHE_CACHE[key] = clahe
        inp = clahe.apply(inp)
    if cfg.use_tophat:
        ksize = int(cfg.tophat_ksize)
        if ksize < 3 or ksize % 2 == 0:
            ksize = 7
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        inp = cv2.morphologyEx(inp, cv2.MORPH_TOPHAT, k)
    if cfg.use_dog:
        g1 = cv2.GaussianBlur(inp, (0, 0), float(cfg.dog_sigma1))
        g2 = cv2.GaussianBlur(inp, (0, 0), float(cfg.dog_sigma2))
        inp = cv2.subtract(g1, g2)
    return inp


def find_blob_centers(
    frame_bgr: np.ndarray,
    *,
    blob_cfg: BlobDetectorConfig = BlobDetectorConfig(),
    pre_cfg: PreprocessConfig = PreprocessConfig(),
) -> List[Tuple[int, int]]:
    if frame_bgr is None:
        return []
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return []

    scale = 1.0
    max_dim = pre_cfg.downscale_max_dim
    if max_dim is not None:
        try:
            max_dim_i = int(max_dim)
        except Exception:
            max_dim_i = 0
        if max_dim_i > 0 and max(h, w) > max_dim_i:
            scale = float(max_dim_i) / float(max(h, w))
            nh = max(1, int(round(float(h) * scale)))
            nw = max(1, int(round(float(w) * scale)))
            frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    proc = _preprocess(gray, cfg=pre_cfg)

    det_key = (
        float(blob_cfg.min_area_px),
        float(blob_cfg.max_area_scale),
        float(blob_cfg.min_dist_px),
        int(blob_cfg.min_repeat),
        int(blob_cfg.min_threshold),
        int(blob_cfg.max_threshold),
        int(blob_cfg.threshold_step),
        int(w),
        int(h),
    )
    detector = _DETECTOR_CACHE.get(det_key)
    if detector is None:
        detector = _make_blob_detector(cfg=blob_cfg, frame_w=w, frame_h=h)
        _DETECTOR_CACHE[det_key] = detector

    keypoints = detector.detect(proc)
    centers: List[Tuple[int, int]] = []
    for kp in keypoints:
        cx, cy = kp.pt
        if scale != 1.0:
            cx, cy = (float(cx) / scale), (float(cy) / scale)
        centers.append((int(round(cx)), int(round(cy))))
    return centers
