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

    p.minThreshold = 0
    p.maxThreshold = 255
    p.thresholdStep = 1
    p.minRepeatability = int(cfg.min_repeat)
    p.minDistBetweenBlobs = float(cfg.min_dist_px)
    return cv2.SimpleBlobDetector_create(p)


def _preprocess(gray_u8: np.ndarray, *, cfg: PreprocessConfig) -> np.ndarray:
    inp = gray_u8
    if cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip), tileGridSize=tuple(cfg.clahe_tile))
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

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    proc = _preprocess(gray, cfg=pre_cfg)
    detector = _make_blob_detector(cfg=blob_cfg, frame_w=w, frame_h=h)
    keypoints = detector.detect(proc)
    centers: List[Tuple[int, int]] = []
    for kp in keypoints:
        cx, cy = kp.pt
        centers.append((int(round(cx)), int(round(cy))))
    return centers
