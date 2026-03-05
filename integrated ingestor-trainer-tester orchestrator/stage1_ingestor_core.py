#!/usr/bin/env python3
"""
Stage 1 (Ingestor Core)
=======================

This is the core ingestion stage used by `stage1_ingestor.py`.

Ingest a new annotator CSV batch and automatically:
  1) Split FIREFLY rows into TRAIN vs FINAL-VALIDATION (pipeline-heldout).
     Background rows are TRAIN-only (no pipeline validation).
  2) Extract TRAIN patches from the input video (firefly + background).
  3) Version + merge the TRAIN data into:
       - Integrated_prototype_datasets/single species datasets/<species>/vN_DATE/...
       - Integrated_prototype_datasets/integrated pipeline datasets/vN_DATE/...
  4) Version + merge the FINAL-VALIDATION rows (CSV-only) into:
       - Integrated_prototype_validation_datasets/combined species folder/vN_DATE/annotations.csv
       - Integrated_prototype_validation_datasets/individual species folder/<species>/vN_DATE/annotations.csv
  5) Print before/after per-species stats.

Annotator batch expectations
----------------------------
- Filename encodes (preferred):
    <video_name>_<species_name>_<day_time|night_time>_<firefly|background>.csv
  Back-compat:
    <video_name>_<species_name>_<day_time|night_time>.csv   (treated as firefly)
- CSV columns: x,y,w,h,t  (or x,y,w,h,frame)
  where t/frame is a 0-based frame index into the input video.
"""

from __future__ import annotations

import csv
import errno
import os
import random
import re
import shutil
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = e
else:
    _CV2_IMPORT_ERROR = None


# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG (edit these)
# ──────────────────────────────────────────────────────────────────────────────

# Incoming annotator CSV + corresponding input video.
# Preferred naming:
#   <video_name>_<species_name>_<day_time|night_time>_<firefly|background>.csv
ANNOTATIONS_CSV_PATH: Path = Path("/Users/arnavps/Desktop/20240606_cam1_GS010064_pyrallisGoPro_day_time_firefly.csv")
VIDEO_PATH: Path = Path('/Users/arnavps/Desktop/RA inference data/v3 daytime pipeline inference data/20240606_cam1_GS010064/original videos/20240606_cam1_GS010064.mp4')

# 0..1: fraction of rows from the incoming CSV to use for TRAIN data.
# Remaining rows go to FINAL-VALIDATION (CSV-only; never used for training).
TRAIN_FRACTION_OF_BATCH: float = 0.80
TRAIN_VAL_SPLIT_SEED: int = 1337

# If True, and you pass e.g. "..._firefly.csv", the script will also look for
# "..._background.csv" in the same folder and ingest it in the same run (and
# vice-versa). This avoids creating two dataset versions for one batch.
AUTO_LOAD_SIBLING_CLASS_CSV: bool = True

# Where to export this run’s intermediate artifacts (train patches + split CSVs)
# Set this to the *single* root folder that contains:
#   - batch_exports/
#   - Integrated_prototype_datasets/
#   - Integrated_prototype_validation_datasets/
#
# You can set this to a string path safely (it will be converted to Path).
DATA_ROOT: str | Path = '/Volumes/DL Project SSD/integrated prototype data/patch training datasets and pipeline validation data'
DATA_ROOT = Path(str(DATA_ROOT)).expanduser()

# Training dataset roots (folder structure is created by commands once; script
# will create missing subfolders/version folders as needed).
BATCH_EXPORT_ROOT: Path = DATA_ROOT / "batch_exports"
TRAIN_DATASETS_ROOT: Path = DATA_ROOT / "Integrated_prototype_datasets"
VALIDATION_DATASETS_ROOT: Path = DATA_ROOT / "Integrated_prototype_validation_datasets"

# Advanced orchestration:
# If these are set, the scaler will *append* into the provided version folders
# instead of creating a new version per-ingested video. This enables a single
# dataset version for a multi-video batch ingest driven by the orchestrator.
SINGLE_SPECIES_TARGET_VERSION_DIR: Path | None = None
INTEGRATED_TARGET_VERSION_DIR: Path | None = None
VALIDATION_COMBINED_TARGET_VERSION_DIR: Path | None = None
VALIDATION_INDIVIDUAL_TARGET_VERSION_DIR: Path | None = None

# If True, skip rebuilding the final dataset split + split CSVs on this run.
# Useful when ingesting many videos into the same version directory; you can
# finalize once at the end of the batch.
SKIP_FINAL_SPLIT_REBUILD: bool = False

# Final dataset split (must sum to 1.0)
FINAL_TRAIN_PCT: float = 0.80
FINAL_VAL_PCT: float = 0.15
FINAL_TEST_PCT: float = 0.05
FINAL_SPLIT_SEED: int = 1337

# Copy mode when versioning datasets.
# - "copy": full copy
# - "hardlink": make hardlinks to previous version files when possible (falls back to copy on failure)
VERSION_COPY_MODE: str = "copy"

# If True, skip writing a patch if a file with the same name already exists.
# If False, auto-suffix duplicates as ...__dupN.png
SKIP_EXISTING_PATCHES: bool = True

# Auto background patch generation (when only firefly annotations exist)
# --------------------------------------------------------------------
# If True, the scaler will synthesize BACKGROUND training annotations by sampling
# frames in the video that do NOT contain any annotated firefly rows (based on the
# firefly CSV's frame indices). This allows training a 2-class patch classifier
# even when only fireflies are annotated.
AUTO_GENERATE_BACKGROUND_PATCHES: bool = False
AUTO_BACKGROUND_TO_FIREFLY_RATIO: float = 20.0  # target background count = ratio * train_firefly_count
AUTO_BACKGROUND_PATCH_SIZE_PX: int = 10
AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME: int = 10
AUTO_BACKGROUND_MAX_FRAME_SAMPLES: int = 5000  # safety cap to avoid infinite loops
AUTO_BACKGROUND_SEED: int = 1337
AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS: bool = True  # if SBD finds 0 blobs in a frame, sample random centers

# Safety/performance: blob detection can be extremely slow on high-res frames (e.g. 4K).
# If a single-frame blob detection call exceeds this threshold, we will disable blob detection
# for the rest of the video and fall back to random centers only (if enabled).
AUTO_BACKGROUND_BLOB_DETECT_SLOW_FRAME_SECONDS: float = 2.0
AUTO_BACKGROUND_BLOB_DETECT_DISABLE_AFTER_SLOW_FRAMES: int = 1

# Blob detector + preprocessing knobs for background sampling
# Mirrors tools/v3 daytime pipeline negative patch generator.py defaults.
AUTO_BACKGROUND_SBD_MIN_AREA_PX: float = 0.5
AUTO_BACKGROUND_SBD_MAX_AREA_SCALE: float = 1.0
AUTO_BACKGROUND_SBD_MIN_DIST: float = 0.25
AUTO_BACKGROUND_SBD_MIN_REPEAT: int = 1

AUTO_BACKGROUND_USE_CLAHE: bool = True
AUTO_BACKGROUND_CLAHE_CLIP: float = 2.0
AUTO_BACKGROUND_CLAHE_TILE: Tuple[int, int] = (8, 8)

AUTO_BACKGROUND_USE_TOPHAT: bool = False
AUTO_BACKGROUND_TOPHAT_KSIZE: int = 7

AUTO_BACKGROUND_USE_DOG: bool = False
AUTO_BACKGROUND_DOG_SIGMA1: float = 0.8
AUTO_BACKGROUND_DOG_SIGMA2: float = 1.6

# File naming / CSV naming
PATCH_IMAGE_EXT: str = ".png"
PATCH_LOCATIONS_CSV_NAME: str = "patch_locations.csv"
PATCH_LOCATIONS_SPLIT_PREFIX: str = "patch_locations_"  # e.g. patch_locations_train.csv
PATCH_LOCATIONS_BACKGROUND_CSV_NAME: str = "patch_locations_background.csv"
PATCH_LOCATIONS_BACKGROUND_SPLIT_PREFIX: str = "patch_locations_background_"  # e.g. patch_locations_background_train.csv
VALIDATION_CSV_NAME: str = "annotations.csv"

# Annotation coordinate semantics for x,y in annotator CSVs.
# - "center": x,y is patch center (requested dataset convention)
# - "top_left": x,y is top-left corner
ANNOTATION_XY_SEMANTICS: str = "center"


# ──────────────────────────────────────────────────────────────────────────────
# Folder-name constants (match your on-disk schema)
# ──────────────────────────────────────────────────────────────────────────────

INTEGRATED_PIPELINE_DATASETS_DIRNAME = "integrated pipeline datasets"
SINGLE_SPECIES_DATASETS_DIRNAME = "single species datasets"

DAY_DATASET_DIRNAME = "day_time_dataset"
NIGHT_DATASET_DIRNAME = "night_time_dataset"

INITIAL_DATASET_DIRNAME = "initial dataset"
FINAL_DATASET_DIRNAME = "final dataset"

CLASS_FIREFLY = "firefly"
CLASS_BACKGROUND = "background"
SPLIT_NAMES = ("train", "val", "test")

VALIDATION_COMBINED_DIRNAME = "combined species folder"
VALIDATION_INDIVIDUAL_DIRNAME = "individual species folder"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Annotation:
    x: int
    y: int
    w: int
    h: int
    t: int


def _today_tag() -> str:
    return datetime.now().strftime("%Y%m%d")


def _safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _iter_image_files(d: Path) -> Iterator[Path]:
    if not d.exists():
        return
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _count_images(d: Path) -> int:
    return sum(1 for _ in _iter_image_files(d))


def _parse_batch_identity(csv_path: Path) -> Tuple[str, str, str, str]:
    """
    Return (video_name, species_name, time_of_day, class_label) from CSV filename.

    Preferred naming (recommended):
      <video_name>_<species_name>_<day_time|night_time>_<firefly|background>.csv

    Back-compat (treated as firefly):
      <video_name>_<species_name>_<day_time|night_time>.csv
    """
    stem = csv_path.stem.strip()
    # tolerate mild variations
    # Keep hyphens so multi-word species can remain a single token, e.g.
    # "photinus-knulli" stays intact during parsing.
    stem = stem.replace("(", "_").replace(")", "_")
    stem = re.sub(r"_+", "_", stem).strip("_")

    parts = [p for p in stem.split("_") if p]
    if len(parts) < 4:
        raise ValueError(
            f"Could not parse identity from CSV name {csv_path.name!r}. "
            "Expected <video_name>_<species_name>_<day_time|night_time>_<firefly|background>."
        )

    label = parts[-1].lower()
    if label in {CLASS_FIREFLY, CLASS_BACKGROUND}:
        parts = parts[:-1]
    else:
        # old convention: no explicit label => assume firefly
        label = CLASS_FIREFLY

    if len(parts) < 3 or parts[-1].lower() != "time" or parts[-2].lower() not in {"day", "night"}:
        raise ValueError(
            f"Could not parse time_of_day from CSV name {csv_path.name!r}. "
            "Expected ..._day_time_... or ..._night_time_...."
        )
    time_of_day = f"{parts[-2].lower()}_time"

    base = parts[:-2]  # everything before day_time|night_time tokens
    if len(base) < 2:
        raise ValueError(
            f"Could not parse video/species from CSV name {csv_path.name!r}. "
            "Expected <video_name>_<species_name>_..."
        )
    species_name = base[-1]
    video_name = "_".join(base[:-1])
    return _safe_name(video_name), _safe_name(species_name), time_of_day, label


def _read_annotator_csv(csv_path: Path) -> List[Annotation]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return []

        cols = {c.strip(): c for c in reader.fieldnames if c}
        t_col = cols.get("t") or cols.get("frame") or cols.get("time")
        required = ["x", "y", "w", "h"]
        missing = [c for c in required if c not in cols]
        if missing or not t_col:
            raise ValueError(
                f"{csv_path} missing required columns. "
                f"Need x,y,w,h and t (or frame). Got: {reader.fieldnames}"
            )

        out: List[Annotation] = []
        seen: set[Tuple[int, int, int, int, int]] = set()
        for r in reader:
            try:
                x = int(round(float(r[cols["x"]])))
                y = int(round(float(r[cols["y"]])))
                w = int(round(float(r[cols["w"]])))
                h = int(round(float(r[cols["h"]])))
                t = int(round(float(r[t_col])))
            except Exception:
                continue
            if w <= 0 or h <= 0 or t < 0:
                continue
            key = (x, y, w, h, t)
            if key in seen:
                continue
            seen.add(key)
            out.append(Annotation(x=x, y=y, w=w, h=h, t=t))
        return out


def _split_annotations(
    anns: Sequence[Annotation], train_frac: float, seed: int
) -> Tuple[List[Annotation], List[Annotation]]:
    if not (0.0 <= train_frac <= 1.0):
        raise ValueError("TRAIN_FRACTION_OF_BATCH must be within [0, 1].")

    anns_shuf = list(anns)
    rng = random.Random(int(seed))
    rng.shuffle(anns_shuf)
    n = len(anns_shuf)
    n_train = int(n * float(train_frac))
    n_train = max(0, min(n, n_train))
    train = anns_shuf[:n_train]
    val = anns_shuf[n_train:]
    return train, val


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists():
        return [], []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [{k: (r.get(k) or "") for k in fieldnames} for r in reader]
        return rows, fieldnames


def _merge_csv_rows(
    existing: List[Dict[str, str]],
    new_rows: List[Dict[str, str]],
    key_fields: Sequence[str],
) -> List[Dict[str, str]]:
    out = list(existing)
    seen = {
        tuple(r.get(k, "") for k in key_fields)
        for r in existing
    }
    for r in new_rows:
        key = tuple(r.get(k, "") for k in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _crop_with_pad(frame_bgr, x: int, y: int, w: int, h: int):
    import numpy as np  # local import to avoid hard dependency if caller doesn't use extraction

    H, W = frame_bgr.shape[:2]
    xy_mode = str(ANNOTATION_XY_SEMANTICS or "center").strip().lower()
    if xy_mode == "center":
        # Convert center point to top-left for fixed-size crop extraction.
        x0 = int(round(float(x) - float(w) / 2.0))
        y0 = int(round(float(y) - float(h) / 2.0))
    elif xy_mode == "top_left":
        x0, y0 = int(x), int(y)
    else:
        raise ValueError(f"Unsupported ANNOTATION_XY_SEMANTICS={ANNOTATION_XY_SEMANTICS!r}")
    x1, y1 = x0 + int(w), y0 + int(h)
    vx0, vy0 = max(0, x0), max(0, y0)
    vx1, vy1 = min(W, x1), min(H, y1)
    patch = np.zeros((max(1, int(h)), max(1, int(w)), 3), dtype=frame_bgr.dtype)
    if vx1 > vx0 and vy1 > vy0:
        px0, py0 = vx0 - x0, vy0 - y0
        patch[py0:py0 + (vy1 - vy0), px0:px0 + (vx1 - vx0)] = frame_bgr[vy0:vy1, vx0:vx1]
    return patch


def _make_patch_filename(a: Annotation, video_name: str, species_name: str) -> str:
    video_name = _safe_name(video_name)
    species_name = _safe_name(species_name)
    return f"{a.x}_{a.y}_{a.w}_{a.h}_{a.t}_{video_name}_{species_name}{PATCH_IMAGE_EXT}"


def _parse_patch_filename(p: Path) -> Optional[Tuple[int, int, int, int, int, str, str]]:
    """
    Parse <x>_<y>_<w>_<h>_<t>_<video_name>_<species_name>.<ext>
    into (x,y,w,h,t,video_name,species_name).
    """
    parts = p.stem.split("_")
    if len(parts) < 7:
        return None
    try:
        x, y, w, h, t = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
    except Exception:
        return None
    species_name = parts[-1]
    video_name = "_".join(parts[5:-1])
    if not video_name:
        return None
    return x, y, w, h, t, video_name, species_name


def _non_firefly_frame_ranges(total_frames: int, firefly_frames: Sequence[int]) -> List[Tuple[int, int]]:
    total_frames = int(total_frames)
    if total_frames <= 0:
        return []
    forbidden = sorted({int(t) for t in firefly_frames if 0 <= int(t) < total_frames})
    out: List[Tuple[int, int]] = []
    prev = -1
    for t in forbidden:
        if t > prev + 1:
            out.append((prev + 1, t - 1))
        prev = t
    if prev < total_frames - 1:
        out.append((prev + 1, total_frames - 1))
    return out


def _sample_frame_from_ranges(ranges: Sequence[Tuple[int, int]], rng: random.Random) -> int:
    if not ranges:
        raise ValueError("ranges must be non-empty")
    lengths = [max(0, int(e) - int(s) + 1) for s, e in ranges]
    total = sum(lengths)
    if total <= 0:
        raise ValueError("ranges total length must be positive")
    pick = rng.randrange(total)
    for (s, e), L in zip(ranges, lengths):
        if pick < L:
            return int(s) + int(pick)
        pick -= L
    # Fallback (should be unreachable)
    s, e = ranges[-1]
    return int(e) if int(e) >= int(s) else int(s)


def _auto_generate_background_train_annotations(
    *,
    video_path: Path,
    firefly_anns_all: Sequence["Annotation"],
    n_needed: int,
) -> List["Annotation"]:
    """
    Generate synthetic BACKGROUND training annotations (center-x, center-y, w, h, t)
    by sampling frames that do not contain any annotated firefly frames.
    """
    if n_needed <= 0:
        return []
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) is required for auto background patch generation. "
            f"Import error: {_CV2_IMPORT_ERROR}"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for background sampling: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        # Best-effort fallback if backend can't report frame count.
        max_t = max((int(a.t) for a in firefly_anns_all), default=-1)
        total_frames = max_t + 1

    if total_frames <= 0:
        cap.release()
        return []

    firefly_frames = [int(a.t) for a in firefly_anns_all]
    ranges = _non_firefly_frame_ranges(total_frames, firefly_frames)
    if not ranges:
        print("[bg] WARNING: no non-firefly frames available; skipping auto background generation.")
        cap.release()
        return []

    # Stable per-video RNG seed (avoid Python hash randomization).
    seed = int(AUTO_BACKGROUND_SEED) ^ (zlib.crc32(str(video_path).encode("utf-8")) & 0xFFFFFFFF)
    rng = random.Random(seed)

    import stage1_background_patch_generator as bpg  # type: ignore

    blob_cfg = bpg.BlobDetectorConfig(
        min_area_px=float(AUTO_BACKGROUND_SBD_MIN_AREA_PX),
        max_area_scale=float(AUTO_BACKGROUND_SBD_MAX_AREA_SCALE),
        min_dist_px=float(AUTO_BACKGROUND_SBD_MIN_DIST),
        min_repeat=int(AUTO_BACKGROUND_SBD_MIN_REPEAT),
    )
    pre_cfg = bpg.PreprocessConfig(
        use_clahe=bool(AUTO_BACKGROUND_USE_CLAHE),
        clahe_clip=float(AUTO_BACKGROUND_CLAHE_CLIP),
        clahe_tile=tuple(AUTO_BACKGROUND_CLAHE_TILE),
        use_tophat=bool(AUTO_BACKGROUND_USE_TOPHAT),
        tophat_ksize=int(AUTO_BACKGROUND_TOPHAT_KSIZE),
        use_dog=bool(AUTO_BACKGROUND_USE_DOG),
        dog_sigma1=float(AUTO_BACKGROUND_DOG_SIGMA1),
        dog_sigma2=float(AUTO_BACKGROUND_DOG_SIGMA2),
    )

    patch_size = max(1, int(AUTO_BACKGROUND_PATCH_SIZE_PX))
    max_per_frame = max(1, int(AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME))
    max_frame_samples = max(1, int(AUTO_BACKGROUND_MAX_FRAME_SAMPLES))

    out: List[Annotation] = []
    seen: Set[Tuple[int, int, int]] = set()  # (t,x,y)
    sampled = 0
    blob_detection_enabled = True
    slow_frames = 0
    slow_s = float(AUTO_BACKGROUND_BLOB_DETECT_SLOW_FRAME_SECONDS)
    disable_after = max(1, int(AUTO_BACKGROUND_BLOB_DETECT_DISABLE_AFTER_SLOW_FRAMES))

    pbar = (
        tqdm(total=int(n_needed), desc=f"[bg] {video_path.stem}", unit="patch", dynamic_ncols=True)
        if tqdm is not None and int(n_needed) > 0
        else None
    )
    last_len = 0

    while len(out) < n_needed and sampled < max_frame_samples:
        t = _sample_frame_from_ranges(ranges, rng)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame_bgr = cap.read()
        sampled += 1
        if not ok or frame_bgr is None:
            continue

        centers = []
        if blob_detection_enabled:
            t0 = time.time()
            centers = bpg.find_blob_centers(frame_bgr, blob_cfg=blob_cfg, pre_cfg=pre_cfg)
            dt = float(time.time() - t0)
            if slow_s > 0.0 and dt >= slow_s:
                slow_frames += 1
                if slow_frames >= disable_after and bool(AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS):
                    blob_detection_enabled = False
                    print(
                        "[bg] INFO: blob detection is slow; disabling blob detector and using random centers only:",
                        f"dt={dt:.2f}s threshold={slow_s:.2f}s slow_frames={slow_frames}/{disable_after}",
                    )
        if centers:
            if len(centers) > max_per_frame:
                centers = rng.sample(centers, k=max_per_frame)
            else:
                rng.shuffle(centers)
        else:
            if not AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS:
                continue
            h, w = frame_bgr.shape[:2]
            half = patch_size // 2
            if w > patch_size:
                x_lo, x_hi = half, max(half, w - half - 1)
            else:
                x_lo, x_hi = 0, max(0, w - 1)
            if h > patch_size:
                y_lo, y_hi = half, max(half, h - half - 1)
            else:
                y_lo, y_hi = 0, max(0, h - 1)
            centers = [(rng.randint(x_lo, x_hi), rng.randint(y_lo, y_hi)) for _ in range(max_per_frame)]

        for cx, cy in centers:
            key = (int(t), int(cx), int(cy))
            if key in seen:
                continue
            seen.add(key)
            out.append(Annotation(x=int(cx), y=int(cy), w=patch_size, h=patch_size, t=int(t)))
            if len(out) >= n_needed:
                break

        if pbar is not None:
            cur_len = len(out)
            if cur_len > last_len:
                pbar.update(cur_len - last_len)
                last_len = cur_len
            pbar.set_postfix(sampled_frames=int(sampled), ranges=int(len(ranges)))
        elif sampled % 50 == 0:
            print(f"[bg] sampled_frames={sampled} generated={len(out)}/{n_needed}")

    cap.release()
    if pbar is not None:
        pbar.close()
    if len(out) < n_needed:
        print(
            f"[bg] WARNING: generated only {len(out)}/{n_needed} background patches "
            f"(sampled_frames={sampled}, cap={max_frame_samples})."
        )
    return out


def _extract_patches_from_video(
    video_path: Path,
    anns: Sequence[Annotation],
    out_dir: Path,
    video_name: str,
    species_name: str,
) -> int:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) is required for patch extraction. "
            f"Import error: {_CV2_IMPORT_ERROR}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    rows_by_t: Dict[int, List[Annotation]] = {}
    for a in anns:
        rows_by_t.setdefault(int(a.t), []).append(a)

    saved = 0
    ts = sorted(rows_by_t.keys())
    pbar = (
        tqdm(ts, desc=f"[patches] {video_path.stem}", unit="frame", dynamic_ncols=True)
        if tqdm is not None and ts
        else None
    )
    it = pbar if pbar is not None else ts
    for i, t in enumerate(it):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        for a in rows_by_t[t]:
            base_name = _make_patch_filename(a, video_name=video_name, species_name=species_name)
            out_path = out_dir / base_name
            if out_path.exists():
                if SKIP_EXISTING_PATCHES:
                    continue
                stem = out_path.stem
                suffix = out_path.suffix
                k = 1
                while True:
                    cand = out_dir / f"{stem}__dup{k}{suffix}"
                    if not cand.exists():
                        out_path = cand
                        break
                    k += 1

            patch = _crop_with_pad(frame_bgr, a.x, a.y, a.w, a.h)
            okw = cv2.imwrite(str(out_path), patch)
            if okw:
                saved += 1

        if pbar is not None:
            pbar.set_postfix(saved=int(saved), anns=int(len(rows_by_t.get(int(t), []))))
        elif (i + 1) % 25 == 0:
            print(f"[patches] processed {i+1}/{len(rows_by_t)} unique frames…")

    cap.release()
    if pbar is not None:
        pbar.close()
    return saved


def _ensure_train_root_scaffold(train_root: Path) -> Tuple[Path, Path]:
    integrated = train_root / INTEGRATED_PIPELINE_DATASETS_DIRNAME
    single = train_root / SINGLE_SPECIES_DATASETS_DIRNAME
    integrated.mkdir(parents=True, exist_ok=True)
    single.mkdir(parents=True, exist_ok=True)
    return integrated, single


def _ensure_time_dataset_scaffold(time_root: Path) -> None:
    (time_root / INITIAL_DATASET_DIRNAME / CLASS_FIREFLY).mkdir(parents=True, exist_ok=True)
    (time_root / INITIAL_DATASET_DIRNAME / CLASS_BACKGROUND).mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        (time_root / FINAL_DATASET_DIRNAME / split / CLASS_FIREFLY).mkdir(parents=True, exist_ok=True)
        (time_root / FINAL_DATASET_DIRNAME / split / CLASS_BACKGROUND).mkdir(parents=True, exist_ok=True)


def _version_num_from_name(name: str) -> Optional[int]:
    m = re.match(r"^v(?P<n>\d+)(?:_|$)", name)
    if not m:
        return None
    try:
        return int(m.group("n"))
    except Exception:
        return None


def _latest_version_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    best: Tuple[int, str, Path] | None = None
    for p in root.iterdir():
        if not p.is_dir():
            continue
        n = _version_num_from_name(p.name)
        if n is None:
            continue
        key = (n, p.name, p)
        if best is None or key[0] > best[0] or (key[0] == best[0] and key[1] > best[1]):
            best = key
    return best[2] if best else None


def _next_version_dir(root: Path) -> Tuple[Path, Optional[Path]]:
    prev = _latest_version_dir(root)
    prev_n = _version_num_from_name(prev.name) if prev else 0
    new_n = int(prev_n or 0) + 1
    new_dir = root / f"v{new_n}_{_today_tag()}"
    return new_dir, prev


def _copytree(src: Path, dst: Path, mode: str) -> None:
    if not src.is_dir():
        raise ValueError(f"copytree src is not a directory: {src}")
    if dst.exists():
        raise FileExistsError(dst)
    dst.mkdir(parents=True, exist_ok=False)

    mode = (mode or "copy").strip().lower()

    total_files = 0
    for _, _, files in os.walk(src):
        total_files += len(files)

    pbar = (
        tqdm(total=total_files, desc=f"[version] {src.name} -> {dst.name}", unit="file", dynamic_ncols=True)
        if tqdm is not None and total_files > 0
        else None
    )
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (dst / rel / d).mkdir(parents=True, exist_ok=True)
        for f in files:
            s = Path(root) / f
            t = dst / rel / f
            if mode == "hardlink":
                try:
                    os.link(s, t)
                except OSError as e:
                    if e.errno in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EEXIST}:
                        shutil.copy2(s, t)
                    else:
                        shutil.copy2(s, t)
            else:
                shutil.copy2(s, t)
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()


def _safe_rmtree(p: Path) -> None:
    if not p.exists():
        return
    if not p.is_dir():
        raise ValueError(f"Refusing to delete non-directory: {p}")
    shutil.rmtree(p)


def _split_dataset_dir(
    src_initial: Path,
    dst_final: Path,
    *,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    seed: int,
    copy_mode: str,
) -> Dict[str, Dict[str, int]]:
    if abs((train_pct + val_pct + test_pct) - 1.0) > 1e-6:
        raise ValueError("FINAL_TRAIN_PCT + FINAL_VAL_PCT + FINAL_TEST_PCT must sum to 1.0")

    _safe_rmtree(dst_final)
    dst_final.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(seed))

    metrics: Dict[str, Dict[str, int]] = {}
    for cls in (CLASS_FIREFLY, CLASS_BACKGROUND):
        src_cls = src_initial / cls
        files = sorted(list(_iter_image_files(src_cls)))
        rng.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_pct)
        n_val = int(n_total * val_pct)
        split_files = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        metrics[cls] = {"total": n_total}
        pbar = (
            tqdm(total=n_total, desc=f"[final split] {cls}", unit="img", dynamic_ncols=True)
            if tqdm is not None and n_total > 0
            else None
        )
        for split, flist in split_files.items():
            dst_dir = dst_final / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            metrics[cls][split] = len(flist)
            for src_path in flist:
                dst_path = dst_dir / src_path.name
                if copy_mode == "hardlink":
                    try:
                        os.link(src_path, dst_path)
                    except OSError:
                        shutil.copy2(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(split=str(split))
        if pbar is not None:
            pbar.close()
    return metrics


def _write_split_patch_locations_csv(
    firefly_dir: Path,
    out_csv: Path,
    *,
    include_species: bool,
) -> int:
    rows: List[Dict[str, object]] = []
    for img in sorted(_iter_image_files(firefly_dir)):
        meta = _parse_patch_filename(img)
        if not meta:
            continue
        x, y, w, h, t, video_name, species_name = meta
        row: Dict[str, object] = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "t": t,
            "video_name": video_name,
        }
        if include_species:
            row["species_name"] = species_name
        rows.append(row)

    fields = ["x", "y", "w", "h", "t", "video_name"]
    if include_species:
        fields.append("species_name")
    _write_csv_rows(out_csv, fields, rows)
    return len(rows)


def _collect_train_stats(train_root: Path) -> Dict[str, Dict[str, int]]:
    integrated_root = train_root / INTEGRATED_PIPELINE_DATASETS_DIRNAME
    single_root = train_root / SINGLE_SPECIES_DATASETS_DIRNAME

    stats: Dict[str, Dict[str, int]] = {
        "single_species_firefly": {},
        "single_species_background": {},
        "integrated_firefly": {},
        "integrated_background": {},
    }

    # single-species: count images in latest version initial/{firefly,background} per species
    if single_root.exists():
        for species_dir in sorted([p for p in single_root.iterdir() if p.is_dir()]):
            latest = _latest_version_dir(species_dir)
            if not latest:
                continue
            firefly_dir = latest / INITIAL_DATASET_DIRNAME / CLASS_FIREFLY
            background_dir = latest / INITIAL_DATASET_DIRNAME / CLASS_BACKGROUND
            stats["single_species_firefly"][species_dir.name] = _count_images(firefly_dir)
            stats["single_species_background"][species_dir.name] = _count_images(background_dir)

    # integrated: count images by parsing filenames in latest version initial/{firefly,background} across day+night
    latest_int = _latest_version_dir(integrated_root)
    if latest_int:
        counts_ff: Dict[str, int] = {}
        counts_bg: Dict[str, int] = {}
        for tod_dir in (DAY_DATASET_DIRNAME, NIGHT_DATASET_DIRNAME):
            for cls, counts in ((CLASS_FIREFLY, counts_ff), (CLASS_BACKGROUND, counts_bg)):
                cls_dir = latest_int / tod_dir / INITIAL_DATASET_DIRNAME / cls
                for img in _iter_image_files(cls_dir):
                    meta = _parse_patch_filename(img)
                    if not meta:
                        continue
                    _, _, _, _, _, _, sp = meta
                    counts[sp] = counts.get(sp, 0) + 1
        stats["integrated_firefly"] = dict(sorted(counts_ff.items(), key=lambda kv: (-kv[1], kv[0])))
        stats["integrated_background"] = dict(sorted(counts_bg.items(), key=lambda kv: (-kv[1], kv[0])))

    return stats


def _collect_validation_stats(val_root: Path) -> Dict[str, Dict[str, int]]:
    combined_root = val_root / VALIDATION_COMBINED_DIRNAME
    individual_root = val_root / VALIDATION_INDIVIDUAL_DIRNAME

    out: Dict[str, Dict[str, int]] = {"combined": {}, "individual": {}}

    latest_combined = _latest_version_dir(combined_root)
    if latest_combined:
        rows, _ = _read_csv_rows(latest_combined / VALIDATION_CSV_NAME)
        counts: Dict[str, int] = {}
        for r in rows:
            sp = r.get("species_name", "") or ""
            if not sp:
                continue
            counts[sp] = counts.get(sp, 0) + 1
        out["combined"] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    if individual_root.exists():
        for species_dir in sorted([p for p in individual_root.iterdir() if p.is_dir()]):
            latest = _latest_version_dir(species_dir)
            if not latest:
                continue
            rows, _ = _read_csv_rows(latest / VALIDATION_CSV_NAME)
            out["individual"][species_dir.name] = len(rows)

    return out


def _print_stats(title: str, stats: Dict[str, Dict[str, int]]) -> None:
    print(f"\n=== {title} ===")
    for section, d in stats.items():
        items = ", ".join([f"{k}={v}" for k, v in d.items()]) if d else "(empty)"
        print(f"{section}: {items}")

def main() -> None:
    csv_path = Path(ANNOTATIONS_CSV_PATH).expanduser()
    video_path = Path(VIDEO_PATH).expanduser()

    video_name, species_name, time_of_day, primary_label = _parse_batch_identity(csv_path)
    dataset_time_dirname = DAY_DATASET_DIRNAME if time_of_day == "day_time" else NIGHT_DATASET_DIRNAME

    csv_paths: Dict[str, Path] = {primary_label: csv_path}
    if AUTO_LOAD_SIBLING_CLASS_CSV:
        other = CLASS_BACKGROUND if primary_label == CLASS_FIREFLY else CLASS_FIREFLY
        cand = csv_path.with_name(f"{video_name}_{species_name}_{time_of_day}_{other}{csv_path.suffix}")
        if cand.exists():
            try:
                v2, s2, t2, l2 = _parse_batch_identity(cand)
            except Exception as e:
                print(f"[warn] Skipping sibling CSV {cand.name!r}: {e}")
            else:
                if (v2, s2, t2, l2) == (video_name, species_name, time_of_day, other):
                    csv_paths[other] = cand
                else:
                    print(f"[warn] Skipping sibling CSV {cand.name!r}: identity mismatch")

    integrated_root, single_root = _ensure_train_root_scaffold(TRAIN_DATASETS_ROOT)

    # ── stats before ──
    train_before = _collect_train_stats(TRAIN_DATASETS_ROOT)
    val_before = _collect_validation_stats(VALIDATION_DATASETS_ROOT)
    _print_stats("TRAIN stats (before)", train_before)
    _print_stats("VALIDATION stats (before)", val_before)

    # ── read + split incoming annotations (firefly/background) ──
    anns_by_label: Dict[str, List[Annotation]] = {}
    for lbl, p in csv_paths.items():
        anns = _read_annotator_csv(p)
        anns_by_label[lbl] = anns

    if not any(anns_by_label.values()):
        print(f"No valid rows found in: {', '.join(str(p) for p in csv_paths.values())}; nothing to do.")
        return

    train_anns_by_label: Dict[str, List[Annotation]] = {}
    final_validation_firefly: List[Annotation] = []

    for lbl, anns in anns_by_label.items():
        if lbl == CLASS_FIREFLY:
            tr, fv = _split_annotations(anns, TRAIN_FRACTION_OF_BATCH, TRAIN_VAL_SPLIT_SEED)
            train_anns_by_label[lbl] = tr
            final_validation_firefly = fv
        else:
            # Background batches are training-only (no pipeline final validation)
            train_anns_by_label[lbl] = list(anns)

    # Auto background generation (TRAIN only)
    if AUTO_GENERATE_BACKGROUND_PATCHES:
        firefly_train = train_anns_by_label.get(CLASS_FIREFLY, [])
        if firefly_train:
            existing_bg = train_anns_by_label.get(CLASS_BACKGROUND, [])
            target_bg = int(round(float(AUTO_BACKGROUND_TO_FIREFLY_RATIO) * float(len(firefly_train))))
            need_bg = max(0, int(target_bg) - int(len(existing_bg)))
            if need_bg > 0:
                print(
                    "[bg] Auto-generating background patches:",
                    f"train_firefly={len(firefly_train)} target_bg={target_bg} existing_bg={len(existing_bg)} need={need_bg}",
                )
                try:
                    gen = _auto_generate_background_train_annotations(
                        video_path=video_path,
                        firefly_anns_all=anns_by_label.get(CLASS_FIREFLY, []),
                        n_needed=need_bg,
                    )
                except Exception as e:
                    print(f"[bg] WARNING: background auto-generation failed: {e}")
                    gen = []
                if gen:
                    train_anns_by_label.setdefault(CLASS_BACKGROUND, []).extend(gen)
                    print(f"[bg] Generated background train rows: {len(gen)}")

    # ── export this run’s split CSVs + train patches ──
    run_tag = _today_tag() + "__" + datetime.now().strftime("%H%M%S")
    batch_tag = _safe_name(f"{video_name}_{species_name}_{time_of_day}")
    export_dir = BATCH_EXPORT_ROOT / f"{batch_tag}__{run_tag}"
    export_patches_root = export_dir / "train_patches"

    export_dir.mkdir(parents=True, exist_ok=True)

    if CLASS_FIREFLY in anns_by_label:
        _write_csv_rows(
            export_dir / "train_annotations_firefly.csv",
            ["x", "y", "w", "h", "t"],
            [{"x": a.x, "y": a.y, "w": a.w, "h": a.h, "t": a.t} for a in train_anns_by_label.get(CLASS_FIREFLY, [])],
        )
        _write_csv_rows(
            export_dir / "final_validation_annotations_firefly.csv",
            ["x", "y", "w", "h", "t"],
            [{"x": a.x, "y": a.y, "w": a.w, "h": a.h, "t": a.t} for a in final_validation_firefly],
        )

    # Write background TRAIN rows if present (including auto-generated backgrounds).
    if train_anns_by_label.get(CLASS_BACKGROUND, []):
        _write_csv_rows(
            export_dir / "train_annotations_background.csv",
            ["x", "y", "w", "h", "t"],
            [{"x": a.x, "y": a.y, "w": a.w, "h": a.h, "t": a.t} for a in train_anns_by_label.get(CLASS_BACKGROUND, [])],
        )

    n_in_firefly = len(anns_by_label.get(CLASS_FIREFLY, []))
    n_in_bg = len(anns_by_label.get(CLASS_BACKGROUND, []))
    print(
        f"\nIncoming rows: firefly={n_in_firefly}, background={n_in_bg} | "
        f"train_firefly={len(train_anns_by_label.get(CLASS_FIREFLY, []))}, "
        f"train_background={len(train_anns_by_label.get(CLASS_BACKGROUND, []))}, "
        f"final_validation_firefly={len(final_validation_firefly)}"
    )

    # Patch extraction
    saved_by_label: Dict[str, int] = {}
    for lbl, tr_anns in train_anns_by_label.items():
        if not tr_anns:
            continue
        out_dir = export_patches_root / lbl
        print(f"Extracting {len(tr_anns)} {lbl.upper()} TRAIN patches → {out_dir}")
        saved_by_label[lbl] = _extract_patches_from_video(
            video_path=video_path,
            anns=tr_anns,
            out_dir=out_dir,
            video_name=video_name,
            species_name=species_name,
        )
        print(f"Saved {saved_by_label[lbl]} {lbl} patches.")

    has_train_data = any(train_anns_by_label.get(lbl) for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND))

    # ────────────────────────────────────────────────────────────────────
    # TRAIN DATASETS (patches + patch_locations.csv + final split)
    # ────────────────────────────────────────────────────────────────────

    if has_train_data:
        # ---- single-species dataset update ----
        species_root = single_root / species_name
        species_root.mkdir(parents=True, exist_ok=True)

        target_species_ver = Path(SINGLE_SPECIES_TARGET_VERSION_DIR).expanduser() if SINGLE_SPECIES_TARGET_VERSION_DIR else None
        if target_species_ver is not None:
            try:
                target_species_ver.relative_to(species_root)
            except Exception as e:
                raise ValueError(
                    f"SINGLE_SPECIES_TARGET_VERSION_DIR must be under {species_root} (got {target_species_ver})"
                ) from e

            new_species_ver = target_species_ver
            prev_species_ver = _latest_version_dir(species_root)
            if not new_species_ver.exists():
                if prev_species_ver:
                    print(f"\n[single] Initializing {new_species_ver} (copying from {prev_species_ver.name})")
                    _copytree(prev_species_ver, new_species_ver, VERSION_COPY_MODE)
                else:
                    print(f"\n[single] Initializing {new_species_ver} (new species)")
                    new_species_ver.mkdir(parents=True, exist_ok=False)
            else:
                print(f"\n[single] Using existing target version dir: {new_species_ver}")
        else:
            new_species_ver, prev_species_ver = _next_version_dir(species_root)
            if prev_species_ver:
                print(f"\n[single] Creating {new_species_ver} (copying from {prev_species_ver.name})")
                _copytree(prev_species_ver, new_species_ver, VERSION_COPY_MODE)
            else:
                print(f"\n[single] Creating {new_species_ver} (new species)")
                new_species_ver.mkdir(parents=True, exist_ok=False)

        _ensure_time_dataset_scaffold(new_species_ver)

        # Add new patches into initial/{firefly,background}
        for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
            src_dir = export_patches_root / lbl
            if not src_dir.exists():
                continue
            dst_dir = new_species_ver / INITIAL_DATASET_DIRNAME / lbl
            dst_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            src_files = sorted(_iter_image_files(src_dir))
            pbar = (
                tqdm(src_files, desc=f"[single] copy {lbl}", unit="img", dynamic_ncols=True)
                if tqdm is not None and src_files
                else None
            )
            it = pbar if pbar is not None else src_files
            for p in it:
                out = dst_dir / p.name
                if out.exists() and SKIP_EXISTING_PATCHES:
                    continue
                if out.exists() and not SKIP_EXISTING_PATCHES:
                    stem = out.stem
                    suffix = out.suffix
                    k = 1
                    while (dst_dir / f"{stem}__dup{k}{suffix}").exists():
                        k += 1
                    out = dst_dir / f"{stem}__dup{k}{suffix}"
                shutil.copy2(p, out)
                copied += 1
                if pbar is not None:
                    pbar.set_postfix(copied=int(copied))
            if pbar is not None:
                pbar.close()
            print(f"[single] Added {copied} {lbl} patches → {dst_dir}")

        # Update patch_locations CSVs for this version (train rows only)
        for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
            tr_anns = train_anns_by_label.get(lbl, [])
            if not tr_anns:
                continue
            out_name = PATCH_LOCATIONS_CSV_NAME if lbl == CLASS_FIREFLY else PATCH_LOCATIONS_BACKGROUND_CSV_NAME
            out_csv = new_species_ver / out_name
            existing_rows, _ = _read_csv_rows(out_csv)
            new_rows = [
                {
                    "x": str(a.x),
                    "y": str(a.y),
                    "w": str(a.w),
                    "h": str(a.h),
                    "t": str(a.t),
                    "video_name": video_name,
                }
                for a in tr_anns
            ]
            merged = _merge_csv_rows(existing_rows, new_rows, key_fields=["x", "y", "w", "h", "t", "video_name"])
            _write_csv_rows(out_csv, ["x", "y", "w", "h", "t", "video_name"], merged)
            print(f"[single] {out_csv.name} rows: {len(merged)}")

        if SKIP_FINAL_SPLIT_REBUILD:
            print("[single] SKIP_FINAL_SPLIT_REBUILD=True; leaving final split unchanged for now.")
        else:
            # Rebuild final dataset split
            src_initial = new_species_ver / INITIAL_DATASET_DIRNAME
            dst_final = new_species_ver / FINAL_DATASET_DIRNAME
            metrics = _split_dataset_dir(
                src_initial,
                dst_final,
                train_pct=FINAL_TRAIN_PCT,
                val_pct=FINAL_VAL_PCT,
                test_pct=FINAL_TEST_PCT,
                seed=FINAL_SPLIT_SEED,
                copy_mode=VERSION_COPY_MODE,
            )
            print(f"[single] Final split metrics: {metrics}")

            # Write split CSVs (firefly + background)
            for split in SPLIT_NAMES:
                for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
                    cls_dir = dst_final / split / lbl
                    prefix = (
                        PATCH_LOCATIONS_SPLIT_PREFIX
                        if lbl == CLASS_FIREFLY
                        else PATCH_LOCATIONS_BACKGROUND_SPLIT_PREFIX
                    )
                    out_csv = new_species_ver / f"{prefix}{split}.csv"
                    nrows = _write_split_patch_locations_csv(cls_dir, out_csv, include_species=False)
                    print(f"[single] Wrote {out_csv.name} rows={nrows}")

        # ---- integrated dataset update ----
        target_int_ver = Path(INTEGRATED_TARGET_VERSION_DIR).expanduser() if INTEGRATED_TARGET_VERSION_DIR else None
        if target_int_ver is not None:
            try:
                target_int_ver.relative_to(integrated_root)
            except Exception as e:
                raise ValueError(
                    f"INTEGRATED_TARGET_VERSION_DIR must be under {integrated_root} (got {target_int_ver})"
                ) from e

            new_int_ver = target_int_ver
            prev_int_ver = _latest_version_dir(integrated_root)
            if not new_int_ver.exists():
                if prev_int_ver:
                    print(f"\n[integrated] Initializing {new_int_ver} (copying from {prev_int_ver.name})")
                    _copytree(prev_int_ver, new_int_ver, VERSION_COPY_MODE)
                else:
                    print(f"\n[integrated] Initializing {new_int_ver} (new integrated dataset)")
                    new_int_ver.mkdir(parents=True, exist_ok=False)
            else:
                print(f"\n[integrated] Using existing target version dir: {new_int_ver}")
        else:
            new_int_ver, prev_int_ver = _next_version_dir(integrated_root)
            if prev_int_ver:
                print(f"\n[integrated] Creating {new_int_ver} (copying from {prev_int_ver.name})")
                _copytree(prev_int_ver, new_int_ver, VERSION_COPY_MODE)
            else:
                print(f"\n[integrated] Creating {new_int_ver} (new integrated dataset)")
                new_int_ver.mkdir(parents=True, exist_ok=False)

        # Ensure both day/night scaffolds exist (even if only one is updated this run)
        day_root = new_int_ver / DAY_DATASET_DIRNAME
        night_root = new_int_ver / NIGHT_DATASET_DIRNAME
        _ensure_time_dataset_scaffold(day_root)
        _ensure_time_dataset_scaffold(night_root)

        target_time_root = new_int_ver / dataset_time_dirname
        # Add new patches into <time>/initial/{firefly,background}
        for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
            src_dir = export_patches_root / lbl
            if not src_dir.exists():
                continue
            dst_dir = target_time_root / INITIAL_DATASET_DIRNAME / lbl
            dst_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            src_files = sorted(_iter_image_files(src_dir))
            pbar = (
                tqdm(src_files, desc=f"[integrated] copy {lbl}", unit="img", dynamic_ncols=True)
                if tqdm is not None and src_files
                else None
            )
            it = pbar if pbar is not None else src_files
            for p in it:
                out = dst_dir / p.name
                if out.exists() and SKIP_EXISTING_PATCHES:
                    continue
                if out.exists() and not SKIP_EXISTING_PATCHES:
                    stem = out.stem
                    suffix = out.suffix
                    k = 1
                    while (dst_dir / f"{stem}__dup{k}{suffix}").exists():
                        k += 1
                    out = dst_dir / f"{stem}__dup{k}{suffix}"
                shutil.copy2(p, out)
                copied += 1
                if pbar is not None:
                    pbar.set_postfix(copied=int(copied))
            if pbar is not None:
                pbar.close()
            print(f"[integrated] Added {copied} {lbl} patches → {dst_dir}")

        # Update integrated patch_locations CSVs (at version root)
        for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
            tr_anns = train_anns_by_label.get(lbl, [])
            if not tr_anns:
                continue
            out_name = PATCH_LOCATIONS_CSV_NAME if lbl == CLASS_FIREFLY else PATCH_LOCATIONS_BACKGROUND_CSV_NAME
            out_csv = new_int_ver / out_name
            existing_rows, _ = _read_csv_rows(out_csv)
            new_rows = [
                {
                    "x": str(a.x),
                    "y": str(a.y),
                    "w": str(a.w),
                    "h": str(a.h),
                    "t": str(a.t),
                    "video_name": video_name,
                    "species_name": species_name,
                }
                for a in tr_anns
            ]
            merged = _merge_csv_rows(
                existing_rows, new_rows, key_fields=["x", "y", "w", "h", "t", "video_name", "species_name"]
            )
            _write_csv_rows(
                out_csv,
                ["x", "y", "w", "h", "t", "video_name", "species_name"],
                merged,
            )
            print(f"[integrated] {out_csv.name} rows: {len(merged)}")

        if SKIP_FINAL_SPLIT_REBUILD:
            print("[integrated] SKIP_FINAL_SPLIT_REBUILD=True; leaving final split unchanged for now.")
        else:
            # Rebuild final dataset split for the affected time-of-day dataset only
            src_initial = target_time_root / INITIAL_DATASET_DIRNAME
            dst_final = target_time_root / FINAL_DATASET_DIRNAME
            metrics = _split_dataset_dir(
                src_initial,
                dst_final,
                train_pct=FINAL_TRAIN_PCT,
                val_pct=FINAL_VAL_PCT,
                test_pct=FINAL_TEST_PCT,
                seed=FINAL_SPLIT_SEED,
                copy_mode=VERSION_COPY_MODE,
            )
            print(f"[integrated] ({dataset_time_dirname}) Final split metrics: {metrics}")

            for split in SPLIT_NAMES:
                for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
                    cls_dir = dst_final / split / lbl
                    prefix = (
                        PATCH_LOCATIONS_SPLIT_PREFIX
                        if lbl == CLASS_FIREFLY
                        else PATCH_LOCATIONS_BACKGROUND_SPLIT_PREFIX
                    )
                    out_csv = target_time_root / f"{prefix}{split}.csv"
                    nrows = _write_split_patch_locations_csv(cls_dir, out_csv, include_species=True)
                    print(f"[integrated] Wrote {out_csv} rows={nrows}")
    else:
        print("\nNo TRAIN rows; skipping patch extraction + TRAIN dataset updates.")

    # ────────────────────────────────────────────────────────────────────
    # FINAL VALIDATION DATASETS (CSV-only)
    # ────────────────────────────────────────────────────────────────────

    # Validation datasets (firefly only)
    if final_validation_firefly:
        combined_root = VALIDATION_DATASETS_ROOT / VALIDATION_COMBINED_DIRNAME
        individual_root = VALIDATION_DATASETS_ROOT / VALIDATION_INDIVIDUAL_DIRNAME / species_name
        combined_root.mkdir(parents=True, exist_ok=True)
        individual_root.mkdir(parents=True, exist_ok=True)

        target_comb_ver = (
            Path(VALIDATION_COMBINED_TARGET_VERSION_DIR).expanduser()
            if VALIDATION_COMBINED_TARGET_VERSION_DIR
            else None
        )
        if target_comb_ver is not None:
            try:
                target_comb_ver.relative_to(combined_root)
            except Exception as e:
                raise ValueError(
                    f"VALIDATION_COMBINED_TARGET_VERSION_DIR must be under {combined_root} (got {target_comb_ver})"
                ) from e
            new_comb_ver = target_comb_ver
            prev_comb_ver = _latest_version_dir(combined_root)
            if not new_comb_ver.exists():
                if prev_comb_ver:
                    _copytree(prev_comb_ver, new_comb_ver, VERSION_COPY_MODE)
                else:
                    new_comb_ver.mkdir(parents=True, exist_ok=False)
        else:
            # Combined validation
            new_comb_ver, prev_comb_ver = _next_version_dir(combined_root)
            if prev_comb_ver:
                _copytree(prev_comb_ver, new_comb_ver, VERSION_COPY_MODE)
            else:
                new_comb_ver.mkdir(parents=True, exist_ok=False)
        comb_csv = new_comb_ver / VALIDATION_CSV_NAME
        existing_rows, _ = _read_csv_rows(comb_csv)
        new_rows = [
            {
                "x": str(a.x),
                "y": str(a.y),
                "w": str(a.w),
                "h": str(a.h),
                "t": str(a.t),
                "video_name": video_name,
                "species_name": species_name,
            }
            for a in final_validation_firefly
        ]
        merged = _merge_csv_rows(
            existing_rows, new_rows, key_fields=["x", "y", "w", "h", "t", "video_name", "species_name"]
        )
        _write_csv_rows(comb_csv, ["x", "y", "w", "h", "t", "video_name", "species_name"], merged)
        print(f"\n[validation-combined] Wrote {comb_csv} rows={len(merged)}")

        target_ind_ver = (
            Path(VALIDATION_INDIVIDUAL_TARGET_VERSION_DIR).expanduser()
            if VALIDATION_INDIVIDUAL_TARGET_VERSION_DIR
            else None
        )
        if target_ind_ver is not None:
            try:
                target_ind_ver.relative_to(individual_root)
            except Exception as e:
                raise ValueError(
                    f"VALIDATION_INDIVIDUAL_TARGET_VERSION_DIR must be under {individual_root} (got {target_ind_ver})"
                ) from e
            new_ind_ver = target_ind_ver
            prev_ind_ver = _latest_version_dir(individual_root)
            if not new_ind_ver.exists():
                if prev_ind_ver:
                    _copytree(prev_ind_ver, new_ind_ver, VERSION_COPY_MODE)
                else:
                    new_ind_ver.mkdir(parents=True, exist_ok=False)
        else:
            # Individual species validation
            new_ind_ver, prev_ind_ver = _next_version_dir(individual_root)
            if prev_ind_ver:
                _copytree(prev_ind_ver, new_ind_ver, VERSION_COPY_MODE)
            else:
                new_ind_ver.mkdir(parents=True, exist_ok=False)
        ind_csv = new_ind_ver / VALIDATION_CSV_NAME
        existing_rows, _ = _read_csv_rows(ind_csv)
        merged = _merge_csv_rows(
            existing_rows, new_rows, key_fields=["x", "y", "w", "h", "t", "video_name", "species_name"]
        )
        _write_csv_rows(ind_csv, ["x", "y", "w", "h", "t", "video_name", "species_name"], merged)
        print(f"[validation-individual] Wrote {ind_csv} rows={len(merged)}")
    else:
        print("\nNo FINAL-VALIDATION rows after split; skipping validation dataset updates.")

    # ── stats after ──
    train_after = _collect_train_stats(TRAIN_DATASETS_ROOT)
    val_after = _collect_validation_stats(VALIDATION_DATASETS_ROOT)
    _print_stats("TRAIN stats (after)", train_after)
    _print_stats("VALIDATION stats (after)", val_after)

    print(f"\n✅ Done. Batch export: {export_dir}")


if __name__ == "__main__":
    main()
