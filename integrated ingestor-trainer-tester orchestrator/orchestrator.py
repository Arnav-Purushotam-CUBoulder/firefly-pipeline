#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import stage1_ingestor
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# User-configurable globals (defaults; CLI args can override)
# -----------------------------------------------------------------------------

# Optional single root folder. If set (or passed via --root), the orchestrator will
# create/use:
#   <ROOT>/patch training datasets and pipeline validation data
#   <ROOT>/model zoo
#   <ROOT>/inference outputs
ROOT_PATH: str | Path = "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data" 



# Folder containing *many* observed videos + their annotator CSVs (and potentially other files).
# The orchestrator will:
# - match annotation CSVs to .mp4 videos by filename
# - split matched video/csv pairs: TRAIN fraction → training ingestion, remainder → validation/testing
OBSERVED_DATA_DIR: str | Path = "/mnt/Samsung_SSD_2TB/integrated prototype raw videos/Photinus Knulli"

# Species name for this run. This is the source of truth; the orchestrator does not
# infer species from CSV filenames.
SPECIES_NAME: str = "photinus-knulli"

# Global video type for this run (used for dataset routing + model-zoo selection).
# Allowed values: "day" | "night"
TYPE_OF_VIDEO: str = "night"

# Train/validation split at the *video* level when ingesting an observed folder.
# Example: 0.8 -> 80% of videos for training ingestion, 20% held out for validation/testing.
TRAIN_PAIR_FRACTION: float = 0.8

# Ingestion versioning:
# If True, ingest all TRAIN videos into a single new dataset version and all held-out
# VALIDATION videos into a single new validation-dataset version (instead of one
# version per video).
ONE_DATASET_VERSION_PER_BATCH: bool = True

# Search dirs for locating existing validation videos by stem (used when no new held-out validation rows).
VALIDATION_VIDEO_SEARCH_DIRS: List[str | Path] = []

# Stage 1 ingestor-core config overrides (passed into stage1_ingestor_core)
AUTO_LOAD_SIBLING_CLASS_CSV: bool = True

# Auto background patch generation (passed into stage1_ingestor_core)
# ------------------------------------------------------------
# Enables training a 2-class patch classifier even when the input CSVs only contain
# FIREFFLY annotations. Background patches are synthesized by sampling random
# frames that do NOT contain any annotated firefly frames, then extracting 10x10
# "hard negative" blob-centered crops (with optional fallback to random centers).
AUTO_GENERATE_BACKGROUND_PATCHES: bool = True
AUTO_BACKGROUND_TO_FIREFLY_RATIO: float = 10.0
AUTO_BACKGROUND_PATCH_SIZE_PX: int = 10
AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME: int = 10
AUTO_BACKGROUND_MAX_FRAME_SAMPLES: int = 5000
AUTO_BACKGROUND_SEED: int = 1337
AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS: bool = True

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

# Dataset versioning copy mode (Stage 1)
# - "hardlink": fast + space efficient (recommended when working on a single filesystem)
# - "copy": duplicates files per version (slow + uses more storage)
DATASET_VERSION_COPY_MODE: str = "hardlink"

# Training config (ResNet patch/CNN classifier)
TRAIN_EPOCHS: int = 50
TRAIN_BATCH_SIZE: int = 128
TRAIN_LR: float = 3e-4
TRAIN_NUM_WORKERS: int = 2
TRAIN_RESNET: str = "resnet18"
TRAIN_SEED: int = 1337

# Dataset sanitization (Stage 2 preflight)
# ---------------------------------------
# Prevent crashes due to corrupt image files in train/val/test (e.g. 0-byte PNGs).
SANITIZE_DATASET_IMAGES: bool = True
SANITIZE_DATASET_MODE: str = "quarantine"  # "quarantine" | "delete"
SANITIZE_DATASET_VERIFY_WITH_PIL: bool = True
SANITIZE_DATASET_REPORT_MAX: int = 20

# Evaluation config
GATEWAY_BRIGHTNESS_THRESHOLD: float = 60.0
GATEWAY_BRIGHTNESS_NUM_FRAMES: int = 5
GATEWAY_MAX_CONCURRENT: int = 1  # keep 1 for GT-backed validation runs
FORCE_GATEWAY_TESTS: bool = True

# If True and there are no held-out rows in this batch, evaluate on *all* existing validation videos.
EVAL_EXISTING_VALIDATION_WHEN_NO_NEW_HELDOUT: bool = True

# If True, also evaluate the global model (in addition to the single-species model).
EVAL_GLOBAL_MODEL: bool = True

# If True, also evaluate the single-species model.
EVAL_SINGLE_SPECIES_MODEL: bool = True

# If True, trains new models when this batch contains TRAIN rows; otherwise reuses latest models.
TRAIN_MODELS_IF_TRAIN_ROWS_PRESENT: bool = True

# Set True to do everything except actually call stage1 ingest / training / gateway.
DRY_RUN: bool = False

# -----------------------------------------------------------------------------
# Legacy baseline evaluation (optional)
# -----------------------------------------------------------------------------

# If True, also run legacy/lab baseline detectors on the same eval videos and score
# them using the same point-distance validation logic (Stage5 validator).
RUN_BASELINE_EVAL: bool = True

# If True, only run baselines on videos routed to the night pipeline.
BASELINES_ONLY_FOR_NIGHT: bool = True

# Baseline validator settings.
# Use a wider sweep so baseline reports include looser match thresholds too.
BASELINE_DIST_THRESHOLDS_PX: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
BASELINE_VALIDATE_CROP_W: int = 10
BASELINE_VALIDATE_CROP_H: int = 10
BASELINE_MAX_FRAMES: int | None = None

# If True, also render a debug overlay video for baselines (GT=GREEN, preds=RED).
# This is separate from the pipelines' own overlay stages.
BASELINE_RENDER_OVERLAY_VIDEO: bool = True
BASELINE_OVERLAY_RENDER_THRESH_VIDEOS: bool = False  # TP/FP/FN per-threshold videos (very expensive)

# Baseline 1: Nolan-style rolling-mean background detector
RUN_BASELINE_LAB_METHOD: bool = True
LAB_BASELINE_THRESHOLD: float = 0.12
LAB_BASELINE_BLUR_SIGMA: float = 1.0
LAB_BASELINE_BKGR_WINDOW_SEC: float = 2.0

# Baseline 2: Raphael OOrb tracking detector (ffnet) + gaussian centroids
RUN_BASELINE_RAPHAEL_METHOD: bool = True
RAPHAEL_MODEL_SOURCE_PATH: str | Path = "/Users/arnavps/Desktop/RA info/New Deep Learning project/firefl-eye-net/ffnet_best.pth"
RAPHAEL_BW_THR: float = 0.2
RAPHAEL_CLASSIFY_THR: float = 0.98
RAPHAEL_BKGR_WINDOW_SEC: float = 2.0
RAPHAEL_BLUR_SIGMA: float = 0.0
RAPHAEL_PATCH_SIZE_PX: int = 33
RAPHAEL_BATCH_SIZE: int = 1000
RAPHAEL_GAUSS_CROP_SIZE: int = 10
RAPHAEL_DEVICE: str = "auto"


# -----------------------------------------------------------------------------
# Internal constants
# -----------------------------------------------------------------------------

DEFAULT_DATA_SUBDIR = "patch training datasets and pipeline validation data"
DEFAULT_MODEL_ZOO_SUBDIR = "model zoo"
DEFAULT_INFERENCE_OUTPUT_SUBDIR = "inference outputs"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}


def _count_images_in_dir(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def _infer_repo_layout(this_file: Path) -> Tuple[Path, Path]:
    """
    This repo is sometimes vendored under a `test1/` folder.

    Supported layouts:
    - <repo_root>/integrated pipeline/gateway.py
    - <repo_root>/test1/integrated pipeline/gateway.py
    """
    this_file = Path(this_file).resolve()
    for root in [this_file.parents[1], this_file.parents[2], this_file.parents[3]]:
        if (root / "integrated pipeline" / "gateway.py").is_file():
            return root, Path(".")
        if (root / "test1" / "integrated pipeline" / "gateway.py").is_file():
            return root, Path("test1")
    # Fall back to the historical assumption (repo root is two levels up, with a test1/ prefix).
    return this_file.parents[2], Path("test1")


_REPO_ROOT, _REPO_SUBDIR_PREFIX = _infer_repo_layout(Path(__file__))

_THIS_DIR = Path(__file__).resolve().parent
LOCAL_TRAINING_SCRIPT_PY = (_THIS_DIR / "stage2_trainer.py").resolve()

GATEWAY_REL_PATH = _REPO_SUBDIR_PREFIX / "integrated pipeline/gateway.py"

DAY_OUTPUT_SUBDIR = "day_pipeline_v3"
NIGHT_OUTPUT_SUBDIR = "night_time_pipeline"

DAY_PIPELINE_DIR_REL_PATH = _REPO_SUBDIR_PREFIX / "day time pipeline v3 (yolo + patch classifier ensemble)"
LAB_BASELINE_SCRIPT_REL_PATH = _REPO_SUBDIR_PREFIX / "tools/legacy_baselines/nolan_mp4_to_predcsv.py"
RAPHAEL_BASELINE_SCRIPT_REL_PATH = _REPO_SUBDIR_PREFIX / "tools/legacy_baselines/raphael_oorb_detect_and_gauss.py"
BASELINE_OUTPUT_SUBDIR = "baselines"
BASELINE_MODELS_SUBDIR = "baseline models"


@dataclass(frozen=True)
class RunIdentity:
    dataset_name: str
    species_name: str
    time_of_day: str  # "day_time" | "night_time"


def _today_tag() -> str:
    return datetime.now().strftime("%Y%m%d")


def _run_tag() -> str:
    return datetime.now().strftime("%Y%m%d__%H%M%S")


def _run_dir_name(
    ident: RunIdentity,
    *,
    n_pairs_total: int,
    n_pairs_train: int,
    n_pairs_val: int,
    train_rows_firefly: int,
    train_rows_background: int,
    val_rows_firefly: int,
    did_train: bool,
) -> str:
    mode = "trained" if did_train else "reused"
    parts = [
        _run_tag(),
        ident.dataset_name,
        ident.species_name,
        ident.time_of_day,
        f"pairs{int(n_pairs_total)}",
        f"trP{int(n_pairs_train)}",
        f"vaP{int(n_pairs_val)}",
        f"trF{int(train_rows_firefly)}",
        f"trB{int(train_rows_background)}",
        f"vaF{int(val_rows_firefly)}",
        mode,
    ]
    safe_parts = [_safe_name(str(p)) for p in parts if str(p).strip()]
    return "__".join(safe_parts)


def _safe_name(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


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


def _next_version_dir(root: Path) -> Path:
    prev = _latest_version_dir(root)
    prev_n = _version_num_from_name(prev.name) if prev else 0
    new_n = int(prev_n or 0) + 1
    return root / f"v{new_n}_{_today_tag()}"


def _normalize_video_type(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"day", "daytime", "day_time"}:
        return "day"
    if s in {"night", "nighttime", "night_time"}:
        return "night"
    raise ValueError(f"type_of_video must be 'day' or 'night' (got {s!r})")


def _time_of_day_from_video_type(video_type: str) -> str:
    v = _normalize_video_type(video_type)
    return f"{v}_time"

# Ingestion mechanics (pair discovery, splitting, CSV staging, stage1_ingestor_core invocation)
# live in: integrated ingestor-trainer-tester orchestrator/stage1_ingestor.py


def _read_annotator_csv(csv_path: Path) -> List[Dict[str, int]]:
    with csv_path.open(newline="") as fh:
        r = csv.DictReader(fh)
        if not r.fieldnames:
            return []
        cols = {c.strip(): c for c in r.fieldnames if c}
        t_col = cols.get("t") or cols.get("frame") or cols.get("time")
        if not t_col or not {"x", "y", "w", "h"}.issubset(set(cols.keys())):
            raise ValueError(
                f"{csv_path} missing required columns. Need x,y,w,h and t (or frame). Got: {r.fieldnames}"
            )

        out: List[Dict[str, int]] = []
        seen: set[Tuple[int, int, int, int, int]] = set()
        for row in r:
            try:
                x = int(round(float(row[cols["x"]])))
                y = int(round(float(row[cols["y"]])))
                w = int(round(float(row[cols["w"]])))
                h = int(round(float(row[cols["h"]])))
                t = int(round(float(row[t_col])))
            except Exception:
                continue
            if w <= 0 or h <= 0 or t < 0:
                continue
            key = (x, y, w, h, t)
            if key in seen:
                continue
            seen.add(key)
            out.append({"x": x, "y": y, "w": w, "h": h, "t": t})
        return out


def _ensure_model_zoo_scaffold(model_root: Path) -> Dict[str, Path]:
    global_root = model_root / "global models"
    day_global = global_root / "daytime global"
    night_global = global_root / "night time global"
    single_root = model_root / "single species models"

    for d in (day_global, night_global, single_root):
        d.mkdir(parents=True, exist_ok=True)

    history = model_root / "results_history.jsonl"
    if not history.exists():
        history.parent.mkdir(parents=True, exist_ok=True)
        history.write_text("")

    registry = model_root / "video_registry.json"
    if not registry.exists():
        registry.write_text(json.dumps({}, indent=2))

    return {
        "day_global": day_global,
        "night_global": night_global,
        "single_root": single_root,
        "history": history,
        "registry": registry,
    }


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_registry(path: Path) -> Dict[str, str]:
    try:
        data = json.loads(path.read_text() or "{}")
        if isinstance(data, dict):
            out: Dict[str, str] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str):
                    out[k] = v
            return out
    except Exception:
        pass
    return {}


def _save_registry(path: Path, reg: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(reg.items())), indent=2))


def _find_video_by_stem(stem: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in VIDEO_EXTS:
            cand = d / f"{stem}{ext}"
            if cand.exists():
                return cand
    # fallback: recursive search (first match)
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stem == stem:
                return p
    return None


def _write_gt_csv(gt_path: Path, rows: Sequence[Dict[str, int]]) -> None:
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with gt_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "t"])
        w.writeheader()
        for r in rows:
            w.writerow({"x": int(r["x"]), "y": int(r["y"]), "t": int(r["t"])})


def _read_validation_combined_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        if not fieldnames:
            return []
        return [{k: (row.get(k) or "") for k in fieldnames} for row in r]


def _group_validation_rows_by_video(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, int]]]:
    out: Dict[str, List[Dict[str, int]]] = {}
    for r in rows:
        video_name = _safe_name(r.get("video_name", "") or "")
        if not video_name:
            continue
        try:
            x = int(round(float(r.get("x", "") or 0)))
            y = int(round(float(r.get("y", "") or 0)))
            t = int(round(float(r.get("t", "") or 0)))
        except Exception:
            continue
        out.setdefault(video_name, []).append({"x": x, "y": y, "t": t, "w": 0, "h": 0})
    return out


def _species_summary_from_patch_locations(csv_path: Path) -> Dict[str, Any]:
    """
    Extract unique species list (and optional counts) from a patch_locations*.csv.
    Expected column: species_name
    """
    if not csv_path.exists():
        return {"species": [], "counts": {}, "source_csv": str(csv_path), "error": "not_found"}

    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "species_name" not in (r.fieldnames or []):
            return {
                "species": [],
                "counts": {},
                "source_csv": str(csv_path),
                "error": f"missing_species_name_column (fields={r.fieldnames})",
            }

        counts: Counter[str] = Counter()
        for row in r:
            sp = _safe_name((row.get("species_name") or "").strip())
            if sp:
                counts[sp] += 1

    ordered_counts = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {"species": list(ordered_counts.keys()), "counts": ordered_counts, "source_csv": str(csv_path)}


def _infer_species_from_patch_filename(filename: str) -> str:
    """
    Heuristic parser for patch filenames.

    We have multiple legacy naming conventions:
      - photinus-knulli patches: "..._photinus-knulli.png"
      - legacy single-species patches: "forresti__img_0123.png", "frontalis__f00123_b003.png",
        "tremulans__frame_000900_component_535_1.png", etc.

    Returns a safe species_name or "" if unknown.
    """
    stem = Path(filename).stem
    stem = stem.strip()
    if not stem:
        return ""

    # Legacy: "<species>__...".
    if "__" in stem:
        prefix = stem.split("__", 1)[0]
        prefix = _safe_name(prefix)
        # Guard against numeric-leading stems (e.g., "100_200_...").
        if prefix and not prefix[0].isdigit():
            return prefix

    # Modern: "..._<species-with-hyphens>" at the end (e.g., photinus-knulli).
    m = re.search(r"(?:^|_)([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)$", stem)
    if m:
        return _safe_name(m.group(1))

    return ""


def _species_summary_from_dataset_dir(dataset_dir: Path) -> Dict[str, Any]:
    """
    Summarize which species are present in a dataset folder by scanning *filenames* in:
      dataset_dir/{train,val,test}/firefly

    This is the source of truth for training (ImageFolder trains on the folder contents),
    and catches legacy imported patches that may be missing from patch_locations*.csv.
    """
    splits = ("train", "val", "test")
    counts_by_split: Dict[str, Dict[str, int]] = {}
    unknown_by_split: Dict[str, int] = {}
    total: Counter[str] = Counter()

    for split in splits:
        firefly_dir = dataset_dir / split / "firefly"
        if not firefly_dir.exists():
            counts_by_split[split] = {}
            unknown_by_split[split] = 0
            continue

        counts: Counter[str] = Counter()
        unknown = 0
        for p in firefly_dir.iterdir():
            if not p.is_file():
                continue
            sp = _infer_species_from_patch_filename(p.name)
            if sp:
                counts[sp] += 1
            else:
                unknown += 1

        ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        counts_by_split[split] = ordered
        unknown_by_split[split] = unknown
        total.update(counts)

    total_ordered = dict(sorted(total.items(), key=lambda kv: (-kv[1], kv[0])))
    train_counts = counts_by_split.get("train", {}) or {}
    return {
        "species": list(train_counts.keys()),
        "counts": train_counts,
        "counts_by_split": counts_by_split,
        "unknown_counts_by_split": unknown_by_split,
        "counts_total": total_ordered,
        "source_dir": str(dataset_dir),
    }


def _run_training(
    *,
    repo_root: Path,
    data_dir: Path,
    out_model_path: Path,
    metrics_out_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    resnet: str,
    seed: int,
    dry_run: bool,
) -> Dict[str, Any]:
    if dry_run:
        print(
            "[dry-run] Would run training:",
            f"data_dir={data_dir} out_model_path={out_model_path} epochs={epochs} batch_size={batch_size} lr={lr} "
            f"num_workers={num_workers} resnet={resnet} seed={seed} metrics_out={metrics_out_path}",
        )
        return {}

    if not LOCAL_TRAINING_SCRIPT_PY.exists():
        raise FileNotFoundError(LOCAL_TRAINING_SCRIPT_PY)

    import stage2_trainer as tr  # type: ignore

    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = tr.train_resnet_classifier(
        data_dir=Path(data_dir).expanduser().resolve(),
        best_model_path=Path(out_model_path).expanduser().resolve(),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        num_workers=int(num_workers),
        resnet_model=str(resnet),
        seed=int(seed),
        no_gui=True,
        sanitize_dataset=bool(SANITIZE_DATASET_IMAGES),
        sanitize_mode=str(SANITIZE_DATASET_MODE),
        sanitize_verify_with_pil=bool(SANITIZE_DATASET_VERIFY_WITH_PIL),
        sanitize_report_max=int(SANITIZE_DATASET_REPORT_MAX),
    )
    metrics_out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def _avg_brightness_first_frames(video_path: Path, *, num_frames: int) -> float:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    vals: List[float] = []
    try:
        for _ in range(max(1, int(num_frames))):
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vals.append(float(gray.mean()))
    finally:
        cap.release()

    if not vals:
        raise RuntimeError(f"Could not read any frames from: {video_path}")
    return sum(vals) / len(vals)


def _route_for_video(video_path: Path, *, thr: float, frames: int) -> str:
    b = _avg_brightness_first_frames(video_path, num_frames=frames)
    return "night" if b < float(thr) else "day"


def _parse_validation_metrics(stage_dir: Path) -> Dict[str, Any]:
    """
    Parse Stage5/Stage9 validation outputs written as:
      stage_dir/thr_*.px/{fps.csv,tps.csv,fns.csv}
    Returns best-F1 threshold summary + per-threshold stats.
    """
    if not stage_dir.exists():
        return {"error": f"validation dir not found: {stage_dir}"}

    thr_dirs = sorted([p for p in stage_dir.iterdir() if p.is_dir() and p.name.startswith("thr_")])
    per: List[Dict[str, Any]] = []
    for td in thr_dirs:
        csv_fp = td / "fps.csv"
        csv_tp = td / "tps.csv"
        csv_fn = td / "fns.csv"
        if not (csv_fp.exists() and csv_tp.exists() and csv_fn.exists()):
            continue

        def _count_rows(p: Path) -> int:
            with p.open(newline="") as f:
                r = csv.reader(f)
                next(r, None)  # header
                return sum(1 for _ in r)

        fp = _count_rows(csv_fp)
        tp = _count_rows(csv_tp)
        fn = _count_rows(csv_fn)
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per.append({"thr": td.name, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1})

    if not per:
        return {"error": f"no thr_* folders with fp/tp/fn csvs found under: {stage_dir}"}

    best = max(per, key=lambda d: (float(d.get("f1", 0.0)), -float(d.get("fp", 0.0))))
    return {"best": best, "per_threshold": per}


def _thr_name_to_float(thr_name: str) -> float | None:
    """
    Convert 'thr_5.0px' -> 5.0 (or None if it doesn't match).
    """
    try:
        s = str(thr_name).strip()
        if not s.startswith("thr_"):
            return None
        s = s.replace("thr_", "", 1)
        if s.endswith("px"):
            s = s[: -len("px")]
        return float(s)
    except Exception:
        return None


def _combine_metrics_across_videos(
    rows: Sequence[Dict[str, Any]],
    *,
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    """
    Combine TP/FP/FN across videos (per threshold).

    Input `rows` is the orchestrator's per-video result rows for a single method_key:
      {model_key, video_name, validation_metrics={per_threshold:[{thr,tp,fp,fn,...}]}}
    """
    thr_list = [float(t) for t in thresholds_px]
    totals_by_thr: Dict[float, Dict[str, Any]] = {
        t: {"tp": 0, "fp": 0, "fn": 0, "n_videos_used": 0} for t in thr_list
    }

    n_videos_total = 0
    n_videos_with_metrics = 0
    failures: List[Dict[str, Any]] = []

    for r in rows:
        n_videos_total += 1
        video_name = str(r.get("video_name") or "")
        vm = r.get("validation_metrics")
        if not isinstance(vm, dict) or vm.get("error"):
            failures.append({"video_name": video_name, "error": vm.get("error") if isinstance(vm, dict) else "no_metrics"})
            continue
        per = vm.get("per_threshold")
        if not isinstance(per, list) or not per:
            failures.append({"video_name": video_name, "error": "no_per_threshold"})
            continue

        # Map thr->counts for this video.
        by_thr: Dict[float, Dict[str, int]] = {}
        for it in per:
            if not isinstance(it, dict):
                continue
            thr = _thr_name_to_float(str(it.get("thr") or ""))
            if thr is None:
                continue
            try:
                by_thr[float(thr)] = {
                    "tp": int(it.get("tp") or 0),
                    "fp": int(it.get("fp") or 0),
                    "fn": int(it.get("fn") or 0),
                }
            except Exception:
                continue

        if not by_thr:
            failures.append({"video_name": video_name, "error": "no_thr_parsed"})
            continue

        n_videos_with_metrics += 1
        for thr in thr_list:
            c = by_thr.get(thr)
            if c is None:
                continue
            totals_by_thr[thr]["tp"] += c["tp"]
            totals_by_thr[thr]["fp"] += c["fp"]
            totals_by_thr[thr]["fn"] += c["fn"]
            totals_by_thr[thr]["n_videos_used"] += 1

    per_threshold_combined: List[Dict[str, Any]] = []
    for thr in thr_list:
        tp = int(totals_by_thr[thr]["tp"])
        fp = int(totals_by_thr[thr]["fp"])
        fn = int(totals_by_thr[thr]["fn"])
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_threshold_combined.append(
            {
                "thr_px": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "n_videos_used": int(totals_by_thr[thr]["n_videos_used"]),
            }
        )

    return {
        "n_videos_total": int(n_videos_total),
        "n_videos_with_metrics": int(n_videos_with_metrics),
        "failures": failures,
        "per_threshold": per_threshold_combined,
    }


def _format_combined_metrics_txt(
    combined_by_method: Dict[str, Dict[str, Any]],
    *,
    thresholds_px: Sequence[float],
) -> str:
    thr_list = [float(t) for t in thresholds_px]
    lines: List[str] = []
    lines.append("COMBINED RESULTS (SUM TP/FP/FN ACROSS VIDEOS)")
    lines.append(f"Thresholds (px): {', '.join(f'{t:.1f}' for t in thr_list)}")
    lines.append("")
    for method_key, payload in combined_by_method.items():
        lines.append(f"== {method_key} ==")
        lines.append(
            f"videos_total={payload.get('n_videos_total', 0)}  videos_with_metrics={payload.get('n_videos_with_metrics', 0)}"
        )
        lines.append("thr_px  tp      fp      fn      precision  recall     f1        videos_used")
        lines.append("-----  ------  ------  ------  ---------  --------  --------  ----------")
        per = payload.get("per_threshold")
        per_map: Dict[float, Dict[str, Any]] = {}
        if isinstance(per, list):
            for it in per:
                try:
                    per_map[float(it.get("thr_px"))] = it
                except Exception:
                    continue
        for thr in thr_list:
            it = per_map.get(thr, {})
            tp = int(it.get("tp") or 0)
            fp = int(it.get("fp") or 0)
            fn = int(it.get("fn") or 0)
            prec = float(it.get("precision") or 0.0)
            rec = float(it.get("recall") or 0.0)
            f1 = float(it.get("f1") or 0.0)
            used = int(it.get("n_videos_used") or 0)
            lines.append(f"{thr:5.1f}  {tp:6d}  {fp:6d}  {fn:6d}  {prec:9.4f}  {rec:8.4f}  {f1:8.4f}  {used:10d}")
        failures = payload.get("failures")
        if isinstance(failures, list) and failures:
            n_fail = len(failures)
            lines.append(f"failures={n_fail} (see run_record.json for per-video errors)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _run_gateway(
    *,
    repo_root: Path,
    gateway_path: Path,
    video_path: Path,
    output_root: Path,
    day_patch_model: Path | None,
    night_cnn_model: Path | None,
    thr: float,
    frames: int,
    max_concurrent: int,
    force_tests: bool,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        str(gateway_path),
        "--input",
        str(video_path),
        "--output-root",
        str(output_root),
        "--threshold",
        str(float(thr)),
        "--frames",
        str(int(frames)),
        "--max-concurrent",
        str(int(max_concurrent)),
    ]
    if day_patch_model is not None:
        cmd += ["--day-patch-model", str(day_patch_model)]
    if night_cnn_model is not None:
        cmd += ["--night-cnn-model", str(night_cnn_model)]
    if force_tests:
        cmd.append("--force-tests")

    if dry_run:
        print("[dry-run] Would run gateway:", " ".join(cmd))
        return

    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _copy_file_if_needed(src: Path, dst: Path, *, dry_run: bool) -> None:
    src = Path(src)
    dst = Path(dst)
    try:
        if src.resolve() == dst.resolve():
            return
    except Exception:
        pass

    if dry_run:
        print(f"[dry-run] Would copy baseline asset: {src} -> {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if dst.stat().st_size == src.stat().st_size:
                return
        except Exception:
            pass
    shutil.copy2(src, dst)


def _ensure_raphael_model_copy(*, model_root: Path, dry_run: bool) -> Dict[str, Any]:
    src_str = str(RAPHAEL_MODEL_SOURCE_PATH).strip() if RAPHAEL_MODEL_SOURCE_PATH else ""
    if not src_str:
        return {"source": None, "copied": None, "error": "no_source_path"}

    src = Path(src_str).expanduser().resolve()
    if not src.exists():
        return {"source": str(src), "copied": None, "error": "source_not_found"}

    dst_dir = model_root / BASELINE_MODELS_SUBDIR / "raphael_ffnet"
    dst = dst_dir / src.name
    try:
        _copy_file_if_needed(src, dst, dry_run=dry_run)
        return {"source": str(src), "copied": str(dst), "error": None}
    except Exception as e:
        return {"source": str(src), "copied": None, "error": f"copy_failed: {e}"}


def _run_baseline_lab_detector(
    *,
    repo_root: Path,
    video_path: Path,
    out_pred_csv: Path,
    threshold: float,
    blur_sigma: float,
    bkgr_window_sec: float,
    max_frames: int | None,
    dry_run: bool,
) -> None:
    script = (repo_root / LAB_BASELINE_SCRIPT_REL_PATH).resolve()
    if not script.exists():
        raise FileNotFoundError(script)

    cmd = [
        sys.executable,
        str(script),
        "--video",
        str(video_path),
        "--out-csv",
        str(out_pred_csv),
        "--threshold",
        str(float(threshold)),
        "--blur-sigma",
        str(float(blur_sigma)),
        "--bkgr-window-sec",
        str(float(bkgr_window_sec)),
    ]
    if max_frames is not None:
        cmd += ["--max-frames", str(int(max_frames))]

    if dry_run:
        print("[dry-run] Would run lab baseline detector:", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _run_baseline_raphael_detector(
    *,
    repo_root: Path,
    video_path: Path,
    model_path: Path,
    out_pred_csv: Path,
    out_raw_csv: Path,
    out_gauss_csv: Path,
    bw_thr: float,
    classify_thr: float,
    bkgr_window_sec: float,
    blur_sigma: float,
    patch_size_px: int,
    batch_size: int,
    gauss_crop_size: int,
    max_frames: int | None,
    device: str,
    dry_run: bool,
) -> None:
    script = (repo_root / RAPHAEL_BASELINE_SCRIPT_REL_PATH).resolve()
    if not script.exists():
        raise FileNotFoundError(script)

    cmd = [
        sys.executable,
        str(script),
        "--video",
        str(video_path),
        "--model",
        str(model_path),
        "--out-csv",
        str(out_pred_csv),
        "--raw-csv",
        str(out_raw_csv),
        "--gauss-csv",
        str(out_gauss_csv),
        "--bw-thr",
        str(float(bw_thr)),
        "--classify-thr",
        str(float(classify_thr)),
        "--bkgr-window-sec",
        str(float(bkgr_window_sec)),
        "--blur-sigma",
        str(float(blur_sigma)),
        "--patch-size",
        str(int(patch_size_px)),
        "--batch-size",
        str(int(batch_size)),
        "--gauss-crop-size",
        str(int(gauss_crop_size)),
        "--device",
        str(device),
    ]
    if max_frames is not None:
        cmd += ["--max-frames", str(int(max_frames))]

    if dry_run:
        print("[dry-run] Would run Raphael baseline detector:", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _run_stage5_validator(
    *,
    repo_root: Path,
    orig_video_path: Path,
    pred_csv_path: Path,
    gt_csv_path: Path,
    out_dir: Path,
    dist_thresholds: Sequence[float],
    crop_w: int,
    crop_h: int,
    gt_t_offset: int,
    max_frames: int | None,
    dry_run: bool,
) -> None:
    day_dir = (repo_root / DAY_PIPELINE_DIR_REL_PATH).resolve()
    stage5_py = day_dir / "stage5_validate.py"
    if not stage5_py.exists():
        raise FileNotFoundError(stage5_py)

    # Prevent stage5 validator from auto-searching for weights for FN scoring.
    # Passing a non-existent path avoids environment scanning / auto-discovery.
    no_weights = out_dir / "__no_fn_scoring_weights__.pt"
    max_frames_code = str(int(max_frames)) if max_frames is not None else "None"

    code = "\n".join(
        [
            "from pathlib import Path",
            "from stage5_validate import stage5_validate_against_gt",
            "stage5_validate_against_gt(",
            f"    orig_video_path=Path({repr(str(orig_video_path))}),",
            f"    pred_csv_path=Path({repr(str(pred_csv_path))}),",
            f"    gt_csv_path=Path({repr(str(gt_csv_path))}),",
            f"    out_dir=Path({repr(str(out_dir))}),",
            f"    dist_thresholds={list(float(x) for x in dist_thresholds)!r},",
            f"    crop_w={int(crop_w)},",
            f"    crop_h={int(crop_h)},",
            f"    gt_t_offset={int(gt_t_offset)},",
            f"    max_frames={max_frames_code},",
            "    only_firefly_rows=True,",
            "    show_per_frame=False,",
            f"    model_path=Path({repr(str(no_weights))}),",
            "    print_load_status=False,",
            ")",
        ]
    )

    if dry_run:
        print(f"[dry-run] Would run Stage5 validator in {day_dir} on {pred_csv_path.name}")
        return
    subprocess.run([sys.executable, "-c", code], cwd=str(day_dir), check=True)


def _run_stage6_overlay(
    *,
    repo_root: Path,
    orig_video_path: Path,
    pred_csv_path: Path,
    out_video_path: Path,
    thickness: int,
    gt_box_w: int,
    gt_box_h: int,
    only_firefly_rows: bool,
    max_frames: int | None,
    stage5_dir_hint: Path | None,
    render_threshold_overlays: bool,
    thr_box_w: int | None,
    thr_box_h: int | None,
    dry_run: bool,
) -> None:
    day_dir = (repo_root / DAY_PIPELINE_DIR_REL_PATH).resolve()
    stage6_py = day_dir / "stage6_overlay_gt_vs_model.py"
    if not stage6_py.exists():
        raise FileNotFoundError(stage6_py)

    max_frames_code = str(int(max_frames)) if max_frames is not None else "None"
    stage5_hint_code = (
        f"Path({repr(str(stage5_dir_hint))})" if stage5_dir_hint is not None else "None"
    )
    thr_box_w_code = str(int(thr_box_w)) if thr_box_w is not None else "None"
    thr_box_h_code = str(int(thr_box_h)) if thr_box_h is not None else "None"

    code = "\n".join(
        [
            "from pathlib import Path",
            "from stage6_overlay_gt_vs_model import overlay_gt_vs_model",
            "overlay_gt_vs_model(",
            f"    orig_video_path=Path({repr(str(orig_video_path))}),",
            f"    pred_csv_path=Path({repr(str(pred_csv_path))}),",
            f"    out_video_path=Path({repr(str(out_video_path))}),",
            "    gt_norm_csv_path=None,",
            f"    thickness={int(thickness)},",
            f"    gt_box_w={int(gt_box_w)},",
            f"    gt_box_h={int(gt_box_h)},",
            f"    only_firefly_rows={bool(only_firefly_rows)},",
            f"    max_frames={max_frames_code},",
            f"    stage5_dir_hint={stage5_hint_code},",
            f"    render_threshold_overlays={bool(render_threshold_overlays)},",
            f"    thr_box_w={thr_box_w_code},",
            f"    thr_box_h={thr_box_h_code},",
            ")",
        ]
    )

    if dry_run:
        print(f"[dry-run] Would run Stage6 overlay in {day_dir} -> {out_video_path.name}")
        return
    subprocess.run([sys.executable, "-c", code], cwd=str(day_dir), check=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrated ingest → train → validate orchestrator.")
    p.add_argument(
        "--species-name",
        type=str,
        default=str(SPECIES_NAME) if SPECIES_NAME else "",
        help="Species name for this run (source of truth; used for staging + dataset/model selection).",
    )
    p.add_argument(
        "--observed-dir",
        type=str,
        default=str(OBSERVED_DATA_DIR) if OBSERVED_DATA_DIR else "",
        help="Folder containing observed videos + annotator CSVs (matched by filename to .mp4).",
    )
    p.add_argument(
        "--type-of-video",
        type=str,
        default=str(TYPE_OF_VIDEO),
        help="Global video type for this run: day|night (controls dataset routing + model-zoo selection).",
    )
    p.add_argument(
        "--train-pair-fraction",
        type=float,
        default=float(TRAIN_PAIR_FRACTION),
        help="Fraction of discovered (video,csv) pairs assigned to training (remainder held out for validation/testing).",
    )

    p.add_argument("--skip-ingest", action="store_true", default=False, help="Skip Stage 1 ingestion.")
    p.add_argument("--skip-train", action="store_true", default=False, help="Skip Stage 2 training.")
    p.add_argument("--skip-test", action="store_true", default=False, help="Skip Stage 3 testing/evaluation.")

    p.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Single root folder under which data/models/inference outputs will be stored. "
            "If provided, defaults to: "
            f"data=<root>/{DEFAULT_DATA_SUBDIR!r}, "
            f"model_zoo=<root>/{DEFAULT_MODEL_ZOO_SUBDIR!r}, "
            f"inference=<root>/{DEFAULT_INFERENCE_OUTPUT_SUBDIR!r}."
        ),
    )
    p.add_argument("--dry-run", action="store_true", default=bool(DRY_RUN))
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    species_arg = str(args.species_name or "").strip() or str(SPECIES_NAME or "").strip()
    if not species_arg:
        raise SystemExit("Pass --species-name (or set SPECIES_NAME at top of file).")
    species_name = _safe_name(species_arg)
    if not species_name:
        raise SystemExit(f"Invalid species name: {species_arg!r}")

    observed_dir_arg = str(args.observed_dir or "").strip() or str(OBSERVED_DATA_DIR or "").strip()
    if not observed_dir_arg:
        raise SystemExit("Pass --observed-dir (or set OBSERVED_DATA_DIR at top of file).")
    observed_dir = Path(observed_dir_arg).expanduser().resolve()

    video_type = str(args.type_of_video or TYPE_OF_VIDEO)
    try:
        video_type_norm = _normalize_video_type(video_type)
    except Exception as e:
        raise SystemExit(str(e)) from e
    time_of_day = _time_of_day_from_video_type(video_type_norm)
    expected_route = video_type_norm  # "day" | "night"

    root_arg = (str(args.root).strip() if args.root else "") or (str(ROOT_PATH).strip() if ROOT_PATH else "")
    if not root_arg:
        raise SystemExit("Pass --root (recommended) or set ROOT_PATH at top of file.")
    root = Path(root_arg).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    repo_root = _REPO_ROOT
    gateway_path = (repo_root / GATEWAY_REL_PATH).resolve()
    if not gateway_path.exists():
        raise FileNotFoundError(gateway_path)

    data_root = (root / DEFAULT_DATA_SUBDIR).expanduser().resolve()
    model_root = (root / DEFAULT_MODEL_ZOO_SUBDIR).expanduser().resolve()
    inference_root = (root / DEFAULT_INFERENCE_OUTPUT_SUBDIR).expanduser().resolve()

    # Ensure required folder scaffolds exist under the chosen roots.
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "batch_exports").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_datasets" / "integrated pipeline datasets").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_datasets" / "single species datasets").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_validation_datasets" / "combined species folder").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_validation_datasets" / "individual species folder").mkdir(parents=True, exist_ok=True)
    inference_root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Stage 1: Ingest (extract patches + version datasets)
    # -------------------------------------------------------------------------

    # Discover and split observed (video,csv) pairs.
    pairs = stage1_ingestor.discover_observed_pairs(observed_dir)

    try:
        train_pairs, val_pairs = stage1_ingestor.split_pairs_train_vs_val(
            pairs, train_fraction=float(args.train_pair_fraction)
        )
    except Exception as e:
        raise SystemExit(f"Invalid --train-pair-fraction: {e}") from e
    train_video_names = {p.video_name for p in train_pairs}
    val_video_names = {p.video_name for p in val_pairs}
    print(
        "[orchestrator] Discovered pairs:",
        f"total={len(pairs)} train={len(train_pairs)} val={len(val_pairs)} train_fraction={float(args.train_pair_fraction):.3f}",
    )

    # Count rows for run naming + train decision.
    train_firefly_rows_total = sum(len(_read_annotator_csv(p.firefly_csv)) for p in train_pairs)
    train_background_rows_total = sum(
        len(_read_annotator_csv(p.background_csv)) for p in train_pairs if p.background_csv is not None
    )
    val_firefly_rows_total = sum(len(_read_annotator_csv(p.firefly_csv)) for p in val_pairs)

    has_train_rows = bool(train_firefly_rows_total or train_background_rows_total)

    skip_ingest = bool(getattr(args, "skip_ingest", False))
    skip_train = bool(getattr(args, "skip_train", False))
    skip_test = bool(getattr(args, "skip_test", False))
    if skip_ingest:
        print("[orchestrator] Skipping Stage 1 ingestion (--skip-ingest).")
    if skip_train:
        print("[orchestrator] Skipping Stage 2 training (--skip-train).")
    if skip_test:
        print("[orchestrator] Skipping Stage 3 testing (--skip-test).")

    zoo = _ensure_model_zoo_scaffold(model_root)
    history_path = zoo["history"]
    registry_path = zoo["registry"]
    registry = _load_registry(registry_path)
    for p in pairs:
        registry[p.video_name] = str(p.video_path)
    _save_registry(registry_path, registry)

    # Dataset roots produced by stage1_ingestor_core.
    integrated_root = data_root / "Integrated_prototype_datasets" / "integrated pipeline datasets"
    single_root = data_root / "Integrated_prototype_datasets" / "single species datasets" / species_name
    validation_combined_root = data_root / "Integrated_prototype_validation_datasets" / "combined species folder"
    validation_species_root = (
        data_root / "Integrated_prototype_validation_datasets" / "individual species folder" / species_name
    )

    # Stage CSVs into canonical names expected by stage1_ingestor_core.
    staging_dir = (
        data_root
        / "batch_exports"
        / "orchestrator_observed_dir_staging"
        / f"{_run_tag()}__{_safe_name(observed_dir.name)}"
    )

    if not skip_ingest:
        staged_pairs = stage1_ingestor.stage_pairs_for_ingestor(
            pairs,
            species_name=species_name,
            staging_dir=staging_dir,
            time_of_day=time_of_day,
            dry_run=bool(args.dry_run),
        )
        staged_by_video: Dict[str, stage1_ingestor.StagedPair] = {sp.pair.video_name: sp for sp in staged_pairs}

        scaler_overrides: Dict[str, Any] = {
            # Versioning behavior
            "VERSION_COPY_MODE": str(DATASET_VERSION_COPY_MODE),
            # Background auto-generation knobs
            "AUTO_GENERATE_BACKGROUND_PATCHES": bool(AUTO_GENERATE_BACKGROUND_PATCHES),
            "AUTO_BACKGROUND_TO_FIREFLY_RATIO": float(AUTO_BACKGROUND_TO_FIREFLY_RATIO),
            "AUTO_BACKGROUND_PATCH_SIZE_PX": int(AUTO_BACKGROUND_PATCH_SIZE_PX),
            "AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME": int(AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME),
            "AUTO_BACKGROUND_MAX_FRAME_SAMPLES": int(AUTO_BACKGROUND_MAX_FRAME_SAMPLES),
            "AUTO_BACKGROUND_SEED": int(AUTO_BACKGROUND_SEED),
            "AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS": bool(AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS),
            # SBD + preprocessing
            "AUTO_BACKGROUND_SBD_MIN_AREA_PX": float(AUTO_BACKGROUND_SBD_MIN_AREA_PX),
            "AUTO_BACKGROUND_SBD_MAX_AREA_SCALE": float(AUTO_BACKGROUND_SBD_MAX_AREA_SCALE),
            "AUTO_BACKGROUND_SBD_MIN_DIST": float(AUTO_BACKGROUND_SBD_MIN_DIST),
            "AUTO_BACKGROUND_SBD_MIN_REPEAT": int(AUTO_BACKGROUND_SBD_MIN_REPEAT),
            "AUTO_BACKGROUND_USE_CLAHE": bool(AUTO_BACKGROUND_USE_CLAHE),
            "AUTO_BACKGROUND_CLAHE_CLIP": float(AUTO_BACKGROUND_CLAHE_CLIP),
            "AUTO_BACKGROUND_CLAHE_TILE": tuple(AUTO_BACKGROUND_CLAHE_TILE),
            "AUTO_BACKGROUND_USE_TOPHAT": bool(AUTO_BACKGROUND_USE_TOPHAT),
            "AUTO_BACKGROUND_TOPHAT_KSIZE": int(AUTO_BACKGROUND_TOPHAT_KSIZE),
            "AUTO_BACKGROUND_USE_DOG": bool(AUTO_BACKGROUND_USE_DOG),
            "AUTO_BACKGROUND_DOG_SIGMA1": float(AUTO_BACKGROUND_DOG_SIGMA1),
            "AUTO_BACKGROUND_DOG_SIGMA2": float(AUTO_BACKGROUND_DOG_SIGMA2),
        }

        batch_integrated_target_ver: Path | None = None
        batch_single_target_ver: Path | None = None
        batch_val_comb_target_ver: Path | None = None
        batch_val_species_target_ver: Path | None = None

        if ONE_DATASET_VERSION_PER_BATCH:
            # One dataset version per "observed_dir ingest" run.
            batch_integrated_target_ver = _next_version_dir(integrated_root)
            batch_single_target_ver = _next_version_dir(single_root)
            if val_pairs:
                batch_val_comb_target_ver = _next_version_dir(validation_combined_root)
                batch_val_species_target_ver = _next_version_dir(validation_species_root)

            print(
                "[orchestrator] Batch ingest versions:",
                f"integrated={batch_integrated_target_ver.name} single_species={batch_single_target_ver.name}"
                + (
                    f" validation_combined={batch_val_comb_target_ver.name} validation_species={batch_val_species_target_ver.name}"
                    if val_pairs
                    and batch_val_comb_target_ver is not None
                    and batch_val_species_target_ver is not None
                    else ""
                ),
            )

            # TRAIN ingestion (append into a single target version; finalize split once at the end).
            train_pbar = (
                tqdm(train_pairs, desc="[stage1] ingest TRAIN", unit="video", dynamic_ncols=True)
                if tqdm is not None and train_pairs
                else None
            )
            train_it = train_pbar if train_pbar is not None else train_pairs
            for i, p in enumerate(train_it):
                sp = staged_by_video[p.video_name]
                finalize = (i == (len(train_pairs) - 1))
                if train_pbar is not None:
                    train_pbar.set_postfix(video=str(p.video_name), finalize=bool(finalize))
                stage1_ingestor.run_ingestor_core(
                    annotations_csv=sp.staged_firefly_csv,
                    video_path=p.video_path,
                    data_root=data_root,
                    train_fraction=1.0,
                    train_val_seed=1337,
                    auto_load_sibling=bool(AUTO_LOAD_SIBLING_CLASS_CSV),
                    single_species_target_version_dir=batch_single_target_ver,
                    integrated_target_version_dir=batch_integrated_target_ver,
                    validation_combined_target_version_dir=None,
                    validation_individual_target_version_dir=None,
                    skip_final_split_rebuild=(not finalize),
                    scaler_overrides=scaler_overrides,
                    dry_run=bool(args.dry_run),
                )
            if train_pbar is not None:
                train_pbar.close()

            # FINAL-VALIDATION ingestion (append into a single target validation version).
            val_pbar = (
                tqdm(val_pairs, desc="[stage1] ingest VALIDATION", unit="video", dynamic_ncols=True)
                if tqdm is not None and val_pairs
                else None
            )
            val_it = val_pbar if val_pbar is not None else val_pairs
            for p in val_it:
                sp = staged_by_video[p.video_name]
                if val_pbar is not None:
                    val_pbar.set_postfix(video=str(p.video_name))
                stage1_ingestor.run_ingestor_core(
                    annotations_csv=sp.staged_firefly_csv,
                    video_path=p.video_path,
                    data_root=data_root,
                    train_fraction=0.0,
                    train_val_seed=1337,
                    auto_load_sibling=False,
                    single_species_target_version_dir=None,
                    integrated_target_version_dir=None,
                    validation_combined_target_version_dir=batch_val_comb_target_ver,
                    validation_individual_target_version_dir=batch_val_species_target_ver,
                    skip_final_split_rebuild=True,
                    scaler_overrides=scaler_overrides,
                    dry_run=bool(args.dry_run),
                )
            if val_pbar is not None:
                val_pbar.close()
        else:
            # Legacy behavior: one dataset version per video ingested.
            pairs_pbar = (
                tqdm(pairs, desc="[stage1] ingest VIDEOS", unit="video", dynamic_ncols=True)
                if tqdm is not None and pairs
                else None
            )
            pairs_it = pairs_pbar if pairs_pbar is not None else pairs
            for p in pairs_it:
                sp = staged_by_video[p.video_name]
                if p.video_name in train_video_names:
                    if pairs_pbar is not None:
                        pairs_pbar.set_postfix(video=str(p.video_name), split="train")
                    stage1_ingestor.run_ingestor_core(
                        annotations_csv=sp.staged_firefly_csv,
                        video_path=p.video_path,
                        data_root=data_root,
                        train_fraction=1.0,
                        train_val_seed=1337,
                        auto_load_sibling=bool(AUTO_LOAD_SIBLING_CLASS_CSV),
                        scaler_overrides=scaler_overrides,
                        dry_run=bool(args.dry_run),
                    )
                elif p.video_name in val_video_names:
                    if pairs_pbar is not None:
                        pairs_pbar.set_postfix(video=str(p.video_name), split="validation")
                    stage1_ingestor.run_ingestor_core(
                        annotations_csv=sp.staged_firefly_csv,
                        video_path=p.video_path,
                        data_root=data_root,
                        train_fraction=0.0,
                        train_val_seed=1337,
                        auto_load_sibling=False,
                        scaler_overrides=scaler_overrides,
                        dry_run=bool(args.dry_run),
                    )
            if pairs_pbar is not None:
                pairs_pbar.close()

    integrated_ver = _latest_version_dir(integrated_root)
    single_ver = _latest_version_dir(single_root)
    validation_ver = _latest_version_dir(validation_combined_root)
    validation_species_ver = _latest_version_dir(validation_species_root)

    if integrated_ver is None:
        raise SystemExit(f"Integrated dataset version not found under: {integrated_root}")
    if single_ver is None:
        raise SystemExit(f"Single-species dataset version not found under: {single_root}")

    dataset_time_dir = "day_time_dataset" if time_of_day == "day_time" else "night_time_dataset"
    global_data_dir = integrated_ver / dataset_time_dir / "final dataset"
    species_data_dir = single_ver / "final dataset"

    # -------------------------------------------------------------------------
    # Stage 2: Train (patch classifier models)
    # -------------------------------------------------------------------------

    # Train models (if this batch added training rows)
    global_model_path: Path | None = None
    species_model_path: Path | None = None
    global_train_metrics: Dict[str, Any] | None = None
    species_train_metrics: Dict[str, Any] | None = None

    do_train = bool(TRAIN_MODELS_IF_TRAIN_ROWS_PRESENT) and has_train_rows and (not skip_train)

    if do_train:
        print(f"[orchestrator] TRAIN rows present; training new models (time_of_day={time_of_day})")
        if not global_data_dir.exists():
            raise SystemExit(f"Global dataset folder missing: {global_data_dir}")
        if not species_data_dir.exists():
            raise SystemExit(f"Species dataset folder missing: {species_data_dir}")

        if EVAL_GLOBAL_MODEL:
            global_parent = zoo["day_global"] if time_of_day == "day_time" else zoo["night_global"]
            global_ver = _next_version_dir(global_parent)
            global_ver.mkdir(parents=True, exist_ok=False)
            global_model_path = global_ver / "model.pt"
            global_metrics_path = global_ver / "training_metrics.json"
            global_train_metrics = _run_training(
                repo_root=repo_root,
                data_dir=global_data_dir,
                out_model_path=global_model_path,
                metrics_out_path=global_metrics_path,
                epochs=int(TRAIN_EPOCHS),
                batch_size=int(TRAIN_BATCH_SIZE),
                lr=float(TRAIN_LR),
                num_workers=int(TRAIN_NUM_WORKERS),
                resnet=str(TRAIN_RESNET),
                seed=int(TRAIN_SEED),
                dry_run=bool(args.dry_run),
            )

            patch_locations_train_csv = integrated_ver / dataset_time_dir / "patch_locations_train.csv"
            species_summary_csv = _species_summary_from_patch_locations(patch_locations_train_csv)
            species_summary_dir = _species_summary_from_dataset_dir(global_data_dir)

            # Prefer the dataset-dir scan (training uses ImageFolder over the folder contents),
            # but keep the patch_locations summary for traceability/debugging.
            species_summary = species_summary_dir if species_summary_dir.get("counts") else species_summary_csv

            warn_mismatch = None
            if species_summary_dir.get("counts") and species_summary_csv.get("counts"):
                if species_summary_dir.get("counts") != species_summary_csv.get("counts"):
                    warn_mismatch = (
                        "patch_locations_train.csv species counts do not match dataset_dir/train/firefly. "
                        "This usually means legacy-imported patches exist on disk but are missing from patch_locations*.csv."
                    )
            (global_ver / "model_card.txt").write_text(
                json.dumps(
                    {
                        "kind": "global",
                        "time_of_day": time_of_day,
                        "dataset_version": integrated_ver.name,
                        "dataset_dir": str(global_data_dir),
                        "trained_species": species_summary.get("species", []),
                        "trained_species_counts": species_summary.get("counts", {}),
                        "trained_species_source_csv": species_summary_csv.get("source_csv"),
                        "trained_species_error": species_summary_csv.get("error"),
                        "trained_species_source_train_dir": str(global_data_dir / "train" / "firefly"),
                        "trained_species_counts_by_split": species_summary_dir.get("counts_by_split", {}),
                        "trained_species_counts_total": species_summary_dir.get("counts_total", {}),
                        "trained_species_unknown_counts_by_split": species_summary_dir.get("unknown_counts_by_split", {}),
                        "trained_species_warning": warn_mismatch,
                        "train_metrics": global_train_metrics,
                    },
                    indent=2,
                )
            )

        if EVAL_SINGLE_SPECIES_MODEL:
            ff_n = _count_images_in_dir(species_data_dir / "train" / "firefly")
            bg_n = _count_images_in_dir(species_data_dir / "train" / "background")
            if ff_n <= 0 or bg_n <= 0:
                print(
                    "[orchestrator] WARNING: skipping single-species training (need both classes).",
                    f"train_firefly_images={ff_n} train_background_images={bg_n} dataset={species_data_dir}",
                )
            else:
                sp_root = zoo["single_root"] / species_name
                sp_root.mkdir(parents=True, exist_ok=True)
                sp_ver = _next_version_dir(sp_root)
                sp_ver.mkdir(parents=True, exist_ok=False)
                species_model_path = sp_ver / "model.pt"
                species_metrics_path = sp_ver / "training_metrics.json"
                species_train_metrics = _run_training(
                    repo_root=repo_root,
                    data_dir=species_data_dir,
                    out_model_path=species_model_path,
                    metrics_out_path=species_metrics_path,
                    epochs=int(TRAIN_EPOCHS),
                    batch_size=int(TRAIN_BATCH_SIZE),
                    lr=float(TRAIN_LR),
                    num_workers=int(TRAIN_NUM_WORKERS),
                    resnet=str(TRAIN_RESNET),
                    seed=int(TRAIN_SEED),
                    dry_run=bool(args.dry_run),
                )
                (sp_ver / "model_card.txt").write_text(
                    json.dumps(
                        {
                            "kind": "single_species",
                            "species": species_name,
                            "time_of_day": time_of_day,
                            "dataset_version": single_ver.name,
                            "dataset_dir": str(species_data_dir),
                            "train_metrics": species_train_metrics,
                        },
                        indent=2,
                    )
                )

    # If we didn't train, use latest models from the zoo
    if global_model_path is None and EVAL_GLOBAL_MODEL:
        global_parent = zoo["day_global"] if time_of_day == "day_time" else zoo["night_global"]
        latest = _latest_version_dir(global_parent)
        if latest is None:
            print(f"[orchestrator] No existing global model found under: {global_parent} (skipping global eval)")
        else:
            global_model_path = latest / "model.pt"
            if not global_model_path.exists():
                pts = sorted(latest.glob("*.pt"))
                global_model_path = pts[0] if pts else None

    if species_model_path is None and EVAL_SINGLE_SPECIES_MODEL:
        sp_root = zoo["single_root"] / species_name
        latest = _latest_version_dir(sp_root)
        if latest is None:
            print(f"[orchestrator] No existing species model found under: {sp_root} (skipping species eval)")
        else:
            species_model_path = latest / "model.pt"
            if not species_model_path.exists():
                pts = sorted(latest.glob("*.pt"))
                species_model_path = pts[0] if pts else None

    # -------------------------------------------------------------------------
    # Stage 3: Test (run pipelines + score vs GT)
    # -------------------------------------------------------------------------

    # Determine evaluation set (validation pairs)
    eval_items: List[Tuple[str, Path, List[Dict[str, int]]]] = []
    eval_source: Dict[str, Any] = {}
    eval_error: str | None = None
    eval_source = {
        "kind": "observed_dir_split",
        "observed_dir": str(observed_dir),
        "dataset_name": _safe_name(observed_dir.name),
        "species_name": species_name,
        "type_of_video": expected_route,
        "time_of_day": time_of_day,
        "n_pairs_total": len(pairs),
        "n_pairs_train": len(train_pairs),
        "n_pairs_validation": len(val_pairs),
    }
    if not skip_test:
        for p in val_pairs:
            gt_rows = _read_annotator_csv(p.firefly_csv)
            if not gt_rows:
                continue
            try:
                route = _route_for_video(
                    p.video_path, thr=float(GATEWAY_BRIGHTNESS_THRESHOLD), frames=int(GATEWAY_BRIGHTNESS_NUM_FRAMES)
                )
            except Exception:
                continue
            if expected_route == "day" and route != "day":
                print(f"[orchestrator] WARNING: skipping {p.video_path.name} (routed {route}, expected day)")
                continue
            if expected_route == "night" and route != "night":
                print(f"[orchestrator] WARNING: skipping {p.video_path.name} (routed {route}, expected night)")
                continue
            eval_items.append((p.video_name, p.video_path, gt_rows))

        if not eval_items:
            if eval_error is None:
                eval_error = "no_eval_videos_selected"
            print(f"[orchestrator] WARNING: {eval_error}")
    else:
        eval_error = "skipped"

    run_ident = RunIdentity(dataset_name=_safe_name(observed_dir.name), species_name=species_name, time_of_day=time_of_day)
    run_id = _run_dir_name(
        run_ident,
        n_pairs_total=len(pairs),
        n_pairs_train=len(train_pairs),
        n_pairs_val=len(val_pairs),
        train_rows_firefly=int(train_firefly_rows_total),
        train_rows_background=int(train_background_rows_total),
        val_rows_firefly=int(val_firefly_rows_total),
        did_train=bool(do_train),
    )
    results: List[Dict[str, Any]] = []

    def _eval_one_model(model_key: str, *, day_patch_model: Path | None, night_cnn_model: Path | None) -> None:
        pbar = (
            tqdm(eval_items, desc=f"[stage3] {model_key}", unit="video", dynamic_ncols=True)
            if tqdm is not None and eval_items
            else None
        )
        it = pbar if pbar is not None else eval_items
        for video_name, vp, gt_rows in it:
            out_root = inference_root / run_id / model_key / video_name
            out_root.mkdir(parents=True, exist_ok=True)

            # Write GT for both pipeline roots (only one will be used depending on route)
            day_root = out_root / DAY_OUTPUT_SUBDIR
            night_root = out_root / NIGHT_OUTPUT_SUBDIR
            day_gt_dir = day_root / "ground truth"
            night_gt_dir = night_root / "ground truth"
            _write_gt_csv(day_gt_dir / f"gt_{vp.stem}.csv", gt_rows)
            _write_gt_csv(day_gt_dir / "gt.csv", gt_rows)
            # Night pipeline's stage0_cleanup expects a root-level CSV to seed ground truth.
            _write_gt_csv(day_root / "gt.csv", gt_rows)
            _write_gt_csv(night_gt_dir / "gt.csv", gt_rows)
            _write_gt_csv(night_root / "gt.csv", gt_rows)

            t0 = time.time()
            try:
                _run_gateway(
                    repo_root=repo_root,
                    gateway_path=gateway_path,
                    video_path=vp,
                    output_root=out_root,
                    day_patch_model=day_patch_model,
                    night_cnn_model=night_cnn_model,
                    thr=float(GATEWAY_BRIGHTNESS_THRESHOLD),
                    frames=int(GATEWAY_BRIGHTNESS_NUM_FRAMES),
                    max_concurrent=int(GATEWAY_MAX_CONCURRENT),
                    force_tests=bool(FORCE_GATEWAY_TESTS),
                    dry_run=bool(args.dry_run),
                )
            except subprocess.CalledProcessError as e:
                dt = time.time() - t0
                route = "unknown"
                try:
                    route = _route_for_video(
                        vp, thr=float(GATEWAY_BRIGHTNESS_THRESHOLD), frames=int(GATEWAY_BRIGHTNESS_NUM_FRAMES)
                    )
                except Exception:
                    pass
                if pbar is not None:
                    pbar.set_postfix(video=str(video_name), route=str(route), exit=int(e.returncode), dt_s=f"{dt:.1f}")
                results.append(
                    {
                        "model_key": model_key,
                        "video_name": video_name,
                        "video_path": str(vp),
                        "route": route,
                        "output_root": str(out_root),
                        "duration_s": float(dt),
                        "validation_metrics": {
                            "error": "gateway_failed",
                            "exit_code": int(e.returncode),
                            "cmd": [str(x) for x in (e.cmd or [])],
                        },
                    }
                )
                continue
            dt = time.time() - t0

            # Parse validation metrics from whichever pipeline was chosen
            route = _route_for_video(vp, thr=float(GATEWAY_BRIGHTNESS_THRESHOLD), frames=int(GATEWAY_BRIGHTNESS_NUM_FRAMES))
            if route == "day":
                stage_dir = out_root / DAY_OUTPUT_SUBDIR / "stage5 validation" / vp.stem
            else:
                stage_dir = out_root / NIGHT_OUTPUT_SUBDIR / "stage9 validation" / vp.stem
            metrics = _parse_validation_metrics(stage_dir)
            if pbar is not None:
                best = metrics.get("best") if isinstance(metrics, dict) else None
                if isinstance(best, dict):
                    pbar.set_postfix(
                        video=str(video_name),
                        route=str(route),
                        tp=int(best.get("tp") or 0),
                        fp=int(best.get("fp") or 0),
                        fn=int(best.get("fn") or 0),
                        f1=float(best.get("f1") or 0.0),
                        dt_s=f"{dt:.1f}",
                    )
                else:
                    pbar.set_postfix(video=str(video_name), route=str(route), dt_s=f"{dt:.1f}")

            results.append(
                {
                    "model_key": model_key,
                    "video_name": video_name,
                    "video_path": str(vp),
                    "route": route,
                    "output_root": str(out_root),
                    "duration_s": float(dt),
                    "validation_metrics": metrics,
                }
            )
        if pbar is not None:
            pbar.close()

    if EVAL_GLOBAL_MODEL and global_model_path is not None:
        if time_of_day == "day_time":
            _eval_one_model("global_model", day_patch_model=global_model_path, night_cnn_model=None)
        else:
            _eval_one_model("global_model", day_patch_model=None, night_cnn_model=global_model_path)

    if EVAL_SINGLE_SPECIES_MODEL and species_model_path is not None:
        if time_of_day == "day_time":
            _eval_one_model("single_species_model", day_patch_model=species_model_path, night_cnn_model=None)
        else:
            _eval_one_model("single_species_model", day_patch_model=None, night_cnn_model=species_model_path)

    def _summarize_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        by_model: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            mk = str(r.get("model_key") or "")
            if mk:
                by_model.setdefault(mk, []).append(r)

        summary: Dict[str, Any] = {}
        for mk, items in by_model.items():
            bests: List[Dict[str, Any]] = []
            for it in items:
                vm = it.get("validation_metrics")
                if isinstance(vm, dict) and isinstance(vm.get("best"), dict):
                    bests.append(
                        {
                            "video_name": it.get("video_name"),
                            "thr": vm["best"].get("thr"),
                            "tp": vm["best"].get("tp"),
                            "fp": vm["best"].get("fp"),
                            "fn": vm["best"].get("fn"),
                            "precision": vm["best"].get("precision"),
                            "recall": vm["best"].get("recall"),
                            "f1": vm["best"].get("f1"),
                        }
                    )

            f1s = [float(b.get("f1") or 0.0) for b in bests]
            mean_f1 = (sum(f1s) / len(f1s)) if f1s else None
            summary[mk] = {
                "n_videos": len(items),
                "n_with_metrics": len(bests),
                "mean_best_f1": mean_f1,
                "best_by_video": bests,
            }
        return summary

    results_summary = _summarize_results(results)

    # ───────────────────────────────── baseline evaluation ─────────────────────────────────
    baseline_assets: Dict[str, Any] = {
        "enabled": bool(RUN_BASELINE_EVAL),
        "only_for_night": bool(BASELINES_ONLY_FOR_NIGHT),
        "validator": {
            "dist_thresholds_px": list(BASELINE_DIST_THRESHOLDS_PX),
            "crop_w": int(BASELINE_VALIDATE_CROP_W),
            "crop_h": int(BASELINE_VALIDATE_CROP_H),
            "max_frames": int(BASELINE_MAX_FRAMES) if BASELINE_MAX_FRAMES is not None else None,
        },
        "lab_method": {
            "enabled": bool(RUN_BASELINE_LAB_METHOD),
            "threshold": float(LAB_BASELINE_THRESHOLD),
            "blur_sigma": float(LAB_BASELINE_BLUR_SIGMA),
            "bkgr_window_sec": float(LAB_BASELINE_BKGR_WINDOW_SEC),
        },
        "raphael_method": {
            "enabled": bool(RUN_BASELINE_RAPHAEL_METHOD),
            "bw_thr": float(RAPHAEL_BW_THR),
            "classify_thr": float(RAPHAEL_CLASSIFY_THR),
            "bkgr_window_sec": float(RAPHAEL_BKGR_WINDOW_SEC),
            "blur_sigma": float(RAPHAEL_BLUR_SIGMA),
            "patch_size_px": int(RAPHAEL_PATCH_SIZE_PX),
            "batch_size": int(RAPHAEL_BATCH_SIZE),
            "gauss_crop_size": int(RAPHAEL_GAUSS_CROP_SIZE),
            "device": str(RAPHAEL_DEVICE),
            "model": {},
        },
    }

    baseline_results: List[Dict[str, Any]] = []
    if RUN_BASELINE_EVAL and eval_items:
        raphael_model_info = _ensure_raphael_model_copy(model_root=model_root, dry_run=bool(args.dry_run))
        baseline_assets["raphael_method"]["model"] = raphael_model_info

        raphael_model_path: Path | None = (
            Path(raphael_model_info["copied"]).expanduser().resolve()
            if raphael_model_info.get("copied")
            else None
        )

        warned_no_raphael_model = False

        base_pbar = (
            tqdm(eval_items, desc="[stage3] baselines", unit="video", dynamic_ncols=True)
            if tqdm is not None and eval_items
            else None
        )
        base_it = base_pbar if base_pbar is not None else eval_items
        for video_name, vp, gt_rows in base_it:
            if base_pbar is not None:
                base_pbar.set_postfix(video=str(video_name))
            try:
                route = _route_for_video(
                    vp, thr=float(GATEWAY_BRIGHTNESS_THRESHOLD), frames=int(GATEWAY_BRIGHTNESS_NUM_FRAMES)
                )
            except Exception:
                continue

            if BASELINES_ONLY_FOR_NIGHT and route != "night":
                continue

            # Baseline: lab method
            if RUN_BASELINE_LAB_METHOD:
                out_root = inference_root / run_id / BASELINE_OUTPUT_SUBDIR / "lab_method" / video_name
                out_root.mkdir(parents=True, exist_ok=True)
                gt_dir = out_root / "ground truth"
                gt_csv = gt_dir / "gt.csv"
                _write_gt_csv(gt_csv, gt_rows)

                pred_csv = out_root / "predictions.csv"
                stage5_dir = out_root / "stage5 validation" / vp.stem

                t0 = time.time()
                overlay_video = None
                overlay_error = None
                try:
                    _run_baseline_lab_detector(
                        repo_root=repo_root,
                        video_path=vp,
                        out_pred_csv=pred_csv,
                        threshold=float(LAB_BASELINE_THRESHOLD),
                        blur_sigma=float(LAB_BASELINE_BLUR_SIGMA),
                        bkgr_window_sec=float(LAB_BASELINE_BKGR_WINDOW_SEC),
                        max_frames=int(BASELINE_MAX_FRAMES) if BASELINE_MAX_FRAMES is not None else None,
                        dry_run=bool(args.dry_run),
                    )
                    _run_stage5_validator(
                        repo_root=repo_root,
                        orig_video_path=vp,
                        pred_csv_path=pred_csv,
                        gt_csv_path=gt_csv,
                        out_dir=stage5_dir,
                        dist_thresholds=BASELINE_DIST_THRESHOLDS_PX,
                        crop_w=int(BASELINE_VALIDATE_CROP_W),
                        crop_h=int(BASELINE_VALIDATE_CROP_H),
                        gt_t_offset=0,
                        max_frames=int(BASELINE_MAX_FRAMES) if BASELINE_MAX_FRAMES is not None else None,
                        dry_run=bool(args.dry_run),
                    )
                    metrics = _parse_validation_metrics(stage5_dir)
                except Exception as e:
                    metrics = {"error": str(e)}

                if BASELINE_RENDER_OVERLAY_VIDEO and isinstance(metrics, dict) and not metrics.get("error"):
                    legend = "LEGEND_GT=GREEN_MODEL=RED_OVERLAP=YELLOW"
                    overlay_dir = out_root / "stage6 overlay videos"
                    overlay_path = overlay_dir / f"{vp.stem}_gt_vs_model__{legend}.mp4"
                    try:
                        _run_stage6_overlay(
                            repo_root=repo_root,
                            orig_video_path=vp,
                            pred_csv_path=pred_csv,
                            out_video_path=overlay_path,
                            thickness=1,
                            gt_box_w=int(BASELINE_VALIDATE_CROP_W),
                            gt_box_h=int(BASELINE_VALIDATE_CROP_H),
                            only_firefly_rows=True,
                            max_frames=int(BASELINE_MAX_FRAMES) if BASELINE_MAX_FRAMES is not None else None,
                            stage5_dir_hint=(out_root / "stage5 validation"),
                            render_threshold_overlays=bool(BASELINE_OVERLAY_RENDER_THRESH_VIDEOS),
                            thr_box_w=None,
                            thr_box_h=None,
                            dry_run=bool(args.dry_run),
                        )
                        if not bool(args.dry_run) and overlay_path.exists():
                            overlay_video = str(overlay_path)
                    except Exception as e:
                        overlay_error = str(e)
                        print(f"[baseline-overlay] WARNING: lab overlay failed for {video_name}: {e}")

                baseline_results.append(
                    {
                        "model_key": "baseline_lab_method",
                        "video_name": video_name,
                        "video_path": str(vp),
                        "route": route,
                        "output_root": str(out_root),
                        "duration_s": float(time.time() - t0),
                        "validation_metrics": metrics,
                        "overlay_video_path": overlay_video,
                        "overlay_error": overlay_error,
                    }
                )
                if base_pbar is not None and isinstance(metrics, dict):
                    best = metrics.get("best") if isinstance(metrics.get("best"), dict) else None
                    if isinstance(best, dict):
                        base_pbar.set_postfix(
                            video=str(video_name),
                            route=str(route),
                            kind="lab",
                            tp=int(best.get("tp") or 0),
                            fp=int(best.get("fp") or 0),
                            fn=int(best.get("fn") or 0),
                            f1=float(best.get("f1") or 0.0),
                        )

            # Baseline: Raphael method
            if RUN_BASELINE_RAPHAEL_METHOD:
                if raphael_model_path is None or not raphael_model_path.exists():
                    if not warned_no_raphael_model:
                        print(
                            "[orchestrator] WARNING: Raphael baseline enabled but model not found. "
                            f"Set RAPHAEL_MODEL_SOURCE_PATH; got: {RAPHAEL_MODEL_SOURCE_PATH!r}"
                        )
                        warned_no_raphael_model = True
                    continue

                out_root = inference_root / run_id / BASELINE_OUTPUT_SUBDIR / "raphael_method" / video_name
                out_root.mkdir(parents=True, exist_ok=True)
                gt_dir = out_root / "ground truth"
                gt_csv = gt_dir / "gt.csv"
                _write_gt_csv(gt_csv, gt_rows)

                pred_csv = out_root / "predictions.csv"
                raw_csv = out_root / "raw_detections.csv"
                gauss_csv = out_root / "gauss_centroids.csv"
                stage5_dir = out_root / "stage5 validation" / vp.stem

                t0 = time.time()
                overlay_video = None
                overlay_error = None
                try:
                    _run_baseline_raphael_detector(
                        repo_root=repo_root,
                        video_path=vp,
                        model_path=raphael_model_path,
                        out_pred_csv=pred_csv,
                        out_raw_csv=raw_csv,
                        out_gauss_csv=gauss_csv,
                        bw_thr=float(RAPHAEL_BW_THR),
                        classify_thr=float(RAPHAEL_CLASSIFY_THR),
                        bkgr_window_sec=float(RAPHAEL_BKGR_WINDOW_SEC),
                        blur_sigma=float(RAPHAEL_BLUR_SIGMA),
                        patch_size_px=int(RAPHAEL_PATCH_SIZE_PX),
                        batch_size=int(RAPHAEL_BATCH_SIZE),
                        gauss_crop_size=int(RAPHAEL_GAUSS_CROP_SIZE),
                        max_frames=int(BASELINE_MAX_FRAMES) if BASELINE_MAX_FRAMES is not None else None,
                        device=str(RAPHAEL_DEVICE),
                        dry_run=bool(args.dry_run),
                    )
                    _run_stage5_validator(
                        repo_root=repo_root,
                        orig_video_path=vp,
                        pred_csv_path=pred_csv,
                        gt_csv_path=gt_csv,
                        out_dir=stage5_dir,
                        dist_thresholds=BASELINE_DIST_THRESHOLDS_PX,
                        crop_w=int(BASELINE_VALIDATE_CROP_W),
                        crop_h=int(BASELINE_VALIDATE_CROP_H),
                        gt_t_offset=0,
                        max_frames=int(BASELINE_MAX_FRAMES) if BASELINE_MAX_FRAMES is not None else None,
                        dry_run=bool(args.dry_run),
                    )
                    metrics = _parse_validation_metrics(stage5_dir)
                except Exception as e:
                    metrics = {"error": str(e)}

                if BASELINE_RENDER_OVERLAY_VIDEO and isinstance(metrics, dict) and not metrics.get("error"):
                    legend = "LEGEND_GT=GREEN_MODEL=RED_OVERLAP=YELLOW"
                    overlay_dir = out_root / "stage6 overlay videos"
                    overlay_path = overlay_dir / f"{vp.stem}_gt_vs_model__{legend}.mp4"
                    try:
                        _run_stage6_overlay(
                            repo_root=repo_root,
                            orig_video_path=vp,
                            pred_csv_path=pred_csv,
                            out_video_path=overlay_path,
                            thickness=1,
                            gt_box_w=int(BASELINE_VALIDATE_CROP_W),
                            gt_box_h=int(BASELINE_VALIDATE_CROP_H),
                            only_firefly_rows=True,
                            max_frames=int(BASELINE_MAX_FRAMES) if BASELINE_MAX_FRAMES is not None else None,
                            stage5_dir_hint=(out_root / "stage5 validation"),
                            render_threshold_overlays=bool(BASELINE_OVERLAY_RENDER_THRESH_VIDEOS),
                            thr_box_w=None,
                            thr_box_h=None,
                            dry_run=bool(args.dry_run),
                        )
                        if not bool(args.dry_run) and overlay_path.exists():
                            overlay_video = str(overlay_path)
                    except Exception as e:
                        overlay_error = str(e)
                        print(f"[baseline-overlay] WARNING: raphael overlay failed for {video_name}: {e}")

                baseline_results.append(
                    {
                        "model_key": "baseline_raphael_method",
                        "video_name": video_name,
                        "video_path": str(vp),
                        "route": route,
                        "output_root": str(out_root),
                        "duration_s": float(time.time() - t0),
                        "validation_metrics": metrics,
                        "overlay_video_path": overlay_video,
                        "overlay_error": overlay_error,
                    }
                )
                if base_pbar is not None and isinstance(metrics, dict):
                    best = metrics.get("best") if isinstance(metrics.get("best"), dict) else None
                    if isinstance(best, dict):
                        base_pbar.set_postfix(
                            video=str(video_name),
                            route=str(route),
                            kind="raphael",
                            tp=int(best.get("tp") or 0),
                            fp=int(best.get("fp") or 0),
                            fn=int(best.get("fn") or 0),
                            f1=float(best.get("f1") or 0.0),
                        )
        if base_pbar is not None:
            base_pbar.close()

    baseline_summary = _summarize_results(baseline_results) if baseline_results else {}

    # Write combined (multi-video) metrics per method at fixed thresholds.
    # This is a SUM of TP/FP/FN across the eval videos for each threshold, then recompute P/R/F1.
    combined_thresholds_px = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    combined_by_method: Dict[str, Dict[str, Any]] = {}
    for mk in ("global_model", "single_species_model"):
        combined_by_method[mk] = _combine_metrics_across_videos(
            [r for r in results if str(r.get("model_key")) == mk],
            thresholds_px=combined_thresholds_px,
        )
    for mk in ("baseline_lab_method", "baseline_raphael_method"):
        combined_by_method[mk] = _combine_metrics_across_videos(
            [r for r in baseline_results if str(r.get("model_key")) == mk],
            thresholds_px=combined_thresholds_px,
        )

    combined_txt = _format_combined_metrics_txt(combined_by_method, thresholds_px=combined_thresholds_px)
    run_dir = inference_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    combined_txt_path = run_dir / "combined_results__thr1-10.txt"
    if bool(args.dry_run):
        print(f"[dry-run] Would write combined results → {combined_txt_path}")
    else:
        combined_txt_path.write_text(combined_txt)
        print(f"[orchestrator] Combined results → {combined_txt_path}")

    record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "batch": {
            "observed_dir": str(observed_dir),
            "staging_dir": str(staging_dir),
            "dataset_name": _safe_name(observed_dir.name),
            "species_name": species_name,
            "type_of_video": expected_route,
            "time_of_day": time_of_day,
            "n_pairs_total": len(pairs),
            "n_pairs_train": len(train_pairs),
            "n_pairs_validation": len(val_pairs),
            "train_rows_firefly": int(train_firefly_rows_total),
            "train_rows_background": int(train_background_rows_total),
            "validation_rows_firefly": int(val_firefly_rows_total),
            "pairs": [
                {
                    "video_name": p.video_name,
                    "video_path": str(p.video_path),
                    "firefly_csv": str(p.firefly_csv),
                    "background_csv": str(p.background_csv) if p.background_csv else None,
                    "split": ("train" if p.video_name in train_video_names else "validation"),
                }
                for p in sorted(pairs, key=lambda x: x.video_name)
            ],
        },
        "paths": {
            "data_root": str(data_root),
            "model_zoo_root": str(model_root),
            "inference_output_root": str(inference_root),
            "integrated_dataset_version": integrated_ver.name if integrated_ver else None,
            "single_species_dataset_version": single_ver.name if single_ver else None,
            "validation_dataset_version": validation_ver.name if validation_ver else None,
            "validation_species_dataset_version": validation_species_ver.name if validation_species_ver else None,
        },
        "models": {
            "global": {
                "path": str(global_model_path) if global_model_path else None,
                "train_metrics": global_train_metrics,
            },
            "single_species": {
                "path": str(species_model_path) if species_model_path else None,
                "train_metrics": species_train_metrics,
            },
        },
        "gateway": {
            "brightness_threshold": float(GATEWAY_BRIGHTNESS_THRESHOLD),
            "brightness_frames": int(GATEWAY_BRIGHTNESS_NUM_FRAMES),
            "max_concurrent": int(GATEWAY_MAX_CONCURRENT),
            "force_tests": bool(FORCE_GATEWAY_TESTS),
        },
        "evaluation": {
            "eval_source": eval_source,
            "n_eval_videos": len(eval_items),
            "eval_error": eval_error,
        },
        "baselines": {
            "assets": baseline_assets,
            "results": baseline_results,
            "results_summary": baseline_summary,
        },
        "combined_results": {
            "thresholds_px": list(combined_thresholds_px),
            "by_method": combined_by_method,
            "text_path": str(combined_txt_path),
        },
        "results": results,
        "results_summary": results_summary,
    }
    # Save a full run record alongside inference outputs for easy inspection.
    record_path = run_dir / "run_record.json"
    if bool(args.dry_run):
        print(f"[dry-run] Would write run record → {record_path}")
    else:
        record_path.write_text(json.dumps(record, indent=2))
        print(f"[orchestrator] Run record → {record_path}")
    _append_jsonl(history_path, record)
    print(f"[orchestrator] Appended results → {history_path}")
    print(f"[orchestrator] Run outputs → {inference_root / run_id}")
    if results_summary:
        print("[orchestrator] Results summary (best-F1 per video):")
        for mk, s in results_summary.items():
            mean_f1 = s.get("mean_best_f1")
            mean_str = f"{float(mean_f1):.4f}" if isinstance(mean_f1, (int, float)) else "n/a"
            print(
                f"  - {mk}: mean_best_f1={mean_str}  n_videos={s.get('n_with_metrics', 0)}/{s.get('n_videos', 0)}"
            )
    if baseline_summary:
        print("[orchestrator] Baselines summary (best-F1 per video):")
        for mk, s in baseline_summary.items():
            mean_f1 = s.get("mean_best_f1")
            mean_str = f"{float(mean_f1):.4f}" if isinstance(mean_f1, (int, float)) else "n/a"
            print(
                f"  - {mk}: mean_best_f1={mean_str}  n_videos={s.get('n_with_metrics', 0)}/{s.get('n_videos', 0)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
