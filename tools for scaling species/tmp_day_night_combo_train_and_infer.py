#!/usr/bin/env python3
from __future__ import annotations

"""
Temporary one-shot runner (safe to delete later).

What it does:
1) Discovers latest single-species final datasets.
2) Splits species and videos into day/night groups from folder/file/species names.
3) Trains route-specific models:
   - 1 global model per route
   - 1 leaveout_<species> model per species in that route
4) Runs gateway inference (day v3 / night pipeline) with the trained models.
5) Runs legacy baselines (lab + Raphael) for each video.
6) Writes the final results CSV after all baseline/pipeline outputs are available.

Routing uses folder/file/species names only (no brightness routing filter).
"""

import argparse
import csv
import hashlib
import json
import importlib.util
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# GLOBALS (edit these if needed)
# =============================================================================

REPO_ROOT: Path = Path("/home/guest/Desktop/arnav's files/firefly pipeline")

# Integrated data roots.
INTEGRATED_OUTER_ROOT: Path = Path("/mnt/Samsung_SSD_2TB/integrated prototype data")
INTEGRATED_INNER_ROOT: Path = INTEGRATED_OUTER_ROOT / "integrated prototype data"
PATCH_DATA_ROOT: Path = INTEGRATED_INNER_ROOT / "patch training datasets and pipeline validation data"

# Raw videos root to infer on (can be overridden via CLI).
RAW_VIDEOS_ROOT: Path = Path("/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)")

# Final outputs for this temporary run.
# Keep all generated outputs under this single root.
RUNS_ROOT: Path = Path("/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run")
RUN_NAME_PREFIX: str = "tmp_day_night_combo_train_and_infer"
FINAL_RESULTS_FILENAME: str = "final_results.csv"
VIDEO_CATALOG_FILENAME: str = "tmp_scaling_species_training_inference_catalog.json"

# -----------------------------------------------------------------------------
# Run Component Switches (edit these first)
# -----------------------------------------------------------------------------
# 1) Model training
RUN_MODEL_TRAINING: bool = True
RUN_DAY_MODEL_TRAINING: bool = False
RUN_NIGHT_MODEL_TRAINING: bool = True
TRAIN_GLOBAL_MODELS: bool = True
TRAIN_LEAVEOUT_MODELS: bool = False

# 2) Baseline methods (Lab + Raphael)
# Set this True to run only baseline methods if day/night pipeline toggles below
# are both False.
RUN_BASELINE_METHODS_INFERENCE: bool = True
RUN_LAB_BASELINE: bool = True
RUN_RAPHAEL_BASELINE: bool = True

# Baseline species switches are independent from your pipeline inference
# switches. Turn on only the species you want each baseline method to run on.
LAB_BASELINE_SPECIES_SWITCHES: Dict[str, bool] = {
    "bicellonycha-wickershamorum": False,
    "photinus-acuminatus": False,
    "photinus-greeni": False,
    "photuris-bethaniensis": False,
    "forresti": False,
    "frontalis": False,
    "photinus-carolinus": False,
    "photinus-knulli": True,
    "tremulans": False,
}

RAPHAEL_BASELINE_SPECIES_SWITCHES: Dict[str, bool] = {
    "bicellonycha-wickershamorum": False,
    "photinus-acuminatus": False,
    "photinus-greeni": False,
    "photuris-bethaniensis": False,
    "forresti": False,
    "frontalis": False,
    "photinus-carolinus": False,
    "photinus-knulli": True,
    "tremulans": False,
}

# 3) Your pipeline inference (split by route)
RUN_DAY_PIPELINE_INFERENCE: bool = False
RUN_NIGHT_PIPELINE_INFERENCE: bool = True
RUN_GLOBAL_MODEL_INFERENCE: bool = True
RUN_LEAVEOUT_MODEL_INFERENCE: bool = False

# Per-route species inference switches.
# Only species with True are evaluated from the cataloged inference-only videos.
# These switches are route-specific to make the day/night split explicit.
DAY_INFERENCE_SPECIES_SWITCHES: Dict[str, bool] = {
    "bicellonycha-wickershamorum": False,
    "photinus-acuminatus": False,
    "photinus-greeni": False,
    "photuris-bethaniensis": False,
}

NIGHT_INFERENCE_SPECIES_SWITCHES: Dict[str, bool] = {
    "forresti": False,
    "frontalis": False,
    "photinus-carolinus": False,
    "photinus-knulli": True,
    "tremulans": False,
}

# Pretrained model selection when RUN_MODEL_TRAINING=False.
# If True, the script auto-discovers the newest compatible trained artifact
# for that model family and uses it. If False, it uses the manual path below.
AUTO_DISCOVER_LATEST_DAY_MODEL_ROOT: bool = True
AUTO_DISCOVER_LATEST_NIGHT_MODEL_ROOT: bool = True

# Manual fallback paths used when the corresponding auto-discovery switch is
# False. The day/night paths should point to the route-specific directory that
# contains global_all_species.pt plus any leaveout_*.pt files for that route.
DAY_PRETRAINED_MODEL_ROOT: Path = Path(
    "/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/"
    "tmp_day_night_combo_train_and_infer__20260328__004250/models/day"
)
NIGHT_PRETRAINED_MODEL_ROOT: Path = Path(
    "/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/"
    "tmp_day_night_combo_train_and_infer__20260305__163237/models/night"
)

# -----------------------------------------------------------------------------
# Day YOLO Switches (edit these together)
# -----------------------------------------------------------------------------
RUN_DAY_YOLO_MODEL_TRAINING: bool = False
TRAIN_DAY_YOLO_GLOBAL_MODEL: bool = False
TRAIN_DAY_YOLO_LEAVEOUT_MODELS: bool = False

DAY_YOLO_TRAINING_SPECIES_SWITCHES: Dict[str, bool] = {
    "bicellonycha-wickershamorum": True,
    "photinus-acuminatus": True,
    "photinus-greeni": True,
    "photuris-bethaniensis": True,
    "pyrallis-gopro": True,
}

# Do not change these paths. The YOLO dataset/model helpers depend on this
# exact folder layout and changing them will break the save/load logic.
DAY_YOLO_DATASET_ROOT: Path = Path(
    "/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/dataset"
)
DAY_YOLO_GLOBAL_MODELS_ROOT: Path = Path(
    "/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/models/global models"
)
DAY_YOLO_LEAVEOUT_MODELS_ROOT: Path = Path(
    "/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/models/leave one out models"
)
DAY_YOLO_LEGACY_MODELS_ROOT: Path = Path(
    "/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/models/legacy models"
)

AUTO_DISCOVER_LATEST_DAY_YOLO_MODEL: bool = True
DAY_YOLO_MODEL_WEIGHTS: Path = DAY_YOLO_GLOBAL_MODELS_ROOT / "20260414" / "best_firefly_yolo.pt"
DAY_YOLO_MODEL_WEIGHTS_INIT: str = "yolov8s.pt"
DAY_YOLO_EPOCHS: int = 50
DAY_YOLO_IMG_SIZE: int | None = None
DAY_YOLO_BATCH_SIZE: int = 1
DAY_YOLO_DEVICE: str | int | None = 0
DAY_YOLO_WORKERS: int = 2
DAY_YOLO_PATIENCE: int = 20
DAY_YOLO_LR0: float | None = 0.01
DAY_YOLO_WEIGHT_DECAY: float | None = 0.0005
DAY_YOLO_MPS_CPU_FALLBACK: bool = True
DAY_YOLO_REUSE_EXISTING_MODELS_IF_PRESENT: bool = True

# Evaluation safeguard: require per-video GT and cap processing to the last
# annotated frame in that GT.
REQUIRE_GT_FOR_INFERENCE: bool = True

# Route assignment by species token.
ROUTE_BY_SPECIES: Dict[str, str] = {
    "bicellonycha-wickershamorum": "day",
    "forresti": "night",
    "frontalis": "night",
    "photinus-acuminatus": "day",
    "photinus-carolinus": "night",
    "photinus-greeni": "day",
    "photinus-knulli": "night",
    "photuris-bethaniensis": "day",
    "tremulans": "night",
}

# Optional exact override by video stem (without extension).
ROUTE_BY_VIDEO_STEM: Dict[str, str] = {}

ROUTE_NAME_HINT_DAY_TOKENS: Sequence[str] = ("day", "daytime", "day_time")
ROUTE_NAME_HINT_NIGHT_TOKENS: Sequence[str] = ("night", "nighttime", "night_time")
ROUTE_DEFAULT: str = "night"
REQUIRE_EXPLICIT_ROUTE: bool = True

# Build settings for merged datasets.
REQUIRE_HARDLINKS: bool = True

# If True, skip retraining if a model file already exists.
REUSE_EXISTING_MODELS_IF_PRESENT: bool = True

# Training hyperparameters.
TRAIN_EPOCHS: int = 50
TRAIN_BATCH_SIZE: int = 128
TRAIN_LR: float = 3e-4
TRAIN_NUM_WORKERS: int = 2
TRAIN_RESNET: str = "resnet18"  # resnet18|34|50|101|152
TRAIN_SEED: int = 1337

# Dataset sanitizer knobs (same convention as stage2 trainer).
SANITIZE_DATASET_IMAGES: bool = True
SANITIZE_DATASET_MODE: str = "quarantine"  # "quarantine" | "delete"
SANITIZE_DATASET_VERIFY_WITH_PIL: bool = True
SANITIZE_DATASET_REPORT_MAX: int = 20

# Gateway settings.
# Threshold/frames are deprecated and ignored by current gateway routing, but
# gateway still accepts these args.
GATEWAY_BRIGHTNESS_THRESHOLD: float = 10.0
GATEWAY_BRIGHTNESS_FRAMES: int = 5
GATEWAY_MAX_CONCURRENT: int = 1
FORCE_GATEWAY_TESTS: bool = False

# Do not change these paths. Baseline artifacts are now stored permanently
# under this fixed SSD layout instead of under per-run temporary folders.
BASELINES_DATA_ROOT: Path = Path("/mnt/Samsung_SSD_2TB/integrated prototype data/Baselines data")
RAPHAEL_MODEL_ROOT: Path = BASELINES_DATA_ROOT / "Raphael's model"
RAPHAEL_METHOD_DATA_ROOT: Path = BASELINES_DATA_ROOT / "Raphael's method"
LAB_BASELINE_DATA_ROOT: Path = BASELINES_DATA_ROOT / "lab_baseline"

# Lab baseline (no model file required)
LAB_BASELINE_THRESHOLD: float = 0.12
LAB_BASELINE_BLUR_SIGMA: float = 1.0
LAB_BASELINE_BKGR_WINDOW_SEC: float = 2.0

# Raphael baseline (requires TorchScript ffnet model)
RAPHAEL_MODEL_PATH: Path = RAPHAEL_MODEL_ROOT / "ffnet_best.pth"
RAPHAEL_BW_THR: float = 0.2
RAPHAEL_CLASSIFY_THR: float = 0.98
RAPHAEL_BKGR_WINDOW_SEC: float = 2.0
RAPHAEL_BLUR_SIGMA: float = 0.0
RAPHAEL_PATCH_SIZE_PX: int = 33
RAPHAEL_BATCH_SIZE: int = 1000
RAPHAEL_GAUSS_CROP_SIZE: int = 10
RAPHAEL_DEVICE: str = "auto"

# Baseline evaluation settings (10px only to keep outputs smaller).
BASELINE_DIST_THRESHOLDS_PX: List[float] = [10.0]
BASELINE_VALIDATE_CROP_W: int = 10
BASELINE_VALIDATE_CROP_H: int = 10
BASELINE_RENDERED_VIDEO_FILENAME: str = "predictions_overlay.mp4"
BASELINE_SPECIES_DATA_DIRNAME: str = "data"
BASELINE_SPECIES_RESULTS_FILENAME: str = "results.json"
BASELINE_RESULTS_REGISTRY_FILENAME: str = "baseline_results_registry.json"

# If True, prints actions but does not train or run inference/baselines.
DRY_RUN: bool = False


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
CLASSES = ("firefly", "background")


FINAL_RESULTS_FIELDNAMES = [
    "run_id",
    "route",
    "species_name",
    "video_name",
    "eval_type",
    "model_used",
    "results",
    "inference_output_path",
    "lab_results",
    "lab_output_path",
    "raphael_results",
    "raphael_output_path",
    "gt_source",
    "gt_rows",
    "gt_max_t",
]


@dataclass(frozen=True)
class SourceSpec:
    species_name: str
    route: str
    path: Path  # .../final dataset


@dataclass(frozen=True)
class RoutedVideo:
    video_path: Path
    route: str
    species_name: Optional[str]


@dataclass(frozen=True)
class EvalVideo:
    species_name: str
    video_name: str
    video_path: Path
    video_key: str
    route: str
    gt_rows: List[Dict[str, int]]
    gt_source: Optional[Path]
    gt_max_t: Optional[int]


@dataclass(frozen=True)
class ModelSpec:
    route: str
    model_key: str  # global_all_species | leaveout_<species>
    ckpt_path: Path
    leaveout_species: Optional[str]


@dataclass(frozen=True)
class YoloSpeciesSource:
    species_name: str
    species_dir: Path
    images_dir: Path
    labels_dir: Path


@dataclass(frozen=True)
class YoloModelSpec:
    model_key: str  # global_all_species | leaveout_<species>
    ckpt_path: Path
    output_dir: Path
    leaveout_species: Optional[str]
    species_included: Tuple[str, ...]


DAY_YOLO_SPECIES_ALIASES: Dict[str, str] = {
    "day-photinus-greeni": "photinus-greeni",
    "night-photinus-greeni": "photinus-greeni",
    "photinus-greeni": "photinus-greeni",
    "bicellonycha-wickershamorum": "bicellonycha-wickershamorum",
    "photinus-acuminatus": "photinus-acuminatus",
    "photuris-bethaniensis": "photuris-bethaniensis",
    "pyrallis-gopro": "pyrallis-gopro",
    "pyrallis-gopro-v3": "pyrallis-gopro",
    "pyractomena-pyralis": "pyrallis-gopro",
}


def _q(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _safe_name(s: str) -> str:
    x = str(s).strip().replace(" ", "_")
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def _short_key(species_name: str, video_name: str) -> str:
    sp = _safe_name(species_name)
    vn = _safe_name(video_name)
    h = hashlib.sha1(vn.encode("utf-8", errors="ignore")).hexdigest()[:10]
    if len(vn) > 120:
        vn = vn[:120].rstrip("_")
    return f"{sp}__{vn}__{h}"


def _slug(s: str) -> str:
    x = str(s).strip().lower()
    x = x.replace("_", "-").replace(" ", "-")
    x = re.sub(r"[^a-z0-9-]+", "-", x)
    x = re.sub(r"-+", "-", x).strip("-")
    return x


def _contains_token(slug_text: str, token: str) -> bool:
    t = _slug(token)
    if not t:
        return False
    full = f"-{slug_text}-"
    return f"-{t}-" in full


def _normalize_route(route: str) -> str:
    x = str(route or "").strip().lower()
    if x not in {"day", "night"}:
        raise ValueError(f"Invalid route '{route}', expected day|night")
    return x


def _route_hint_from_text(text: str) -> Optional[str]:
    src = _slug(text)
    has_day = any(_contains_token(src, t) for t in ROUTE_NAME_HINT_DAY_TOKENS)
    has_night = any(_contains_token(src, t) for t in ROUTE_NAME_HINT_NIGHT_TOKENS)
    if has_day and has_night:
        raise ValueError(f"Conflicting day/night hints in: {text}")
    if has_day:
        return "day"
    if has_night:
        return "night"
    return None


def _infer_species_from_path(path: Path, known_species: Sequence[str]) -> Optional[str]:
    src = _slug(str(path))
    best: Optional[str] = None
    best_len = -1
    for sp in known_species:
        s = _slug(sp)
        if not s:
            continue
        if _contains_token(src, s) or (s in src):
            if len(s) > best_len:
                best = sp
                best_len = len(s)
    return best


def _route_for_item(*, species_name: Optional[str], video_stem: Optional[str], text_for_hint: str) -> str:
    if video_stem:
        v = ROUTE_BY_VIDEO_STEM.get(str(video_stem))
        if v is not None:
            return _normalize_route(v)

    hint = _route_hint_from_text(text_for_hint)
    if hint is not None:
        return hint

    if species_name:
        v = ROUTE_BY_SPECIES.get(str(species_name))
        if v is not None:
            return _normalize_route(v)

    if REQUIRE_EXPLICIT_ROUTE:
        raise ValueError(
            f"No route resolved for species={species_name!r}, video_stem={video_stem!r}. "
            "Add ROUTE_BY_VIDEO_STEM/ROUTE_BY_SPECIES or day/night tokens in names."
        )
    return _normalize_route(ROUTE_DEFAULT)


def _ensure_repo_import_path() -> None:
    repo_root = Path(REPO_ROOT).expanduser().resolve()
    sp = str(repo_root)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _repo_root() -> Path:
    return Path(REPO_ROOT).expanduser().resolve()


def _gateway_py() -> Path:
    p = _repo_root() / "Pipelines" / "Pipeline Gateway" / "gateway.py"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _day_pipeline_dir() -> Path:
    p = _repo_root() / "Pipelines" / "day time pipeline v3 (yolo + patch classifier ensemble)"
    if not (p / "stage5_validate.py").exists():
        raise FileNotFoundError(p / "stage5_validate.py")
    return p


def _baseline_scripts() -> Dict[str, Path]:
    base = _repo_root() / "tools for scaling species" / "legacy_baselines"
    lab = base / "nolan_mp4_to_predcsv.py"
    raphael = base / "raphael_oorb_detect_and_gauss.py"
    match = base / "match_predictions_to_processed_gt.py"
    render = base / "render_baseline_predictions.py"
    return {"lab": lab, "raphael": raphael, "match": match, "render": render}


def _baseline_method_registry_key(method_key: str) -> str:
    if method_key == "lab":
        return "lab_baseline"
    if method_key == "raphael":
        return "raphael_method"
    raise ValueError(f"Unknown baseline method key: {method_key}")


def _baseline_method_display_name(method_key: str) -> str:
    if method_key == "lab":
        return "lab_baseline"
    if method_key == "raphael":
        return "Raphael's method"
    raise ValueError(f"Unknown baseline method key: {method_key}")


def _baseline_method_root(method_key: str) -> Path:
    if method_key == "lab":
        return LAB_BASELINE_DATA_ROOT
    if method_key == "raphael":
        return RAPHAEL_METHOD_DATA_ROOT
    raise ValueError(f"Unknown baseline method key: {method_key}")


def _baseline_species_root(method_key: str, species_name: str) -> Path:
    return _baseline_method_root(method_key) / _safe_name(species_name or "unknown_species")


def _baseline_species_data_root(method_key: str, species_name: str) -> Path:
    return _baseline_species_root(method_key, species_name) / BASELINE_SPECIES_DATA_DIRNAME


def _baseline_species_results_path(method_key: str, species_name: str) -> Path:
    return _baseline_species_root(method_key, species_name) / BASELINE_SPECIES_RESULTS_FILENAME


def _baseline_species_log_root(method_key: str, species_name: str) -> Path:
    return _baseline_species_root(method_key, species_name) / "_logs"


def _baseline_root_registry_path() -> Path:
    return BASELINES_DATA_ROOT / BASELINE_RESULTS_REGISTRY_FILENAME


def _prepare_baseline_species_storage(
    *,
    method_key: str,
    species_names: Sequence[str],
    dry_run: bool,
) -> None:
    for species_name in sorted({str(s or "").strip() for s in species_names if str(s or "").strip()}):
        species_root = _baseline_species_root(method_key, species_name)
        data_root = _baseline_species_data_root(method_key, species_name)
        log_root = _baseline_species_log_root(method_key, species_name)
        if dry_run:
            print(f"[dry-run] Would reset baseline storage: {species_root}")
            continue
        species_root.mkdir(parents=True, exist_ok=True)
        if data_root.exists():
            shutil.rmtree(data_root)
        if log_root.exists():
            shutil.rmtree(log_root)
        data_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)


def _best_metrics_from_validation(metrics: Dict[str, Any] | None) -> Dict[str, Any]:
    if not metrics or not isinstance(metrics, dict):
        return {}
    best = metrics.get("best")
    return dict(best) if isinstance(best, dict) else {}


def _aggregate_baseline_species_records(video_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = 0
    videos_with_metrics = 0
    videos_with_errors = 0
    error_messages: List[str] = []

    for rec in video_records:
        metrics = rec.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if metrics.get("error"):
            videos_with_errors += 1
            error_messages.append(str(metrics.get("error")))
            continue
        best = metrics.get("best")
        if not isinstance(best, dict):
            continue
        videos_with_metrics += 1
        tp += int(best.get("tp") or 0)
        fp += int(best.get("fp") or 0)
        fn += int(best.get("fn") or 0)

    precision = (float(tp) / float(tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (float(tp) / float(tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = _accuracy(tp, fp, fn)
    if videos_with_metrics > 0 and videos_with_errors == 0:
        status = "ok"
    elif videos_with_metrics > 0:
        status = "partial"
    elif videos_with_errors > 0:
        status = "error"
    else:
        status = "empty"

    return {
        "status": status,
        "videos_total": int(len(video_records)),
        "videos_with_metrics": int(videos_with_metrics),
        "videos_with_errors": int(videos_with_errors),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "errors": error_messages,
    }


def _baseline_video_status(metrics: Dict[str, Any] | None) -> str:
    if not metrics or not isinstance(metrics, dict):
        return "empty"
    if metrics.get("error"):
        return "error"
    if isinstance(metrics.get("best"), dict):
        return "ok"
    return "empty"


def _write_baseline_species_results_and_registry(
    *,
    eval_videos: Sequence[EvalVideo],
    baseline_by_video_key: Dict[str, Dict[str, Dict[str, Any]]],
    dry_run: bool,
) -> None:
    methods_payload: Dict[str, Dict[str, List[Dict[str, Any]]]] = {"lab": {}, "raphael": {}}

    for video in eval_videos:
        method_map = baseline_by_video_key.get(video.video_key) or {}
        for method_key in ("lab", "raphael"):
            if method_key not in method_map:
                continue
            methods_payload.setdefault(method_key, {}).setdefault(video.species_name, []).append(
                {
                    "video_key": str(video.video_key),
                    "video_name": str(video.video_name),
                    "video_path": str(video.video_path),
                    "route": str(video.route),
                    "gt_source": str(video.gt_source or ""),
                    "gt_rows": int(len(video.gt_rows)),
                    "gt_max_t": (int(video.gt_max_t) if video.gt_max_t is not None else None),
                    "out_root": str(method_map[method_key].get("out_root") or ""),
                    "rendered_video": str(method_map[method_key].get("rendered_video") or ""),
                    "metrics": method_map[method_key].get("metrics") or {},
                    "best_metrics": _best_metrics_from_validation(method_map[method_key].get("metrics")),
                }
            )

    registry_path = _baseline_root_registry_path()
    registry: Dict[str, Any] = {
        "registry_version": "baseline_results_registry_v1",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "methods": {
            "lab_baseline": {},
            "raphael_method": {},
        },
    }
    if registry_path.exists():
        existing = _read_json_if_exists(registry_path)
        if isinstance(existing, dict):
            registry.update(existing)
            registry["methods"] = dict(existing.get("methods") or registry["methods"])

    now = datetime.now().isoformat(timespec="seconds")
    for method_key in ("lab", "raphael"):
        method_registry_key = _baseline_method_registry_key(method_key)
        method_display_name = _baseline_method_display_name(method_key)
        method_root = _baseline_method_root(method_key)
        registry_methods = dict(registry.get("methods") or {})
        method_registry = dict(registry_methods.get(method_registry_key) or {})

        for species_name, video_records in sorted((methods_payload.get(method_key) or {}).items()):
            species_root = _baseline_species_root(method_key, species_name)
            data_root = _baseline_species_data_root(method_key, species_name)
            results_path = _baseline_species_results_path(method_key, species_name)
            aggregate = _aggregate_baseline_species_records(video_records)
            species_payload = {
                "updated_at": now,
                "method_key": method_registry_key,
                "method_name": method_display_name,
                "method_root": str(method_root),
                "species_name": str(species_name),
                "species_slug": str(species_root.name),
                "species_dir": str(species_root),
                "data_dir": str(data_root),
                "results_json": str(results_path),
                "aggregation_method": "sum_of_per_video_best_threshold_metrics",
                "routes": sorted({str(v.get("route") or "") for v in video_records if str(v.get("route") or "")}),
                "aggregate": aggregate,
                "videos": video_records,
            }
            if not dry_run:
                results_path.parent.mkdir(parents=True, exist_ok=True)
                results_path.write_text(json.dumps(species_payload, indent=2), encoding="utf-8")

            method_registry[str(species_name)] = {
                "updated_at": now,
                "method_key": method_registry_key,
                "method_name": method_display_name,
                "species_name": str(species_name),
                "species_slug": str(species_root.name),
                "species_dir": str(species_root),
                "data_dir": str(data_root),
                "results_json": str(results_path),
                "routes": species_payload["routes"],
                "aggregate": aggregate,
                "videos": [
                    {
                        "video_key": str(v.get("video_key") or ""),
                        "video_name": str(v.get("video_name") or ""),
                        "video_path": str(v.get("video_path") or ""),
                        "route": str(v.get("route") or ""),
                        "out_root": str(v.get("out_root") or ""),
                        "rendered_video": str(v.get("rendered_video") or ""),
                        "status": _baseline_video_status(v.get("metrics")),
                        "best_metrics": dict(v.get("best_metrics") or {}),
                    }
                    for v in video_records
                ],
            }

        registry_methods[method_registry_key] = method_registry
        registry["methods"] = registry_methods

    registry["updated_at"] = now
    if not dry_run:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _version_num_from_name(name: str) -> Optional[int]:
    m = re.match(r"^v(?P<n>\d+)(?:_|$)", str(name))
    if not m:
        return None
    try:
        return int(m.group("n"))
    except Exception:
        return None


def _latest_version_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    best: tuple[int, str, Path] | None = None
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


def _parse_frame_like(value: object) -> Optional[int]:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return int(round(float(s)))
    except Exception:
        m = re.search(r"\d+", s)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
    return None


def _normalize_video_stem(name_or_path: object) -> str:
    s = str(name_or_path or "").strip()
    if not s:
        return ""
    s = Path(s).name
    if s.lower().endswith(".mp4"):
        s = s[: -len(".mp4")]
    return s


def _read_gt_rows_from_csv(csv_path: Path, *, video_stem: Optional[str]) -> List[Dict[str, int]]:
    if not csv_path.exists():
        return []
    out: List[Dict[str, int]] = []
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        if not fieldnames:
            return []
        cols = {str(c).strip().lower(): c for c in fieldnames}
        x_col = cols.get("x")
        y_col = cols.get("y")
        t_col = cols.get("t") or cols.get("frame") or cols.get("frame_idx")
        vn_col = cols.get("video_name")
        if not (x_col and y_col and t_col):
            return []

        want_stem = _normalize_video_stem(video_stem or "")
        for row in r:
            if vn_col and want_stem:
                row_stem = _normalize_video_stem(row.get(vn_col))
                if row_stem != want_stem:
                    continue
            try:
                x = int(round(float(row.get(x_col) or 0)))
                y = int(round(float(row.get(y_col) or 0)))
            except Exception:
                continue
            t = _parse_frame_like(row.get(t_col))
            if t is None:
                continue
            out.append({"x": x, "y": y, "t": int(t)})
    return out


def _load_gt_rows_for_video(*, video_path: Path, species_name: Optional[str]) -> tuple[List[Dict[str, int]], Optional[Path]]:
    video_stem = _normalize_video_stem(video_path.stem)

    # 1) Local per-video GT candidates near the video file.
    local_candidates = [
        video_path.with_suffix(".csv"),
        video_path.parent / f"{video_path.stem}_gt.csv",
        video_path.parent / "gt.csv",
    ]
    for cand in local_candidates:
        if not cand.exists():
            continue
        rows = _read_gt_rows_from_csv(cand, video_stem=video_stem)
        if rows:
            return rows, cand

    return [], None


def _write_gt_csv(path: Path, rows: Sequence[Dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "t"])
        w.writeheader()
        for r in rows:
            w.writerow({"x": int(r["x"]), "y": int(r["y"]), "t": int(r["t"])})


def _iter_image_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return
    for p in sorted(folder.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _iter_video_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*"), key=lambda x: str(x)):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def _reserve_unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    k = 1
    while True:
        cand = dst.with_name(f"{stem}__dup{k}{suffix}")
        if not cand.exists():
            return cand
        k += 1


def _link_or_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst = _reserve_unique_path(dst)
    if REQUIRE_HARDLINKS:
        try:
            os.link(src, dst)
            return dst
        except OSError as exc:
            raise RuntimeError(
                f"Hardlink failed for source '{src}' -> '{dst}'. "
                "REQUIRE_HARDLINKS=True blocks copy fallback."
            ) from exc
    shutil.copy2(src, dst)
    return dst


def _count_images(folder: Path) -> int:
    return sum(1 for _ in _iter_image_files(folder))


def _discover_training_sources_by_route() -> Dict[str, List[SourceSpec]]:
    single_root = PATCH_DATA_ROOT / "Integrated_prototype_datasets" / "single species datasets"
    if not single_root.exists():
        raise FileNotFoundError(single_root)

    out: Dict[str, List[SourceSpec]] = {"day": [], "night": []}
    for sp_dir in sorted([p for p in single_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        species_name = sp_dir.name
        latest = _latest_version_dir(sp_dir)
        if latest is None:
            continue
        final_dir = latest / "final dataset"
        if not final_dir.exists():
            continue

        # Skip empty or unusable species datasets.
        if _count_images(final_dir / "train" / "firefly") <= 0:
            continue

        route = _route_for_item(
            species_name=species_name,
            video_stem=None,
            text_for_hint=f"{species_name} {latest.name}",
        )
        out[route].append(SourceSpec(species_name=species_name, route=route, path=final_dir))

    return out


def _read_video_stems_from_patch_locations(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return []
    out: set[str] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        cols = {str(c).strip().lower(): c for c in reader.fieldnames}
        vn_col = cols.get("video_name")
        if not vn_col:
            return []
        for row in reader:
            stem = _normalize_video_stem(row.get(vn_col))
            if stem:
                out.add(stem)
    return sorted(out)


def _collect_training_video_stems_for_source(source: SourceSpec) -> Dict[str, Any]:
    version_dir = Path(source.path).resolve().parent
    csv_paths = sorted(version_dir.glob("patch_locations*.csv"))
    stems: set[str] = set()
    used_csvs: List[str] = []
    for csv_path in csv_paths:
        csv_stems = _read_video_stems_from_patch_locations(csv_path)
        if not csv_stems:
            continue
        stems.update(csv_stems)
        used_csvs.append(str(csv_path))
    return {
        "species_name": source.species_name,
        "route": source.route,
        "dataset_version_dir": str(version_dir),
        "patch_location_csvs": used_csvs,
        "video_stems": sorted(stems),
    }


def _build_training_inference_catalog(
    *,
    raw_root: Path,
    known_species: Sequence[str],
    sources_by_route: Dict[str, List[SourceSpec]],
) -> Dict[str, Any]:
    species_training_stems: Dict[str, set[str]] = {}
    source_records: List[Dict[str, Any]] = []
    for route_name in ("day", "night"):
        for source in list(sources_by_route.get(route_name) or []):
            rec = _collect_training_video_stems_for_source(source)
            source_records.append(rec)
            stems = species_training_stems.setdefault(source.species_name, set())
            stems.update(str(s) for s in rec.get("video_stems") or [])

    all_training_stems = {stem for stems in species_training_stems.values() for stem in stems}
    catalog_entries: List[Dict[str, str]] = []
    matched_training_stems: Dict[str, set[str]] = {sp: set() for sp in species_training_stems}

    for vp in _iter_video_files(raw_root):
        species_name = _infer_species_from_path(vp, known_species)
        route = _route_for_item(
            species_name=species_name,
            video_stem=vp.stem,
            text_for_hint=str(vp),
        )
        stem = vp.stem
        is_training = False
        if species_name and stem in species_training_stems.get(species_name, set()):
            is_training = True
            matched_training_stems.setdefault(species_name, set()).add(stem)
        elif stem in all_training_stems:
            is_training = True

        catalog_entries.append(
            {
                "video_path": str(vp),
                "video_name": str(vp.name),
                "video_stem": str(stem),
                "species_name": str(species_name or ""),
                "route": str(route),
                "category": ("training" if is_training else "inference"),
            }
        )

    all_species_in_catalog = {
        str(entry.get("species_name") or "").strip()
        for entry in catalog_entries
        if str(entry.get("species_name") or "").strip()
    }

    by_species: Dict[str, Dict[str, Any]] = {}
    for species_name in sorted(set(species_training_stems) | all_species_in_catalog):
        stems = species_training_stems.get(species_name, set())
        training_entries = [
            e for e in catalog_entries if e.get("species_name") == species_name and e.get("category") == "training"
        ]
        inference_entries = [
            e for e in catalog_entries if e.get("species_name") == species_name and e.get("category") == "inference"
        ]
        by_species[species_name] = {
            "training_video_stems_from_dataset": sorted(stems),
            "matched_training_video_paths": sorted(e["video_path"] for e in training_entries),
            "inference_video_paths": sorted(e["video_path"] for e in inference_entries),
            "n_training_videos": int(len(training_entries)),
            "n_inference_videos": int(len(inference_entries)),
            "unmatched_training_video_stems": sorted(stems - matched_training_stems.get(species_name, set())),
        }

    by_route: Dict[str, Dict[str, int]] = {
        "day": {"training": 0, "inference": 0},
        "night": {"training": 0, "inference": 0},
    }
    for entry in catalog_entries:
        route = str(entry.get("route") or "")
        category = str(entry.get("category") or "")
        if route in by_route and category in by_route[route]:
            by_route[route][category] += 1

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_videos_root": str(raw_root),
        "catalog_path": str(raw_root / VIDEO_CATALOG_FILENAME),
        "sources": source_records,
        "summary": {
            "n_total_videos": int(len(catalog_entries)),
            "n_training_videos": int(sum(1 for e in catalog_entries if e["category"] == "training")),
            "n_inference_videos": int(sum(1 for e in catalog_entries if e["category"] == "inference")),
            "by_route": by_route,
        },
        "by_species": by_species,
        "videos": sorted(catalog_entries, key=lambda d: (d["category"], d["route"], d["species_name"], d["video_name"])),
    }


def _write_training_inference_catalog(raw_root: Path, catalog: Dict[str, Any], *, dry_run: bool) -> Path:
    out_path = raw_root / VIDEO_CATALOG_FILENAME
    if not dry_run:
        out_path.write_text(json.dumps(catalog, indent=2))
    return out_path


def _enabled_inference_species_by_route(
    *,
    day_species_switches: Optional[Dict[str, bool]],
    night_species_switches: Optional[Dict[str, bool]],
) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {"day": set(), "night": set()}
    for route_name, switches in (
        ("day", day_species_switches),
        ("night", night_species_switches),
    ):
        if not switches:
            continue
        vals = {_slug(str(k)) for k, v in switches.items() if bool(v) and str(k or "").strip()}
        vals.discard("")
        out[route_name] = vals
    return out


def _enabled_species_from_switches(species_switches: Optional[Dict[str, bool]]) -> set[str]:
    if not species_switches:
        return set()
    vals = {_slug(str(k)) for k, v in species_switches.items() if bool(v) and str(k or "").strip()}
    vals.discard("")
    return vals


def _discover_inference_videos_from_catalog(
    catalog: Dict[str, Any],
    *,
    day_species_switches: Optional[Dict[str, bool]] = None,
    night_species_switches: Optional[Dict[str, bool]] = None,
) -> List[RoutedVideo]:
    allowed_by_route = _enabled_inference_species_by_route(
        day_species_switches=day_species_switches,
        night_species_switches=night_species_switches,
    )
    matched_by_route: Dict[str, set[str]] = {"day": set(), "night": set()}
    out: List[RoutedVideo] = []
    for entry in list(catalog.get("videos") or []):
        if str(entry.get("category") or "") != "inference":
            continue
        video_path = Path(str(entry.get("video_path") or "")).expanduser()
        if not video_path.exists():
            continue
        route = _normalize_route(str(entry.get("route") or ""))
        species_name = str(entry.get("species_name") or "").strip() or None
        species_slug = _slug(species_name or "")
        allowed = allowed_by_route.get(route, set())
        if species_slug not in allowed:
            continue
        matched_by_route.setdefault(route, set()).add(species_slug)
        out.append(RoutedVideo(video_path=video_path, route=route, species_name=species_name))

    for route_name in ("day", "night"):
        missing = sorted(allowed_by_route.get(route_name, set()) - matched_by_route.get(route_name, set()))
        if missing:
            print(
                f"[tmp-run] WARNING: enabled {route_name} inference species not found in catalog inference set: {missing}"
            )
    if not out:
        raise RuntimeError("No inference videos found in training/inference catalog.")
    return out


def _discover_inference_videos_from_catalog_by_species_switches(
    catalog: Dict[str, Any],
    *,
    species_switches: Optional[Dict[str, bool]] = None,
    label: str = "baseline",
) -> List[RoutedVideo]:
    allowed_species = _enabled_species_from_switches(species_switches)
    if not allowed_species:
        return []

    matched_species: set[str] = set()
    out: List[RoutedVideo] = []
    for entry in list(catalog.get("videos") or []):
        if str(entry.get("category") or "") != "inference":
            continue
        video_path = Path(str(entry.get("video_path") or "")).expanduser()
        if not video_path.exists():
            continue
        route = _normalize_route(str(entry.get("route") or ""))
        species_name = str(entry.get("species_name") or "").strip() or None
        species_slug = _slug(species_name or "")
        if species_slug not in allowed_species:
            continue
        matched_species.add(species_slug)
        out.append(RoutedVideo(video_path=video_path, route=route, species_name=species_name))

    missing = sorted(allowed_species - matched_species)
    if missing:
        print(f"[tmp-run] WARNING: enabled {label} species not found in catalog inference set: {missing}")
    return out


def _merge_routed_videos(*groups: Sequence[RoutedVideo]) -> List[RoutedVideo]:
    merged: Dict[tuple[str, str, str], RoutedVideo] = {}
    for group in groups:
        for rv in group:
            key = (
                str(rv.video_path.expanduser().resolve()),
                str(rv.route or ""),
                str(rv.species_name or ""),
            )
            merged.setdefault(key, rv)
    return list(merged.values())


def _build_combined_dataset(*, dst_final_dataset_dir: Path, sources: Sequence[SourceSpec], dry_run: bool) -> Dict[str, Any]:
    if not dry_run:
        for split in SPLITS:
            for cls in CLASSES:
                (dst_final_dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)

    copied_counts: Dict[str, int] = {}
    planned_counts: Dict[str, Dict[str, int]] = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}

    for src in sources:
        copied = 0
        for split in SPLITS:
            for cls in CLASSES:
                src_dir = src.path / split / cls
                dst_dir = dst_final_dataset_dir / split / cls
                for img in _iter_image_files(src_dir):
                    planned_counts[split][cls] += 1
                    if not dry_run:
                        _link_or_copy(img, dst_dir / f"{_slug(src.species_name)}__{img.name}")
                    copied += 1
        copied_counts[f"{src.species_name}@{src.route}"] = copied

    return {
        "sources": [
            {"species": s.species_name, "route": s.route, "path": str(s.path)}
            for s in sources
        ],
        "copied_per_source": copied_counts,
        "counts": planned_counts,
    }


def _assert_trainable(dataset_final_dir: Path) -> None:
    checks = [
        dataset_final_dir / "train" / "firefly",
        dataset_final_dir / "train" / "background",
        dataset_final_dir / "val" / "firefly",
        dataset_final_dir / "val" / "background",
        dataset_final_dir / "test" / "firefly",
        dataset_final_dir / "test" / "background",
    ]
    for p in checks:
        n = _count_images(p)
        if n <= 0:
            raise RuntimeError(f"Dataset check failed: no images in {p}")


def _day_yolo_species_slug(name: str) -> str:
    slug = _slug(name)
    if slug.startswith("day-"):
        slug = slug[len("day-") :]
    elif slug.startswith("night-"):
        slug = slug[len("night-") :]
    return DAY_YOLO_SPECIES_ALIASES.get(slug, slug)


def _enabled_day_yolo_species(species_switches: Optional[Dict[str, bool]]) -> Optional[set[str]]:
    if species_switches is None:
        return None
    out = {_day_yolo_species_slug(k) for k, v in species_switches.items() if bool(v) and str(k or "").strip()}
    out.discard("")
    return out


def _discover_day_yolo_sources(
    *,
    dataset_root: Path,
    species_switches: Optional[Dict[str, bool]],
) -> List[YoloSpeciesSource]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)

    enabled = _enabled_day_yolo_species(species_switches)
    discovered: List[YoloSpeciesSource] = []
    matched: set[str] = set()

    for sp_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        images_dir = sp_dir / "images"
        labels_dir = sp_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        species_name = _day_yolo_species_slug(sp_dir.name)
        if enabled is not None and species_name not in enabled:
            continue

        n_images = sum(1 for _ in _iter_image_files(images_dir))
        n_labels = sum(1 for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        if n_images <= 0 or n_labels <= 0:
            continue

        matched.add(species_name)
        discovered.append(
            YoloSpeciesSource(
                species_name=species_name,
                species_dir=sp_dir,
                images_dir=images_dir,
                labels_dir=labels_dir,
            )
        )

    if enabled is not None:
        missing = sorted(enabled - matched)
        if missing:
            print(f"[tmp-run][yolo] WARNING: enabled YOLO species not found in dataset root: {missing}")

    return discovered


def _write_day_yolo_data_yaml(dataset_root: Path, *, dry_run: bool) -> Path:
    data_yaml = dataset_root / "data.yaml"
    payload = (
        "path: .\n"
        "train: train/images\n"
        "val: train/images\n"
        "nc: 1\n"
        "names:\n"
        "  - streak\n"
    )
    if not dry_run:
        data_yaml.write_text(payload, encoding="utf-8")
    return data_yaml


def _build_combined_day_yolo_dataset(
    *,
    dataset_root: Path,
    sources: Sequence[YoloSpeciesSource],
    dry_run: bool,
) -> Dict[str, Any]:
    train_images_dir = dataset_root / "train" / "images"
    train_labels_dir = dataset_root / "train" / "labels"
    if not dry_run:
        train_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir.mkdir(parents=True, exist_ok=True)

    copied_per_species: Dict[str, int] = {}
    total_pairs = 0

    for src in sources:
        copied = 0
        for img in _iter_image_files(src.images_dir):
            lbl = src.labels_dir / f"{img.stem}.txt"
            if not lbl.exists():
                raise RuntimeError(f"Missing YOLO label for image: {img} (expected {lbl})")

            dest_stem = f"{_day_yolo_species_slug(src.species_name)}__{img.stem}"
            if not dry_run:
                _link_or_copy(img, train_images_dir / f"{dest_stem}{img.suffix.lower()}")
                _link_or_copy(lbl, train_labels_dir / f"{dest_stem}.txt")
            copied += 1
            total_pairs += 1

        copied_per_species[src.species_name] = copied

    data_yaml = _write_day_yolo_data_yaml(dataset_root, dry_run=dry_run)
    return {
        "dataset_root": str(dataset_root),
        "dataset_layout_version": "temporary_combined_train_only_v1",
        "source_dataset_root": str(DAY_YOLO_DATASET_ROOT),
        "data_yaml": str(data_yaml),
        "n_species": len(sources),
        "n_pairs": int(total_pairs),
        "species_counts": copied_per_species,
        "species_included": [s.species_name for s in sources],
        "source_species_dirs": {s.species_name: str(s.species_dir) for s in sources},
    }


def _select_yolo_device(user_choice: str | int | None):
    if isinstance(user_choice, int):
        return user_choice
    if isinstance(user_choice, str) and user_choice.lower() not in {"auto", "", "none"}:
        return user_choice

    try:
        import torch
    except Exception:
        return "cpu"

    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return 0
    return "cpu"


def _auto_imgsz_from_day_yolo_dataset(train_images_dir: Path) -> Optional[int]:
    try:
        import cv2
    except Exception:
        return None

    for p in train_images_dir.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        h, w = im.shape[:2]
        return int(max(h, w))
    return None


def _read_yolo_results_summary(run_dir: Path) -> Dict[str, Any]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"results_csv_found": True, "rows_logged": 0, "final_row": {}}
    return {
        "results_csv_found": True,
        "rows_logged": len(rows),
        "final_row": dict(rows[-1]),
    }


def _find_best_yolo_weight_after_train(*, model: Any, project_dir: Path, run_name: str) -> Path:
    best_attr = None
    try:
        best_attr = getattr(getattr(model, "trainer", None), "best", None)
    except Exception:
        best_attr = None
    if best_attr:
        p = Path(str(best_attr))
        if p.exists():
            return p

    save_dir = project_dir / run_name
    candidate = save_dir / "weights" / "best.pt"
    if candidate.exists():
        return candidate

    for p in project_dir.rglob("best.pt"):
        return p
    raise FileNotFoundError(f"Could not locate best.pt under {project_dir}")


def _cleanup_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _train_one_day_yolo_model(
    *,
    model_key: str,
    leaveout_species: Optional[str],
    sources: Sequence[YoloSpeciesSource],
    temp_dataset_root: Path,
    temp_project_root: Path,
    output_dir: Path,
    dry_run: bool,
) -> Tuple[YoloModelSpec, Dict[str, Any]]:
    dataset_summary = _build_combined_day_yolo_dataset(
        dataset_root=temp_dataset_root,
        sources=sources,
        dry_run=dry_run,
    )

    output_ckpt = output_dir / "best_firefly_yolo.pt"
    output_manifest = output_dir / "training_manifest.json"
    run_name = "train"

    if dry_run:
        print(
            f"[dry-run] Would train day YOLO model={model_key}: "
            f"species={dataset_summary.get('species_included')} -> {output_ckpt}"
        )
        manifest = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model_key": model_key,
            "model_scope": model_key,
            "model_bucket": ("global models" if leaveout_species is None else "leave one out models"),
            "leaveout_species": leaveout_species,
            "species_included": list(dataset_summary.get("species_included") or []),
            "dataset_summary": dataset_summary,
            "source_dataset_root": str(DAY_YOLO_DATASET_ROOT),
            "source_dataset_layout_version": "species_grouped_v1",
            "output_dir": str(output_dir),
            "best_model_export": str(output_ckpt),
            "model_weights_init": str(DAY_YOLO_MODEL_WEIGHTS_INIT),
            "train_args": {},
            "results_summary": {"error": "dry_run"},
            "temporary_artifacts_removed": False,
        }
        return (
            YoloModelSpec(
                model_key=model_key,
                ckpt_path=output_ckpt,
                output_dir=output_dir,
                leaveout_species=leaveout_species,
                species_included=tuple(str(x) for x in dataset_summary.get("species_included") or []),
            ),
            manifest,
        )

    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Ultralytics is required for YOLO training: {e}") from e

    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"torch is required for YOLO training: {e}") from e

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if temp_project_root.exists():
        shutil.rmtree(temp_project_root)
    temp_project_root.mkdir(parents=True, exist_ok=True)

    if str(_select_yolo_device(DAY_YOLO_DEVICE)).lower() == "mps" and bool(DAY_YOLO_MPS_CPU_FALLBACK):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    data_yaml = temp_dataset_root / "data.yaml"
    train_images_dir = temp_dataset_root / "train" / "images"
    imgsz = int(DAY_YOLO_IMG_SIZE) if DAY_YOLO_IMG_SIZE is not None else (_auto_imgsz_from_day_yolo_dataset(train_images_dir) or 640)
    device = _select_yolo_device(DAY_YOLO_DEVICE)

    train_args: Dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(DAY_YOLO_EPOCHS),
        "imgsz": int(imgsz),
        "batch": int(DAY_YOLO_BATCH_SIZE),
        "device": device,
        "workers": int(DAY_YOLO_WORKERS),
        "project": str(temp_project_root),
        "name": str(run_name),
        "patience": int(DAY_YOLO_PATIENCE),
    }
    if DAY_YOLO_LR0 is not None:
        train_args["lr0"] = float(DAY_YOLO_LR0)
    if DAY_YOLO_WEIGHT_DECAY is not None:
        train_args["weight_decay"] = float(DAY_YOLO_WEIGHT_DECAY)

    model = YOLO(str(DAY_YOLO_MODEL_WEIGHTS_INIT))
    print(f"[tmp-run][yolo-train] model={model_key} species={dataset_summary.get('species_included')}")
    for k, v in train_args.items():
        print(f"  - {k}: {v}")

    try:
        model.train(**train_args)
        best_pt = _find_best_yolo_weight_after_train(
            model=model,
            project_dir=temp_project_root,
            run_name=run_name,
        )
        shutil.copy2(best_pt, output_ckpt)
        results_summary = _read_yolo_results_summary(temp_project_root / run_name)
    finally:
        _cleanup_tree(temp_project_root)
        _cleanup_tree(temp_dataset_root)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_key": model_key,
        "model_scope": model_key,
        "model_bucket": ("global models" if leaveout_species is None else "leave one out models"),
        "leaveout_species": leaveout_species,
        "species_included": list(dataset_summary.get("species_included") or []),
        "dataset_summary": dataset_summary,
        "source_dataset_root": str(DAY_YOLO_DATASET_ROOT),
        "source_dataset_layout_version": "species_grouped_v1",
        "output_dir": str(output_dir),
        "best_model_export": str(output_ckpt),
        "model_weights_init": str(DAY_YOLO_MODEL_WEIGHTS_INIT),
        "train_args": train_args,
        "results_summary": results_summary,
        "temporary_artifacts_removed": True,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return (
        YoloModelSpec(
            model_key=model_key,
            ckpt_path=output_ckpt,
            output_dir=output_dir,
            leaveout_species=leaveout_species,
            species_included=tuple(str(x) for x in dataset_summary.get("species_included") or []),
        ),
        manifest,
    )


def _train_day_yolo_models(
    *,
    sources: Sequence[YoloSpeciesSource],
    run_root: Path,
    include_global: bool,
    include_leaveout: bool,
    dry_run: bool,
) -> Tuple[Dict[str, YoloModelSpec], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    models: Dict[str, YoloModelSpec] = {}
    dataset_summaries: Dict[str, Dict[str, Any]] = {}
    training_manifests: Dict[str, Dict[str, Any]] = {}

    if not sources:
        return models, dataset_summaries, training_manifests

    date_tag = datetime.now().strftime("%Y%m%d")
    all_species = sorted({s.species_name for s in sources})
    jobs: List[Tuple[str, Optional[str], List[YoloSpeciesSource], Path]] = []
    if include_global:
        jobs.append(
            (
                "global_all_species",
                None,
                list(sources),
                DAY_YOLO_GLOBAL_MODELS_ROOT / date_tag,
            )
        )
    if include_leaveout:
        for species in all_species:
            keep = [s for s in sources if s.species_name != species]
            if not keep:
                continue
            jobs.append(
                (
                    f"leaveout_{species}",
                    species,
                    keep,
                    DAY_YOLO_LEAVEOUT_MODELS_ROOT / date_tag / species,
                )
            )

    if not jobs:
        return models, dataset_summaries, training_manifests

    for model_key, leaveout_species, job_sources, output_dir in jobs:
        output_ckpt = output_dir / "best_firefly_yolo.pt"
        output_manifest = output_dir / "training_manifest.json"

        if bool(DAY_YOLO_REUSE_EXISTING_MODELS_IF_PRESENT) and output_ckpt.exists() and output_manifest.exists():
            manifest = _read_json_if_exists(output_manifest)
            training_manifests[model_key] = manifest
            dataset_summaries[model_key] = dict(manifest.get("dataset_summary") or {})
            models[model_key] = YoloModelSpec(
                model_key=model_key,
                ckpt_path=output_ckpt,
                output_dir=output_dir,
                leaveout_species=leaveout_species,
                species_included=tuple(str(x) for x in manifest.get("species_included") or []),
            )
            print(f"[tmp-run][yolo-train] reuse existing model: {output_ckpt}")
            continue

        temp_dataset_root = run_root / "_tmp_yolo_training_datasets" / model_key
        temp_project_root = run_root / "_tmp_yolo_training_runs" / model_key
        if temp_dataset_root.exists():
            shutil.rmtree(temp_dataset_root)
        temp_dataset_root.mkdir(parents=True, exist_ok=True)

        model_spec, manifest = _train_one_day_yolo_model(
            model_key=model_key,
            leaveout_species=leaveout_species,
            sources=job_sources,
            temp_dataset_root=temp_dataset_root,
            temp_project_root=temp_project_root,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        models[model_key] = model_spec
        training_manifests[model_key] = manifest
        dataset_summaries[model_key] = dict(manifest.get("dataset_summary") or {})

    if not dry_run:
        _cleanup_tree(run_root / "_tmp_yolo_training_datasets")
        _cleanup_tree(run_root / "_tmp_yolo_training_runs")

    return models, dataset_summaries, training_manifests


def _latest_dated_subdir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    dated = sorted(
        [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)],
        key=lambda p: p.name,
        reverse=True,
    )
    return dated[0] if dated else None


def _auto_discover_latest_day_yolo_model(*, dry_run: bool) -> Optional[Path]:
    if not DAY_YOLO_GLOBAL_MODELS_ROOT.exists():
        return None
    for model_dir in sorted(
        [p for p in DAY_YOLO_GLOBAL_MODELS_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)],
        key=lambda p: p.name,
        reverse=True,
    ):
        candidate = model_dir / "best_firefly_yolo.pt"
        if dry_run or candidate.exists():
            return candidate
    return None


def _auto_discover_latest_day_yolo_leaveout_models(
    *,
    required_species: Sequence[str],
    dry_run: bool,
) -> Dict[str, YoloModelSpec]:
    needed = sorted({_day_yolo_species_slug(s) for s in required_species if str(s or "").strip()})
    if not needed:
        return {}
    if not DAY_YOLO_LEAVEOUT_MODELS_ROOT.exists():
        return {}

    dated_dirs = sorted(
        [p for p in DAY_YOLO_LEAVEOUT_MODELS_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)],
        key=lambda p: p.name,
        reverse=True,
    )
    for run_dir in dated_dirs:
        out: Dict[str, YoloModelSpec] = {}
        missing: List[str] = []
        for species in needed:
            ckpt = run_dir / species / "best_firefly_yolo.pt"
            if not ckpt.exists():
                missing.append(species)
                continue
            out[species] = YoloModelSpec(
                model_key=f"leaveout_{species}",
                ckpt_path=ckpt,
                output_dir=ckpt.parent,
                leaveout_species=species,
                species_included=tuple(),
            )
        if not missing:
            return out
    return {}


def sanitize_dataset_images(
    data_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    exts: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    mode: str = "quarantine",
    quarantine_dir: str | Path | None = None,
    verify_with_pil: bool = True,
    report_max: int = 20,
) -> dict:
    """
    Best-effort sanitize to avoid DataLoader crashes due to corrupt image files.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Pillow (PIL) is required for dataset sanitization: {e}") from e

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    data_dir = Path(data_dir).expanduser().resolve()
    mode_norm = (mode or "").strip().lower()
    if mode_norm not in {"quarantine", "delete"}:
        raise ValueError(f"sanitize mode must be 'quarantine' or 'delete' (got {mode!r})")

    qroot: Path | None = None
    if mode_norm == "quarantine":
        qroot = Path(quarantine_dir).expanduser().resolve() if quarantine_dir else (data_dir / "__corrupt__")
        qroot.mkdir(parents=True, exist_ok=True)

    preview: list[dict] = []
    per_split: dict[str, dict[str, int]] = {}
    total_bad = 0

    for split in splits:
        split_root = data_dir / split
        if not split_root.exists():
            per_split[split] = {"scanned": 0, "bad": 0, "actioned": 0}
            continue

        total = 0
        for root, _, files in os.walk(split_root):
            for f in files:
                if Path(f).suffix.lower() in exts:
                    total += 1

        scanned = bad = actioned = 0
        pbar = (
            tqdm(total=total, desc=f"[sanitize] {split}", unit="img", dynamic_ncols=True)
            if tqdm is not None and total > 0
            else None
        )
        for root, _, files in os.walk(split_root):
            for f in files:
                if Path(f).suffix.lower() not in exts:
                    continue
                p = Path(root) / f
                scanned += 1

                reason: str | None = None
                try:
                    if p.stat().st_size == 0:
                        reason = "zero_bytes"
                    elif verify_with_pil:
                        with Image.open(p) as im:  # type: ignore[attr-defined]
                            im.verify()
                        with Image.open(p) as im:  # type: ignore[attr-defined]
                            im.load()
                except Exception as e:
                    reason = f"{type(e).__name__}: {e}"

                if reason is not None:
                    bad += 1
                    total_bad += 1
                    if len(preview) < int(report_max):
                        preview.append({"path": str(p), "split": split, "reason": reason})

                    try:
                        if mode_norm == "delete":
                            p.unlink(missing_ok=True)
                            actioned += 1
                        else:
                            assert qroot is not None
                            rel = p.relative_to(split_root)
                            dst = qroot / split / rel
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            if dst.exists():
                                dst = dst.with_name(f"{dst.stem}__dup{total_bad}{dst.suffix}")
                            shutil.move(str(p), str(dst))
                            actioned += 1
                    except Exception:
                        pass

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(bad=int(bad))

        if pbar is not None:
            pbar.close()

        per_split[split] = {"scanned": int(scanned), "bad": int(bad), "actioned": int(actioned)}

    if total_bad > 0:
        print(
            "[sanitize] Found corrupt images:",
            f"bad={total_bad} mode={mode_norm} verify_with_pil={bool(verify_with_pil)}"
            + (f" quarantine_dir={qroot}" if qroot is not None else ""),
        )
        for r in preview:
            print(f"  - {r['path']} ({r['split']} | {r['reason']})")
        if total_bad > len(preview):
            print(f"  ... and {total_bad - len(preview)} more")

    return {
        "enabled": True,
        "mode": mode_norm,
        "verify_with_pil": bool(verify_with_pil),
        "quarantine_dir": str(qroot) if qroot is not None else None,
        "per_split": per_split,
        "bad_total": int(total_bad),
        "bad_preview": preview,
    }


class RandomRotate90:
    def __call__(self, img):
        from torchvision.transforms import functional as F  # type: ignore

        return F.rotate(img, random.choice([0, 90, 180, 270]))


def build_resnet(name: str, num_classes: int = 2):
    import torch.nn as nn
    from torchvision import models  # type: ignore

    resnet_fns = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }
    if name not in resnet_fns:
        raise ValueError(f"Unknown ResNet model '{name}'. Choose from {list(resnet_fns)}")
    net = resnet_fns[name](weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    nn.init.xavier_uniform_(net.fc.weight)
    nn.init.zeros_(net.fc.bias)
    return net


def train_resnet_classifier(
    *,
    data_dir: str | Path,
    best_model_path: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    resnet_model: str,
    seed: int = 1337,
    no_gui: bool = True,
    metrics_out: str | Path | None = None,
    sanitize_dataset: bool = True,
    sanitize_mode: str = "quarantine",
    sanitize_quarantine_dir: str | Path | None = None,
    sanitize_verify_with_pil: bool = True,
    sanitize_report_max: int = 20,
) -> dict:
    from collections import Counter

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms  # type: ignore

    try:
        from tqdm import tqdm  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"tqdm is required for training progress bars: {e}") from e

    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    best_model_path = Path(best_model_path).expanduser().resolve()
    if not str(best_model_path):
        raise ValueError("best_model_path is required")

    epochs = int(epochs)
    batch_size = int(batch_size)
    lr = float(lr)
    num_workers = int(num_workers)
    resnet_model = str(resnet_model)
    seed = int(seed)

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    plt = None
    if not no_gui:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt  # type: ignore[no-redef]

    print(f"Dataset root: {data_dir}")

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    sanitizer_report: dict = {}
    if sanitize_dataset:
        sanitizer_report = sanitize_dataset_images(
            data_dir,
            mode=str(sanitize_mode),
            quarantine_dir=sanitize_quarantine_dir,
            verify_with_pil=bool(sanitize_verify_with_pil),
            report_max=int(sanitize_report_max),
        )

    train_tfm = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate90(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    )
    plain_tfm = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.ImageFolder(os.path.join(str(data_dir), "train"), train_tfm)
    val_ds = datasets.ImageFolder(os.path.join(str(data_dir), "val"), plain_tfm)
    test_ds = datasets.ImageFolder(os.path.join(str(data_dir), "test"), plain_tfm)

    if len(train_ds.classes) != 2:
        raise RuntimeError(f"Expected 2 classes under train/. Found: {train_ds.classes}")

    counts = Counter(train_ds.targets)
    n_classes = len(train_ds.classes)
    total = sum(counts.values())
    class_weights_list = []
    missing = []
    for c in range(n_classes):
        n_c = int(counts.get(c, 0))
        if n_c <= 0:
            class_weights_list.append(0.0)
            missing.append(train_ds.classes[c])
        else:
            class_weights_list.append(total / n_c)
    if missing:
        print(f"WARNING: Missing classes in train split (no samples): {missing}")
        print("Training will proceed, but metrics/models may be meaningless without negatives/positives.")
    class_weights = torch.tensor(class_weights_list, dtype=torch.float)
    sample_weights = [class_weights[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_dl = DataLoader(train_ds, batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = build_resnet(resnet_model).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_epoch = -1

    if plt is not None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        (ln_tr,) = ax.plot([], [], label="train loss")
        (ln_val,) = ax.plot([], [], label="val loss")
        (ln_acc,) = ax.plot([], [], label="val acc")
        ax.legend()

    def update_plot(ep, tr_loss, val_loss, val_acc, eta_txt):
        if plt is None:
            return
        for ln, y in zip((ln_tr, ln_val, ln_acc), (tr_loss, val_loss, val_acc)):
            ln.set_data(list(ln.get_xdata()) + [ep], list(ln.get_ydata()) + [y])
        ax.relim()
        ax.autoscale_view()
        fig.suptitle(f"Global ETA: {eta_txt}", fontsize=10)
        plt.pause(0.001)

    def run_epoch(loader, train=True, desc=""):
        model.train() if train else model.eval()
        tot_loss = tot_corr = tot_seen = 0
        for x, y in tqdm(loader, desc=desc, ncols=100, leave=False):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x)
                loss = criterion(out, y)
                if train:
                    loss.backward()
                    optimizer.step()
            tot_loss += loss.item() * x.size(0)
            tot_corr += (out.argmax(1) == y).sum().item()
            tot_seen += x.size(0)
        return tot_loss / tot_seen, tot_corr / tot_seen

    start = time.time()
    ep_times = []
    ep_pbar = tqdm(range(1, epochs + 1), desc="[stage2] epochs", unit="epoch", dynamic_ncols=True)
    for ep in ep_pbar:
        t0 = time.time()
        tr_loss, _ = run_epoch(train_dl, True, f"train {ep:02d}")
        val_loss, val_acc = run_epoch(val_dl, False, f"val   {ep:02d}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            torch.save(
                {"epoch": ep, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "val_acc": val_acc},
                str(best_model_path),
            )
            print(f"  saved new best (val_acc={val_acc:.2%}) -> {best_model_path}")

        ep_times.append(time.time() - t0)
        eta_txt = time.strftime("%H:%M:%S", time.gmtime(sum(ep_times) / len(ep_times) * (epochs - ep)))
        ep_pbar.set_postfix(val_acc=f"{val_acc:.2%}", best=f"{best_val_acc:.2%}", eta=str(eta_txt))
        print(
            f"Epoch {ep:02d} | train_loss {tr_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.2%} | "
            f"epoch {time.strftime('%H:%M:%S', time.gmtime(ep_times[-1]))} | ETA {eta_txt}"
        )
        update_plot(ep, tr_loss, val_loss, val_acc, eta_txt)

    ckpt = torch.load(str(best_model_path), map_location=device)
    best_model = build_resnet(resnet_model).to(device)
    best_model.load_state_dict(ckpt["model"])
    best_model.eval()
    test_loss = corr = tot_seen = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            out = best_model(x)
            test_loss += criterion(out, y).item() * x.size(0)
            corr += (out.argmax(1) == y).sum().item()
            tot_seen += x.size(0)
    test_loss = (test_loss / tot_seen) if tot_seen else float("nan")
    test_acc = (corr / tot_seen) if tot_seen else float("nan")
    print(f"\nBest checkpoint epoch {ckpt.get('epoch', best_epoch)} (val_acc={ckpt.get('val_acc', best_val_acc):.2%})")
    print(f"TEST | loss {test_loss:.4f} | acc {test_acc:.2%}")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "best_model_path": str(best_model_path),
        "resnet_model": resnet_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_workers": num_workers,
        "seed": seed,
        "best_epoch": int(ckpt.get("epoch", best_epoch)),
        "best_val_acc": float(ckpt.get("val_acc", best_val_acc)),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "sanitizer": sanitizer_report,
    }

    if metrics_out:
        outp = Path(metrics_out).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(metrics, indent=2))
        print(f"[metrics] Wrote: {outp}")

    if plt is not None:
        plt.ioff()
        plt.show()

    return metrics


def _train_model(
    *,
    route: str,
    dataset_final_dir: Path,
    model_path: Path,
    metrics_path: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    if dry_run:
        print(
            f"[dry-run] Would train {route} model:",
            f"data_dir={dataset_final_dir} -> {model_path} "
            f"(epochs={TRAIN_EPOCHS} batch={TRAIN_BATCH_SIZE} lr={TRAIN_LR} workers={TRAIN_NUM_WORKERS} "
            f"resnet={TRAIN_RESNET} seed={TRAIN_SEED})",
        )
        return {}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = train_resnet_classifier(
        data_dir=dataset_final_dir,
        best_model_path=model_path,
        epochs=int(TRAIN_EPOCHS),
        batch_size=int(TRAIN_BATCH_SIZE),
        lr=float(TRAIN_LR),
        num_workers=int(TRAIN_NUM_WORKERS),
        resnet_model=str(TRAIN_RESNET),
        seed=int(TRAIN_SEED),
        no_gui=True,
        metrics_out=str(metrics_path),
        sanitize_dataset=bool(SANITIZE_DATASET_IMAGES),
        sanitize_mode=str(SANITIZE_DATASET_MODE),
        sanitize_verify_with_pil=bool(SANITIZE_DATASET_VERIFY_WITH_PIL),
        sanitize_report_max=int(SANITIZE_DATASET_REPORT_MAX),
    )
    return metrics if isinstance(metrics, dict) else {}


def _discover_inference_videos(raw_root: Path, known_species: Sequence[str]) -> List[RoutedVideo]:
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    out: List[RoutedVideo] = []
    for vp in _iter_video_files(raw_root):
        species_name = _infer_species_from_path(vp, known_species)
        route = _route_for_item(
            species_name=species_name,
            video_stem=vp.stem,
            text_for_hint=str(vp),
        )
        out.append(RoutedVideo(video_path=vp, route=route, species_name=species_name))

    if not out:
        raise RuntimeError(f"No videos found under: {raw_root}")
    return out


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
                next(r, None)
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


def _accuracy(tp: int, fp: int, fn: int) -> float:
    denom = int(tp) + int(fp) + int(fn)
    return (float(tp) / float(denom)) if denom > 0 else 0.0


def _metrics_str(metrics: Dict[str, Any] | None) -> str:
    if not metrics or not isinstance(metrics, dict):
        return "ERROR: no metrics"
    if metrics.get("error"):
        return f"ERROR: {metrics.get('error')}"
    best = metrics.get("best")
    if not isinstance(best, dict):
        return "ERROR: missing best"
    try:
        tp = int(best.get("tp") or 0)
        fp = int(best.get("fp") or 0)
        fn = int(best.get("fn") or 0)
        prec = float(best.get("precision") or 0.0)
        f1 = float(best.get("f1") or 0.0)
    except Exception:
        return "ERROR: invalid best metrics"
    acc = _accuracy(tp, fp, fn)
    return f"tp={tp} fp={fp} fn={fn} precision={prec:.6f} accuracy={acc:.6f} f1={f1:.6f}"


def _run_subprocess_logged(cmd: Sequence[str], *, cwd: Path, log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] {log_path.name}: {_q(cmd)}")
        return
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(_q(cmd) + "\n\n")
        subprocess.run(list(cmd), cwd=str(cwd), check=True, stdout=f, stderr=subprocess.STDOUT, env=env)


def _run_baseline_processed_gt_matcher(
    *,
    route: str,
    orig_video_path: Path,
    pred_csv_path: Path,
    processed_gt_csv_path: Path,
    max_frames: int | None,
    out_dir: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    scripts = _baseline_scripts()
    matcher_script = scripts["match"]
    if not matcher_script.exists():
        raise FileNotFoundError(f"baseline processed-GT matcher script not found: {matcher_script}")
    cmd = [
        sys.executable,
        str(matcher_script),
        "--route",
        str(route),
        "--video",
        str(orig_video_path),
        "--pred-csv",
        str(pred_csv_path),
        "--processed-gt-csv",
        str(processed_gt_csv_path),
        "--out-dir",
        str(out_dir),
        "--dist-thresholds",
        *[str(float(x)) for x in BASELINE_DIST_THRESHOLDS_PX],
        "--crop-w",
        str(int(BASELINE_VALIDATE_CROP_W)),
        "--crop-h",
        str(int(BASELINE_VALIDATE_CROP_H)),
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(int(max_frames))])
    _run_subprocess_logged(cmd, cwd=_repo_root(), log_path=log_path, dry_run=dry_run)


def _run_baseline_renderer(
    *,
    video_path: Path,
    pred_csv_path: Path,
    out_video_path: Path,
    label: str,
    max_frames: int | None,
    log_path: Path,
    scripts: Dict[str, Path],
    dry_run: bool,
) -> None:
    render_script = scripts.get("render")
    if render_script is None or not render_script.exists():
        raise FileNotFoundError(f"baseline renderer script not found: {render_script}")
    _run_subprocess_logged(
        [
            sys.executable,
            str(render_script),
            "--video",
            str(video_path),
            "--pred-csv",
            str(pred_csv_path),
            "--out-video",
            str(out_video_path),
            "--label",
            str(label),
        ]
        + (["--max-frames", str(int(max_frames))] if max_frames is not None else []),
        cwd=_repo_root(),
        log_path=log_path,
        dry_run=dry_run,
    )


def _video_max_frames_from_gt(video: EvalVideo) -> int | None:
    if video.gt_max_t is None:
        return None
    return int(video.gt_max_t) + 1


def _run_baselines_for_video(
    *,
    run_root: Path,
    video: EvalVideo,
    processed_gt_csv_path: Path,
    logs_dir: Path,
    run_lab: bool,
    run_raphael: bool,
    dry_run: bool,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    scripts = _baseline_scripts()
    max_frames_for_video = _video_max_frames_from_gt(video)

    # Lab baseline
    if run_lab:
        out_root = _baseline_species_data_root("lab", video.species_name) / video.video_key
        log_root = _baseline_species_log_root("lab", video.species_name)
        gt_csv = out_root / "ground truth" / "gt.csv"
        pred_csv = out_root / "predictions.csv"
        rendered_video = out_root / BASELINE_RENDERED_VIDEO_FILENAME
        validation_dir = out_root / "validation" / video.video_name
        if not dry_run:
            out_root.mkdir(parents=True, exist_ok=True)
            log_root.mkdir(parents=True, exist_ok=True)
            gt_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(processed_gt_csv_path, gt_csv)

        if validation_dir.exists():
            metrics = _parse_validation_metrics(validation_dir)
        else:
            if not scripts["lab"].exists():
                metrics = {"error": f"lab baseline script not found: {scripts['lab']}"}
            else:
                try:
                    _run_subprocess_logged(
                        [
                            sys.executable,
                            str(scripts["lab"]),
                            "--video",
                            str(video.video_path),
                            "--out-csv",
                            str(pred_csv),
                            "--threshold",
                            str(float(LAB_BASELINE_THRESHOLD)),
                            "--blur-sigma",
                            str(float(LAB_BASELINE_BLUR_SIGMA)),
                            "--bkgr-window-sec",
                            str(float(LAB_BASELINE_BKGR_WINDOW_SEC)),
                            "--box-w",
                            str(int(BASELINE_VALIDATE_CROP_W)),
                            "--box-h",
                            str(int(BASELINE_VALIDATE_CROP_H)),
                        ]
                        + (["--max-frames", str(int(max_frames_for_video))] if max_frames_for_video is not None else []),
                        cwd=_repo_root(),
                        log_path=log_root / f"baseline_lab__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    _run_baseline_processed_gt_matcher(
                        route=video.route,
                        orig_video_path=video.video_path,
                        pred_csv_path=pred_csv,
                        processed_gt_csv_path=gt_csv,
                        max_frames=max_frames_for_video,
                        out_dir=validation_dir,
                        log_path=log_root / f"baseline_lab_match__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    metrics = _parse_validation_metrics(validation_dir)
                except Exception as e:
                    metrics = {"error": f"lab baseline failed: {e}"}

        if pred_csv.exists() and not rendered_video.exists():
            try:
                _run_baseline_renderer(
                    video_path=video.video_path,
                    pred_csv_path=pred_csv,
                    out_video_path=rendered_video,
                    label="Lab baseline",
                    max_frames=max_frames_for_video,
                    log_path=log_root / f"baseline_lab_render__{video.video_key}.log",
                    scripts=scripts,
                    dry_run=dry_run,
                )
            except Exception as e:
                print(f"[tmp-run] WARNING: lab baseline renderer failed for {video.video_key}: {e}")

        out["lab"] = {
            "metrics": metrics,
            "out_root": str(out_root),
            "rendered_video": str(rendered_video if (dry_run or rendered_video.exists()) else ""),
        }

    # Raphael baseline
    if run_raphael:
        out_root = _baseline_species_data_root("raphael", video.species_name) / video.video_key
        log_root = _baseline_species_log_root("raphael", video.species_name)
        gt_csv = out_root / "ground truth" / "gt.csv"
        pred_csv = out_root / "predictions.csv"
        raw_csv = out_root / "raw.csv"
        gauss_csv = out_root / "gauss.csv"
        rendered_video = out_root / BASELINE_RENDERED_VIDEO_FILENAME
        validation_dir = out_root / "validation" / video.video_name
        if not dry_run:
            out_root.mkdir(parents=True, exist_ok=True)
            log_root.mkdir(parents=True, exist_ok=True)
            gt_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(processed_gt_csv_path, gt_csv)

        if validation_dir.exists():
            metrics = _parse_validation_metrics(validation_dir)
        else:
            model_path_str = str(RAPHAEL_MODEL_PATH or "").strip()
            model_ok = bool(model_path_str) and Path(model_path_str).expanduser().is_file()
            if not model_ok:
                metrics = {"error": "raphael model not found; set RAPHAEL_MODEL_PATH in this script"}
            elif not scripts["raphael"].exists():
                metrics = {"error": f"raphael baseline script not found: {scripts['raphael']}"}
            else:
                try:
                    _run_subprocess_logged(
                        [
                            sys.executable,
                            str(scripts["raphael"]),
                            "--video",
                            str(video.video_path),
                            "--model",
                            str(Path(model_path_str).expanduser().resolve()),
                            "--out-csv",
                            str(pred_csv),
                            "--raw-csv",
                            str(raw_csv),
                            "--gauss-csv",
                            str(gauss_csv),
                            "--bw-thr",
                            str(float(RAPHAEL_BW_THR)),
                            "--classify-thr",
                            str(float(RAPHAEL_CLASSIFY_THR)),
                            "--bkgr-window-sec",
                            str(float(RAPHAEL_BKGR_WINDOW_SEC)),
                            "--blur-sigma",
                            str(float(RAPHAEL_BLUR_SIGMA)),
                            "--patch-size",
                            str(int(RAPHAEL_PATCH_SIZE_PX)),
                            "--batch-size",
                            str(int(RAPHAEL_BATCH_SIZE)),
                            "--gauss-crop-size",
                            str(int(RAPHAEL_GAUSS_CROP_SIZE)),
                            "--device",
                            str(RAPHAEL_DEVICE),
                        ]
                        + (["--max-frames", str(int(max_frames_for_video))] if max_frames_for_video is not None else []),
                        cwd=_repo_root(),
                        log_path=log_root / f"baseline_raphael__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    _run_baseline_processed_gt_matcher(
                        route=video.route,
                        orig_video_path=video.video_path,
                        pred_csv_path=pred_csv,
                        processed_gt_csv_path=gt_csv,
                        max_frames=max_frames_for_video,
                        out_dir=validation_dir,
                        log_path=log_root / f"baseline_raphael_match__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    metrics = _parse_validation_metrics(validation_dir)
                except Exception as e:
                    metrics = {"error": f"raphael baseline failed: {e}"}

        if pred_csv.exists() and not rendered_video.exists():
            try:
                _run_baseline_renderer(
                    video_path=video.video_path,
                    pred_csv_path=pred_csv,
                    out_video_path=rendered_video,
                    label="Raphael baseline",
                    max_frames=max_frames_for_video,
                    log_path=log_root / f"baseline_raphael_render__{video.video_key}.log",
                    scripts=scripts,
                    dry_run=dry_run,
                )
            except Exception as e:
                print(f"[tmp-run] WARNING: Raphael baseline renderer failed for {video.video_key}: {e}")

        out["raphael"] = {
            "metrics": metrics,
            "out_root": str(out_root),
            "rendered_video": str(rendered_video if (dry_run or rendered_video.exists()) else ""),
        }

    return out


def _init_final_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_RESULTS_FIELDNAMES)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())


def _append_final_csv_row(path: Path, row: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_RESULTS_FIELDNAMES)
        w.writerow({k: (row.get(k) or "") for k in FINAL_RESULTS_FIELDNAMES})
        f.flush()
        os.fsync(f.fileno())


def _write_final_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_RESULTS_FIELDNAMES)
        w.writeheader()
        for row in rows:
            w.writerow({k: (row.get(k) or "") for k in FINAL_RESULTS_FIELDNAMES})
        f.flush()
        os.fsync(f.fileno())


def _count_csv_data_rows(path: Path) -> int:
    with path.open("r", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        return sum(1 for _ in r)


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _train_models_for_route(
    *,
    route: str,
    sources: Sequence[SourceSpec],
    combined_dataset_root: Path,
    model_root: Path,
    include_global: bool,
    include_leaveout: bool,
    dry_run: bool,
) -> tuple[Dict[str, ModelSpec], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    models: Dict[str, ModelSpec] = {}
    dataset_summaries: Dict[str, Dict[str, Any]] = {}
    training_metrics: Dict[str, Dict[str, Any]] = {}

    if not sources:
        return models, dataset_summaries, training_metrics

    all_species = sorted({s.species_name for s in sources})
    jobs: List[Tuple[str, Optional[str], List[SourceSpec]]] = []
    if include_global:
        jobs.append(("global_all_species", None, list(sources)))

    if include_leaveout:
        for species in all_species:
            keep = [s for s in sources if s.species_name != species]
            if not keep:
                continue
            jobs.append((f"leaveout_{species}", species, keep))

    if not jobs:
        return models, dataset_summaries, training_metrics

    for model_key, leaveout_species, job_sources in jobs:
        dst_final = combined_dataset_root / route / model_key / "final dataset"
        if not dry_run:
            dst_final.mkdir(parents=True, exist_ok=True)

        summary = _build_combined_dataset(dst_final_dataset_dir=dst_final, sources=job_sources, dry_run=dry_run)
        dataset_summaries[model_key] = summary

        print(f"[tmp-run][train] route={route} model={model_key}")
        for split in SPLITS:
            ff = int(summary["counts"][split]["firefly"])
            bg = int(summary["counts"][split]["background"])
            print(f"  - {split}: firefly={ff}, background={bg}")

        model_path = model_root / route / f"{model_key}.pt"
        metrics_path = model_root / route / f"{model_key}_training_metrics.json"

        if bool(REUSE_EXISTING_MODELS_IF_PRESENT) and model_path.exists() and (not dry_run):
            print(f"[tmp-run][train] reuse existing model: {model_path}")
            metrics = _read_json_if_exists(metrics_path)
        else:
            if not dry_run:
                _assert_trainable(dst_final)
            metrics = _train_model(
                route=route,
                dataset_final_dir=dst_final,
                model_path=model_path,
                metrics_path=metrics_path,
                dry_run=dry_run,
            )

        training_metrics[model_key] = metrics
        models[model_key] = ModelSpec(
            route=route,
            model_key=model_key,
            ckpt_path=model_path,
            leaveout_species=leaveout_species,
        )

    return models, dataset_summaries, training_metrics


def _expected_model_keys_for_sources(
    sources: Sequence[SourceSpec],
    *,
    include_global: bool = True,
    include_leaveout: bool = True,
) -> List[str]:
    all_species = sorted({s.species_name for s in sources})
    keys: List[str] = []
    if include_global:
        keys.append("global_all_species")
    if include_leaveout:
        for species in all_species:
            # Match _train_models_for_route: leaveout model exists only if at least
            # one other species remains in the route-specific training pool.
            if any(s.species_name != species for s in sources):
                keys.append(f"leaveout_{species}")
    return keys


def _route_model_dir_has_required_models(
    *,
    route_model_dir: Path,
    sources: Sequence[SourceSpec],
    dry_run: bool,
    include_global: bool = True,
    include_leaveout: bool = True,
) -> bool:
    for mk in _expected_model_keys_for_sources(
        sources,
        include_global=include_global,
        include_leaveout=include_leaveout,
    ):
        ckpt = route_model_dir / f"{mk}.pt"
        if not ckpt.exists():
            return False
    return True


def _auto_discover_latest_route_model_dir(
    *,
    route: str,
    runs_root: Path,
    current_run_root: Path,
    sources: Sequence[SourceSpec],
    dry_run: bool,
    include_global: bool = True,
    include_leaveout: bool = True,
) -> Optional[Path]:
    candidates: List[Path] = []
    for run_dir in sorted(
        [p for p in runs_root.glob(f"{RUN_NAME_PREFIX}__*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if run_dir.resolve() == current_run_root.resolve():
            continue
        route_model_dir = run_dir / "models" / route
        if not route_model_dir.exists():
            continue
        candidates.append(route_model_dir)

    for route_model_dir in candidates:
        if _route_model_dir_has_required_models(
            route_model_dir=route_model_dir,
            sources=sources,
            dry_run=dry_run,
            include_global=include_global,
            include_leaveout=include_leaveout,
        ):
            return route_model_dir
    return None


def _load_models_for_route(
    *,
    route: str,
    sources: Sequence[SourceSpec],
    route_model_dir: Path,
    dry_run: bool,
    include_global: bool = True,
    include_leaveout: bool = True,
) -> Dict[str, ModelSpec]:
    models: Dict[str, ModelSpec] = {}
    if not sources:
        return models

    missing: List[Path] = []
    for model_key in _expected_model_keys_for_sources(
        sources,
        include_global=include_global,
        include_leaveout=include_leaveout,
    ):
        ckpt = route_model_dir / f"{model_key}.pt"
        if not ckpt.exists():
            missing.append(ckpt)
            continue
        leaveout_species = model_key[len("leaveout_") :] if model_key.startswith("leaveout_") else None
        models[model_key] = ModelSpec(
            route=route,
            model_key=model_key,
            ckpt_path=ckpt,
            leaveout_species=leaveout_species,
        )

    if missing:
        missing_txt = "\n".join(f"  - {p}" for p in missing)
        raise SystemExit(
            "RUN_MODEL_TRAINING=False but required model files are missing:\n"
            f"{missing_txt}\n"
            f"Either enable RUN_MODEL_TRAINING=True or point the manual {route} model root "
            "to a complete route-specific model directory."
        )

    return models


def _resolve_manual_path(path_value: Path, *, label: str, dry_run: bool) -> Path:
    resolved = Path(str(path_value)).expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_route_model_dir(
    *,
    route: str,
    auto_discover: bool,
    manual_root: Path,
    runs_root: Path,
    current_run_root: Path,
    sources: Sequence[SourceSpec],
    dry_run: bool,
    include_global: bool,
    include_leaveout: bool,
) -> tuple[Path, str]:
    if auto_discover:
        resolved = _auto_discover_latest_route_model_dir(
            route=route,
            runs_root=runs_root,
            current_run_root=current_run_root,
            sources=sources,
            dry_run=dry_run,
            include_global=include_global,
            include_leaveout=include_leaveout,
        )
        if resolved is None:
            raise SystemExit(
                f"RUN_MODEL_TRAINING=False and auto-discovery could not find a compatible latest "
                f"{route} model directory under {runs_root}.\n"
                f"Turn AUTO_DISCOVER_LATEST_{route.upper()}_MODEL_ROOT off and set the manual "
                f"{route} model root, or enable RUN_MODEL_TRAINING=True."
            )
        return resolved, "auto"

    return _resolve_manual_path(
        manual_root,
        label=f"manual {route} model root",
        dry_run=dry_run,
    ), "manual"


def _resolve_day_yolo_model_path(*, auto_discover: bool, manual_path: Path, dry_run: bool) -> tuple[Path, str]:
    if auto_discover:
        resolved = _auto_discover_latest_day_yolo_model(dry_run=dry_run)
        if resolved is None:
            raise SystemExit(
                "Auto-discovery could not find a latest day YOLO checkpoint.\n"
                "Turn AUTO_DISCOVER_LATEST_DAY_YOLO_MODEL off and set DAY_YOLO_MODEL_WEIGHTS manually."
            )
        return resolved, "auto"

    return _resolve_manual_path(
        manual_path,
        label="manual day YOLO model",
        dry_run=dry_run,
    ), "manual"


def _has_thr_subdirs(dir_path: Path) -> bool:
    if not dir_path.exists() or not dir_path.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("thr_") for p in dir_path.iterdir())


def _day_analysis_output_requirements() -> Tuple[bool, bool]:
    params_path = _day_pipeline_dir() / "params.py"
    try:
        spec = importlib.util.spec_from_file_location("_tmp_day_v3_params_requirements", params_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load spec for {params_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        need_stage7 = bool(getattr(module, "RUN_STAGE7_FN_ANALYSIS", True))
        need_stage8 = bool(getattr(module, "RUN_STAGE8_FP_ANALYSIS", True))
        return need_stage7, need_stage8
    except Exception:
        return True, True


def _missing_day_analysis_outputs(day_root: Path, video_name: str) -> List[str]:
    need_stage7, need_stage8 = _day_analysis_output_requirements()
    missing: List[str] = []
    if need_stage7:
        stage7 = day_root / "stage7 fn analysis" / video_name
        if not _has_thr_subdirs(stage7):
            missing.append("stage7")
    if need_stage8:
        stage8 = day_root / "stage8 fp analysis" / video_name
        if not _has_thr_subdirs(stage8):
            missing.append("stage8")
    return missing


def _day_analysis_outputs_ready(day_root: Path, video_name: str) -> bool:
    return not _missing_day_analysis_outputs(day_root, video_name)


def _format_day_analysis_missing_error(day_root: Path, video_name: str) -> str:
    missing = _missing_day_analysis_outputs(day_root, video_name)
    missing_str = "/".join(missing) if missing else "stage7/stage8"
    return (
        f"day validation exists but analysis outputs are incomplete for {video_name}: "
        f"missing {missing_str} under {day_root}"
    )


def _run_gateway_for_video(
    *,
    out_root: Path,
    video: EvalVideo,
    model_path: Path,
    day_yolo_model_path: Optional[Path],
    log_path: Path,
    dry_run: bool,
) -> Tuple[str, Dict[str, Any], Optional[Path]]:
    """
    Returns: (route_used, validation_metrics)
    """
    day_root = out_root / "day_pipeline_v3"
    night_root = out_root / "night_time_pipeline"
    max_frames_for_video = _video_max_frames_from_gt(video)
    stage_day = day_root / "stage5 validation" / video.video_name
    stage_night = night_root / "stage9 validation" / video.video_name

    def _processed_gt_from_output(route_name: str) -> Optional[Path]:
        route_root = day_root if route_name == "day" else night_root
        stage_root = stage_day if route_name == "day" else stage_night
        candidates: List[Path] = []
        csv_dir = route_root / "csv files"
        if csv_dir.exists():
            candidates.extend(sorted(csv_dir.glob("gt_norm_offset*.csv"), key=lambda p: p.stat().st_mtime, reverse=True))
        if stage_root.exists():
            candidates.extend(sorted(stage_root.glob("gt_norm_offset*.csv"), key=lambda p: p.stat().st_mtime, reverse=True))
        seen: set[str] = set()
        unique: List[Path] = []
        for cand in candidates:
            key = str(cand.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(cand)
        return unique[0] if unique else None

    # Resume-friendly cache: reuse existing valid outputs.
    if stage_day.exists():
        m = _parse_validation_metrics(stage_day)
        if not m.get("error") and _day_analysis_outputs_ready(day_root, video.video_name):
            return "day", m, _processed_gt_from_output("day")
        if not m.get("error"):
            print(
                f"[tmp-run][eval] cache incomplete for day video={video.video_name}: "
                f"{_format_day_analysis_missing_error(day_root, video.video_name)}; rerunning gateway."
            )
    if stage_night.exists():
        m = _parse_validation_metrics(stage_night)
        if not m.get("error"):
            return "night", m, _processed_gt_from_output("night")

    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        _write_gt_csv(day_root / "ground truth" / f"gt_{video.video_name}.csv", video.gt_rows)
        _write_gt_csv(day_root / "ground truth" / "gt.csv", video.gt_rows)
        _write_gt_csv(day_root / "gt.csv", video.gt_rows)
        _write_gt_csv(night_root / "ground truth" / "gt.csv", video.gt_rows)
        _write_gt_csv(night_root / "gt.csv", video.gt_rows)

    cmd = [
        sys.executable,
        str(_gateway_py()),
        "--input",
        str(video.video_path),
        "--output-root",
        str(out_root),
        "--route-override",
        str(video.route),
        "--max-concurrent",
        str(int(GATEWAY_MAX_CONCURRENT)),
        "--day-patch-model",
        str(model_path),
        "--night-cnn-model",
        str(model_path),
        *(["--day-yolo-model", str(day_yolo_model_path)] if day_yolo_model_path is not None else []),
        *(["--max-frames", str(int(max_frames_for_video))] if max_frames_for_video is not None else []),
        "--threshold",
        str(float(GATEWAY_BRIGHTNESS_THRESHOLD)),
        "--frames",
        str(int(GATEWAY_BRIGHTNESS_FRAMES)),
    ]
    if bool(FORCE_GATEWAY_TESTS):
        cmd.append("--force-tests")

    _run_subprocess_logged(cmd, cwd=_repo_root(), log_path=log_path, dry_run=dry_run)

    if stage_day.exists():
        m = _parse_validation_metrics(stage_day)
        if (not m.get("error")) and _day_analysis_outputs_ready(day_root, video.video_name):
            return "day", m, _processed_gt_from_output("day")
        if not m.get("error"):
            return "day", {"error": _format_day_analysis_missing_error(day_root, video.video_name)}, _processed_gt_from_output("day")
        return "day", m, _processed_gt_from_output("day")
    if stage_night.exists():
        return "night", _parse_validation_metrics(stage_night), _processed_gt_from_output("night")

    expected = stage_day if video.route == "day" else stage_night
    return video.route, {"error": f"no validation outputs found (expected under: {expected})"}, None


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Temporary runner: route-split global+leaveout training + baselines + gateway inference "
            "with baselines matched against pipeline-processed GT."
        )
    )
    p.add_argument(
        "--raw-videos-root",
        type=str,
        default=str(RAW_VIDEOS_ROOT),
        help="Root folder of videos to infer on (scanned recursively).",
    )
    p.add_argument(
        "--runs-root",
        type=str,
        default=str(RUNS_ROOT),
        help="Where to write run outputs.",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="If >0, process only first N discovered videos (for smoke testing).",
    )
    p.add_argument("--dry-run", action="store_true", default=False, help="Plan only; do not train/infer.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dry_run = bool(DRY_RUN) or bool(args.dry_run)

    _ensure_repo_import_path()

    if str(TRAIN_RESNET).strip().lower() != "resnet18":
        raise SystemExit(
            f"TRAIN_RESNET={TRAIN_RESNET!r} is incompatible with current inference loaders "
            "(expect resnet18 checkpoints). Set TRAIN_RESNET='resnet18'."
        )

    raw_videos_root = Path(str(args.raw_videos_root)).expanduser().resolve()
    runs_root = Path(str(args.runs_root)).expanduser().resolve()
    if not runs_root.exists() and not dry_run:
        runs_root.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.now().strftime("%Y%m%d__%H%M%S")
    run_id = f"{RUN_NAME_PREFIX}__{run_tag}"
    run_root = runs_root / run_id

    combined_dataset_root = run_root / "combined_training_datasets"
    model_root = run_root / "models"
    inference_root = run_root / "inference_outputs"
    logs_dir = run_root / "logs"
    final_csv_path = run_root / FINAL_RESULTS_FILENAME
    metadata_path = run_root / "run_metadata.json"

    print(f"[tmp-run] run_id: {run_id}")
    print(f"[tmp-run] run_root: {run_root}")
    print(f"[tmp-run] raw_videos_root: {raw_videos_root}")
    print(f"[tmp-run] final_csv: {final_csv_path}")
    print(f"[tmp-run] dry_run={dry_run}")
    run_day_yolo_training = bool(RUN_DAY_YOLO_MODEL_TRAINING)
    run_baselines = bool(RUN_BASELINE_METHODS_INFERENCE)
    run_lab_baseline = bool(run_baselines and RUN_LAB_BASELINE)
    run_raphael_baseline = bool(run_baselines and RUN_RAPHAEL_BASELINE)
    if run_baselines and (not run_lab_baseline) and (not run_raphael_baseline):
        print("[tmp-run] WARNING: baseline methods enabled but both Lab and Raphael toggles are off; disabling baselines.")
        run_baselines = False

    enabled_pipeline_routes = [
        r
        for r, enabled in (("day", bool(RUN_DAY_PIPELINE_INFERENCE)), ("night", bool(RUN_NIGHT_PIPELINE_INFERENCE)))
        if enabled
    ]
    run_global_model_inference = bool(RUN_GLOBAL_MODEL_INFERENCE)
    run_leaveout_model_inference = bool(RUN_LEAVEOUT_MODEL_INFERENCE)
    run_any_pipeline_models = bool(run_global_model_inference or run_leaveout_model_inference)
    if enabled_pipeline_routes and (not run_any_pipeline_models):
        print(
            "[tmp-run] WARNING: pipeline routes are enabled but both "
            "RUN_GLOBAL_MODEL_INFERENCE and RUN_LEAVEOUT_MODEL_INFERENCE are False; "
            "disabling pipeline inference."
        )
    run_any_pipeline = bool(enabled_pipeline_routes) and run_any_pipeline_models

    print(
        f"[tmp-run] baselines: run={run_baselines} lab={run_lab_baseline} "
        f"raphael={run_raphael_baseline}"
    )
    print(
        f"[tmp-run] pipeline inference: day={bool(RUN_DAY_PIPELINE_INFERENCE)} "
        f"night={bool(RUN_NIGHT_PIPELINE_INFERENCE)} "
        f"global={run_global_model_inference} leaveout={run_leaveout_model_inference}"
    )
    print(
        f"[tmp-run] model training: run={bool(RUN_MODEL_TRAINING)} "
        f"day={bool(RUN_DAY_MODEL_TRAINING)} night={bool(RUN_NIGHT_MODEL_TRAINING)} "
        f"global={bool(TRAIN_GLOBAL_MODELS)} leaveout={bool(TRAIN_LEAVEOUT_MODELS)}"
    )
    print(
        f"[tmp-run] day yolo training: run={run_day_yolo_training} "
        f"global={bool(TRAIN_DAY_YOLO_GLOBAL_MODEL)} "
        f"leaveout={bool(TRAIN_DAY_YOLO_LEAVEOUT_MODELS)}"
    )
    print(f"[tmp-run] day_inference_species_switches={DAY_INFERENCE_SPECIES_SWITCHES}")
    print(f"[tmp-run] night_inference_species_switches={NIGHT_INFERENCE_SPECIES_SWITCHES}")
    print(f"[tmp-run] lab_baseline_species_switches={LAB_BASELINE_SPECIES_SWITCHES}")
    print(f"[tmp-run] raphael_baseline_species_switches={RAPHAEL_BASELINE_SPECIES_SWITCHES}")
    print(f"[tmp-run] day_yolo_training_species_switches={DAY_YOLO_TRAINING_SPECIES_SWITCHES}")
    print(f"[tmp-run] run_model_training={bool(RUN_MODEL_TRAINING)}")
    if (not run_baselines) and (not run_any_pipeline) and (not bool(RUN_MODEL_TRAINING)) and (not run_day_yolo_training):
        raise SystemExit(
            "Nothing to run: RUN_MODEL_TRAINING=False, RUN_DAY_YOLO_MODEL_TRAINING=False, RUN_BASELINE_METHODS_INFERENCE=False, "
            "RUN_DAY_PIPELINE_INFERENCE=False, RUN_NIGHT_PIPELINE_INFERENCE=False, "
            "or both pipeline model-scope toggles are False."
        )

    sources_by_route = _discover_training_sources_by_route()
    print("[tmp-run] discovered training sources:")
    for route in ("day", "night"):
        srcs = sources_by_route.get(route) or []
        print(f"  - {route}: {len(srcs)} species")
        for s in srcs:
            print(f"      {s.species_name} -> {s.path}")

    day_yolo_sources = _discover_day_yolo_sources(
        dataset_root=DAY_YOLO_DATASET_ROOT,
        species_switches=DAY_YOLO_TRAINING_SPECIES_SWITCHES,
    )
    print(f"[tmp-run] discovered day yolo sources: {len(day_yolo_sources)} species")
    for s in day_yolo_sources:
        print(f"  - {s.species_name}: {s.species_dir}")

    models_by_route: Dict[str, Dict[str, ModelSpec]] = {"day": {}, "night": {}}
    dataset_summaries: Dict[str, Dict[str, Dict[str, Any]]] = {"day": {}, "night": {}}
    training_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {"day": {}, "night": {}}
    day_yolo_models: Dict[str, YoloModelSpec] = {}
    day_yolo_dataset_summaries: Dict[str, Dict[str, Any]] = {}
    day_yolo_training_manifests: Dict[str, Dict[str, Any]] = {}
    resolved_pretrained_model_dirs: Dict[str, str] = {"day": "", "night": ""}
    resolved_pretrained_model_sources: Dict[str, str] = {"day": "", "night": ""}
    resolved_day_yolo_model: Optional[Path] = None
    resolved_day_yolo_source: str = ""
    resolved_day_yolo_leaveout_models: Dict[str, YoloModelSpec] = {}
    resolved_day_yolo_leaveout_source: str = ""

    if bool(RUN_MODEL_TRAINING):
        enabled_training_routes = [
            r
            for r, enabled in (("day", bool(RUN_DAY_MODEL_TRAINING)), ("night", bool(RUN_NIGHT_MODEL_TRAINING)))
            if enabled
        ]
        if not enabled_training_routes:
            print("[tmp-run] WARNING: RUN_MODEL_TRAINING=True but both RUN_DAY_MODEL_TRAINING and RUN_NIGHT_MODEL_TRAINING are False; skipping model training.")
        if not bool(TRAIN_GLOBAL_MODELS) and not bool(TRAIN_LEAVEOUT_MODELS):
            print("[tmp-run] WARNING: RUN_MODEL_TRAINING=True but both TRAIN_GLOBAL_MODELS and TRAIN_LEAVEOUT_MODELS are False; skipping model training.")
        for route in ("day", "night"):
            if route not in enabled_training_routes:
                print(f"[tmp-run] model training disabled for route={route}; skipping.")
                continue
            srcs = sources_by_route.get(route) or []
            if not srcs:
                print(f"[tmp-run] WARNING: no training sources for route={route}; skipping model training.")
                continue

            route_models, route_ds, route_metrics = _train_models_for_route(
                route=route,
                sources=srcs,
                combined_dataset_root=combined_dataset_root,
                model_root=model_root,
                include_global=bool(TRAIN_GLOBAL_MODELS),
                include_leaveout=bool(TRAIN_LEAVEOUT_MODELS),
                dry_run=dry_run,
            )
            models_by_route[route] = route_models
            dataset_summaries[route] = route_ds
            training_metrics[route] = route_metrics
            resolved_pretrained_model_dirs[route] = str(model_root / route)
            resolved_pretrained_model_sources[route] = "trained_this_run"
    elif run_any_pipeline:
        for route in enabled_pipeline_routes:
            srcs = sources_by_route.get(route) or []
            if not srcs:
                continue
            if route == "day":
                route_model_dir, source_kind = _resolve_route_model_dir(
                    route=route,
                    auto_discover=bool(AUTO_DISCOVER_LATEST_DAY_MODEL_ROOT),
                    manual_root=DAY_PRETRAINED_MODEL_ROOT,
                    runs_root=runs_root,
                    current_run_root=run_root,
                    sources=srcs,
                    dry_run=dry_run,
                    include_global=run_global_model_inference,
                    include_leaveout=run_leaveout_model_inference,
                )
            else:
                route_model_dir, source_kind = _resolve_route_model_dir(
                    route=route,
                    auto_discover=bool(AUTO_DISCOVER_LATEST_NIGHT_MODEL_ROOT),
                    manual_root=NIGHT_PRETRAINED_MODEL_ROOT,
                    runs_root=runs_root,
                    current_run_root=run_root,
                    sources=srcs,
                    dry_run=dry_run,
                    include_global=run_global_model_inference,
                    include_leaveout=run_leaveout_model_inference,
                )
            resolved_pretrained_model_dirs[route] = str(route_model_dir)
            resolved_pretrained_model_sources[route] = source_kind
            print(f"[tmp-run] {route} model root ({source_kind}): {route_model_dir}")
            models_by_route[route] = _load_models_for_route(
                route=route,
                sources=srcs,
                route_model_dir=route_model_dir,
                dry_run=dry_run,
                include_global=run_global_model_inference,
                include_leaveout=run_leaveout_model_inference,
            )
    else:
        print("[tmp-run] model prep skipped (training disabled and both day/night pipeline inference toggles are off).")

    if run_day_yolo_training:
        if not bool(TRAIN_DAY_YOLO_GLOBAL_MODEL) and not bool(TRAIN_DAY_YOLO_LEAVEOUT_MODELS):
            print("[tmp-run] WARNING: RUN_DAY_YOLO_MODEL_TRAINING=True but both TRAIN_DAY_YOLO_GLOBAL_MODEL and TRAIN_DAY_YOLO_LEAVEOUT_MODELS are False; skipping YOLO training.")
        elif not day_yolo_sources:
            print("[tmp-run] WARNING: no selected day YOLO species folders found; skipping YOLO training.")
        else:
            day_yolo_models, day_yolo_dataset_summaries, day_yolo_training_manifests = _train_day_yolo_models(
                sources=day_yolo_sources,
                run_root=run_root,
                include_global=bool(TRAIN_DAY_YOLO_GLOBAL_MODEL),
                include_leaveout=bool(TRAIN_DAY_YOLO_LEAVEOUT_MODELS),
                dry_run=dry_run,
            )
            if "global_all_species" in day_yolo_models:
                resolved_day_yolo_model = day_yolo_models["global_all_species"].ckpt_path
                resolved_day_yolo_source = "trained_this_run"
            resolved_day_yolo_leaveout_models = {
                m.leaveout_species or "": m
                for m in day_yolo_models.values()
                if m.leaveout_species
            }
            if resolved_day_yolo_leaveout_models:
                resolved_day_yolo_leaveout_source = "trained_this_run"

    if bool(RUN_DAY_PIPELINE_INFERENCE) and resolved_day_yolo_model is None:
        resolved_day_yolo_model, resolved_day_yolo_source = _resolve_day_yolo_model_path(
            auto_discover=bool(AUTO_DISCOVER_LATEST_DAY_YOLO_MODEL),
            manual_path=DAY_YOLO_MODEL_WEIGHTS,
            dry_run=dry_run,
        )
        print(f"[tmp-run] day yolo model ({resolved_day_yolo_source}): {resolved_day_yolo_model}")

    known_species = sorted(
        set(list(ROUTE_BY_SPECIES.keys()) + [s.species_name for vv in sources_by_route.values() for s in vv])
    )
    video_catalog = _build_training_inference_catalog(
        raw_root=raw_videos_root,
        known_species=known_species,
        sources_by_route=sources_by_route,
    )
    video_catalog_path = _write_training_inference_catalog(raw_videos_root, video_catalog, dry_run=dry_run)
    catalog_summary = dict(video_catalog.get("summary") or {})
    print(f"[tmp-run] video catalog: {video_catalog_path}")
    print(
        "[tmp-run] video split catalog:"
        f" training={catalog_summary.get('n_training_videos', 0)}"
        f" inference={catalog_summary.get('n_inference_videos', 0)}"
    )
    for route_name in ("day", "night"):
        route_counts = dict((catalog_summary.get("by_route") or {}).get(route_name) or {})
        print(
            f"  - {route_name}:"
            f" training={int(route_counts.get('training', 0))}"
            f" inference={int(route_counts.get('inference', 0))}"
        )

    pipeline_routed_videos: List[RoutedVideo] = []
    if run_any_pipeline:
        pipeline_routed_videos = _discover_inference_videos_from_catalog(
            video_catalog,
            day_species_switches=DAY_INFERENCE_SPECIES_SWITCHES,
            night_species_switches=NIGHT_INFERENCE_SPECIES_SWITCHES,
        )

    lab_baseline_routed_videos: List[RoutedVideo] = []
    raphael_baseline_routed_videos: List[RoutedVideo] = []
    if run_baselines:
        if run_lab_baseline:
            lab_baseline_routed_videos = _discover_inference_videos_from_catalog_by_species_switches(
                video_catalog,
                species_switches=LAB_BASELINE_SPECIES_SWITCHES,
                label="lab baseline",
            )
        if run_raphael_baseline:
            raphael_baseline_routed_videos = _discover_inference_videos_from_catalog_by_species_switches(
                video_catalog,
                species_switches=RAPHAEL_BASELINE_SPECIES_SWITCHES,
                label="raphael baseline",
            )
        if (not lab_baseline_routed_videos) and (not raphael_baseline_routed_videos):
            print("[tmp-run] WARNING: baselines enabled but no baseline species were selected from the inference catalog.")

    baseline_routed_videos = _merge_routed_videos(lab_baseline_routed_videos, raphael_baseline_routed_videos)
    routed_videos = _merge_routed_videos(pipeline_routed_videos, baseline_routed_videos)

    max_videos = int(args.max_videos or 0)
    if max_videos > 0:
        routed_videos = routed_videos[:max_videos]

    print(f"[tmp-run] pipeline videos selected: {len(pipeline_routed_videos)}")
    print(f"[tmp-run] lab baseline videos selected: {len(lab_baseline_routed_videos)}")
    print(f"[tmp-run] raphael baseline videos selected: {len(raphael_baseline_routed_videos)}")
    print(f"[tmp-run] total baseline videos selected: {len(baseline_routed_videos)}")
    print(f"[tmp-run] unique videos discovered: {len(routed_videos)}")
    by_route_counts = {"day": 0, "night": 0}
    for v in routed_videos:
        by_route_counts[v.route] = by_route_counts.get(v.route, 0) + 1
    print(f"[tmp-run] unique video split: day={by_route_counts.get('day', 0)} night={by_route_counts.get('night', 0)}")
    print(f"[tmp-run] require_gt_for_inference={bool(REQUIRE_GT_FOR_INFERENCE)}")
    if run_baselines and (not run_any_pipeline) and (not baseline_routed_videos):
        raise SystemExit("Baselines enabled, but no baseline species were selected from the inference catalog.")

    eval_videos: List[EvalVideo] = []
    skipped_videos: List[Dict[str, str]] = []

    for rv in routed_videos:
        gt_rows, gt_src = _load_gt_rows_for_video(video_path=rv.video_path, species_name=rv.species_name)
        if not gt_rows and bool(REQUIRE_GT_FOR_INFERENCE):
            skipped_videos.append(
                {
                    "video_path": str(rv.video_path),
                    "route": str(rv.route),
                    "species_name": str(rv.species_name or ""),
                    "error": "missing_gt_csv_or_annotations",
                }
            )
            continue

        species = str(rv.species_name or "unknown_species")
        video_name = rv.video_path.stem
        gt_max_t = max((int(r["t"]) for r in gt_rows), default=None)

        eval_videos.append(
            EvalVideo(
                species_name=species,
                video_name=video_name,
                video_path=rv.video_path,
                video_key=_short_key(species, video_name),
                route=rv.route,
                gt_rows=list(gt_rows),
                gt_source=gt_src,
                gt_max_t=(int(gt_max_t) if gt_max_t is not None else None),
            )
        )

    print(f"[tmp-run] videos with GT: {len(eval_videos)}")
    if skipped_videos:
        print(f"[tmp-run] skipped (missing GT): {len(skipped_videos)}")

    if not eval_videos:
        raise SystemExit("No videos with GT available for evaluation.")

    eval_videos = sorted(eval_videos, key=lambda v: (v.route, v.species_name, v.video_name))
    baseline_eval_video_keys: set[str] = {
        _short_key(str(rv.species_name or "unknown_species"), rv.video_path.stem)
        for rv in baseline_routed_videos
    }
    lab_baseline_video_keys: set[str] = {
        _short_key(str(rv.species_name or "unknown_species"), rv.video_path.stem)
        for rv in lab_baseline_routed_videos
    }
    raphael_baseline_video_keys: set[str] = {
        _short_key(str(rv.species_name or "unknown_species"), rv.video_path.stem)
        for rv in raphael_baseline_routed_videos
    }
    baseline_eval_videos = [
        vid
        for vid in eval_videos
        if vid.video_key in baseline_eval_video_keys
    ]

    rows_out: List[Dict[str, str]] = []
    if not dry_run:
        _init_final_csv(final_csv_path)

    def _record_row(row: Dict[str, str]) -> None:
        rows_out.append(row)
        if not dry_run:
            _append_final_csv_row(final_csv_path, row)

    # Record GT-missing skips immediately.
    for s in skipped_videos:
        _record_row(
            {
                "run_id": run_id,
                "route": str(s.get("route") or ""),
                "species_name": str(s.get("species_name") or ""),
                "video_name": Path(str(s.get("video_path") or "")).stem,
                "eval_type": "video_skip",
                "model_used": "",
                "results": f"ERROR: {s.get('error')}",
                "inference_output_path": "",
                "lab_results": "",
                "lab_output_path": "",
                "raphael_results": "",
                "raphael_output_path": "",
                "gt_source": "",
                "gt_rows": "0",
                "gt_max_t": "",
            }
        )

    # Baselines are run once per video and reused for all model rows.
    baseline_by_video_key: Dict[str, Dict[str, Dict[str, Any]]] = {}
    def _baseline_fields(video_key: str) -> Tuple[str, str, str, str]:
        base = baseline_by_video_key.get(video_key) or {}
        raphael = base.get("raphael") or {}
        lab = base.get("lab") or {}
        raphael_metrics = _metrics_str(raphael.get("metrics")) if raphael else ""
        raphael_path = str(raphael.get("out_root") or "") if raphael else ""
        lab_metrics = _metrics_str(lab.get("metrics")) if lab else ""
        lab_path = str(lab.get("out_root") or "") if lab else ""
        return lab_metrics, lab_path, raphael_metrics, raphael_path

    if run_baselines:
        print("[baselines] deferred until after pipeline inference so baselines can reuse pipeline-processed GT.")

    videos_by_route: Dict[str, List[EvalVideo]] = {"day": [], "night": []}
    for v in eval_videos:
        videos_by_route.setdefault(v.route, []).append(v)

    pipeline_eval_video_keys: set[str] = {
        _short_key(str(rv.species_name or "unknown_species"), rv.video_path.stem)
        for rv in pipeline_routed_videos
    }
    if run_baselines:
        missing_pipeline_for_baselines = [
            vid
            for vid in baseline_eval_videos
            if vid.video_key not in pipeline_eval_video_keys
        ]
        if missing_pipeline_for_baselines:
            missing_desc = ", ".join(
                sorted({f"{vid.route}:{vid.species_name}:{vid.video_name}" for vid in missing_pipeline_for_baselines})
            )
            raise SystemExit(
                "Baselines now reuse pipeline-processed GT, so every baseline-selected video must also be "
                f"selected for pipeline inference in the same run. Missing pipeline coverage for: {missing_desc}"
            )

    processed_gt_by_video_key: Dict[str, Dict[str, str]] = {}

    if bool(RUN_DAY_PIPELINE_INFERENCE) and bool(run_leaveout_model_inference) and not resolved_day_yolo_leaveout_models:
        required_leaveout_species = sorted({v.species_name for v in videos_by_route.get("day") or []})
        resolved_day_yolo_leaveout_models = _auto_discover_latest_day_yolo_leaveout_models(
            required_species=required_leaveout_species,
            dry_run=dry_run,
        )
        if resolved_day_yolo_leaveout_models:
            resolved_day_yolo_leaveout_source = "auto"
            print(
                f"[tmp-run] day yolo leaveout models ({resolved_day_yolo_leaveout_source}): "
                f"{sorted(resolved_day_yolo_leaveout_models.keys())}"
            )
        elif required_leaveout_species:
            print(
                "[tmp-run] WARNING: leaveout day YOLO models were not found; "
                "day leaveout inference will fall back to the global YOLO model."
            )

    # Global model on all videos in each route.
    if run_global_model_inference:
        for route_name in ("day", "night"):
            if route_name not in enabled_pipeline_routes:
                print(f"[eval][global] route={route_name} disabled by run toggle; skipping route.")
                continue
            route_videos = videos_by_route.get(route_name) or []
            if not route_videos:
                continue

            route_models = models_by_route.get(route_name) or {}
            global_model = route_models.get("global_all_species")
            if global_model is None:
                print(f"[eval][global] WARNING: missing global model for route={route_name}; skipping route.")
                for vid in route_videos:
                    lab_metrics, lab_path, raphael_metrics, raphael_path = _baseline_fields(vid.video_key)
                    _record_row(
                        {
                            "run_id": run_id,
                            "route": route_name,
                            "species_name": vid.species_name,
                            "video_name": vid.video_name,
                            "eval_type": "global",
                            "model_used": "global_all_species",
                            "results": f"ERROR: no trained model for route={route_name}",
                            "inference_output_path": "",
                            "lab_results": lab_metrics,
                            "lab_output_path": lab_path,
                            "raphael_results": raphael_metrics,
                            "raphael_output_path": raphael_path,
                            "gt_source": str(vid.gt_source or ""),
                            "gt_rows": str(int(len(vid.gt_rows))),
                            "gt_max_t": str(vid.gt_max_t if vid.gt_max_t is not None else ""),
                        }
                    )
                continue

            print(f"[eval] global model route={route_name}: {global_model.ckpt_path}")
            for i, vid in enumerate(route_videos, start=1):
                out_root = inference_root / "pipelines" / f"{route_name}_videos" / global_model.model_key / vid.video_key
                log_path = logs_dir / f"gateway__{route_name}__{global_model.model_key}__{vid.video_key}.log"
                print(f"[eval][global][{route_name}] {i}/{len(route_videos)} {vid.species_name} :: {vid.video_path.name}")

                if dry_run:
                    metrics = {"error": "dry_run"}
                else:
                    try:
                        route_used, metrics, processed_gt_csv = _run_gateway_for_video(
                            out_root=out_root,
                            video=vid,
                            model_path=global_model.ckpt_path,
                            day_yolo_model_path=(resolved_day_yolo_model if route_name == "day" else None),
                            log_path=log_path,
                            dry_run=bool(dry_run),
                        )
                        if processed_gt_csv is not None and vid.video_key not in processed_gt_by_video_key:
                            processed_gt_by_video_key[vid.video_key] = {
                                "route": str(route_used),
                                "processed_gt_csv": str(processed_gt_csv),
                                "pipeline_output_root": str(out_root),
                                "processed_gt_rows": str(_count_csv_data_rows(processed_gt_csv)),
                            }
                    except Exception as e:
                        metrics = {"error": str(e)}

                lab_metrics, lab_path, raphael_metrics, raphael_path = _baseline_fields(vid.video_key)
                _record_row(
                    {
                        "run_id": run_id,
                        "route": route_name,
                        "species_name": vid.species_name,
                        "video_name": vid.video_name,
                        "eval_type": "global",
                        "model_used": global_model.model_key,
                        "results": _metrics_str(metrics),
                        "inference_output_path": str(out_root),
                        "lab_results": lab_metrics,
                        "lab_output_path": lab_path,
                        "raphael_results": raphael_metrics,
                        "raphael_output_path": raphael_path,
                        "gt_source": str(vid.gt_source or ""),
                        "gt_rows": str(int(len(vid.gt_rows))),
                        "gt_max_t": str(vid.gt_max_t if vid.gt_max_t is not None else ""),
                    }
                )
    else:
        print("[eval][global] disabled by RUN_GLOBAL_MODEL_INFERENCE=False; skipping global-model inference.")

    # Leaveout models on their left-out species videos (within same route only).
    if run_leaveout_model_inference:
        for route_name in ("day", "night"):
            if route_name not in enabled_pipeline_routes:
                print(f"[eval][leaveout] route={route_name} disabled by run toggle; skipping route.")
                continue
            route_videos = videos_by_route.get(route_name) or []
            if not route_videos:
                continue

            route_models = models_by_route.get(route_name) or {}
            leaveout_models = [m for m in route_models.values() if str(m.model_key).startswith("leaveout_")]
            leaveout_models.sort(key=lambda m: m.model_key)

            for lm in leaveout_models:
                left_species = str(lm.leaveout_species or "")
                sel = [v for v in route_videos if v.species_name == left_species]
                if not sel:
                    print(f"[eval][leaveout] WARNING: no route={route_name} videos for species={left_species} ({lm.model_key})")
                    continue

                print(f"[eval] leaveout model {lm.model_key} route={route_name} species={left_species} (n_videos={len(sel)})")
                for i, vid in enumerate(sel, start=1):
                    out_root = inference_root / "pipelines" / f"{route_name}_videos" / lm.model_key / vid.video_key
                    log_path = logs_dir / f"gateway__{route_name}__{lm.model_key}__{vid.video_key}.log"
                    print(f"[eval][{lm.model_key}][{route_name}] {i}/{len(sel)} {vid.video_path.name}")
                    leaveout_day_yolo_path: Optional[Path] = None
                    if route_name == "day":
                        leaveout_day_yolo_path = (
                            resolved_day_yolo_leaveout_models.get(left_species).ckpt_path
                            if left_species in resolved_day_yolo_leaveout_models
                            else resolved_day_yolo_model
                        )

                    if dry_run:
                        metrics = {"error": "dry_run"}
                    else:
                        try:
                            route_used, metrics, processed_gt_csv = _run_gateway_for_video(
                                out_root=out_root,
                                video=vid,
                                model_path=lm.ckpt_path,
                                day_yolo_model_path=leaveout_day_yolo_path,
                                log_path=log_path,
                                dry_run=bool(dry_run),
                            )
                            if processed_gt_csv is not None and vid.video_key not in processed_gt_by_video_key:
                                processed_gt_by_video_key[vid.video_key] = {
                                    "route": str(route_used),
                                    "processed_gt_csv": str(processed_gt_csv),
                                    "pipeline_output_root": str(out_root),
                                    "processed_gt_rows": str(_count_csv_data_rows(processed_gt_csv)),
                                }
                        except Exception as e:
                            metrics = {"error": str(e)}

                    lab_metrics, lab_path, raphael_metrics, raphael_path = _baseline_fields(vid.video_key)
                    _record_row(
                        {
                            "run_id": run_id,
                            "route": route_name,
                            "species_name": vid.species_name,
                            "video_name": vid.video_name,
                            "eval_type": "leaveout",
                            "model_used": lm.model_key,
                            "results": _metrics_str(metrics),
                            "inference_output_path": str(out_root),
                            "lab_results": lab_metrics,
                            "lab_output_path": lab_path,
                            "raphael_results": raphael_metrics,
                            "raphael_output_path": raphael_path,
                            "gt_source": str(vid.gt_source or ""),
                            "gt_rows": str(int(len(vid.gt_rows))),
                            "gt_max_t": str(vid.gt_max_t if vid.gt_max_t is not None else ""),
                        }
                    )
    else:
        print("[eval][leaveout] disabled by RUN_LEAVEOUT_MODEL_INFERENCE=False; skipping leaveout-model inference.")

    if run_baselines:
        print(f"[baselines] enabled={run_baselines}")
        if not baseline_eval_videos:
            print("[baselines] WARNING: no baseline-selected videos with GT were found; skipping baseline execution.")
        lab_baseline_species = sorted({vid.species_name for vid in baseline_eval_videos if vid.video_key in lab_baseline_video_keys})
        raphael_baseline_species = sorted({vid.species_name for vid in baseline_eval_videos if vid.video_key in raphael_baseline_video_keys})
        if run_lab_baseline and lab_baseline_species:
            _prepare_baseline_species_storage(
                method_key="lab",
                species_names=lab_baseline_species,
                dry_run=dry_run,
            )
        if run_raphael_baseline and raphael_baseline_species:
            _prepare_baseline_species_storage(
                method_key="raphael",
                species_names=raphael_baseline_species,
                dry_run=dry_run,
            )
        for i, vid in enumerate(baseline_eval_videos, start=1):
            print(f"[baselines] {i}/{len(baseline_eval_videos)} {vid.species_name} :: {vid.video_path.name} (route={vid.route})")
            run_lab_for_video = vid.video_key in lab_baseline_video_keys
            run_raphael_for_video = vid.video_key in raphael_baseline_video_keys
            processed_gt_info = processed_gt_by_video_key.get(vid.video_key) or {}
            processed_gt_csv_str = str(processed_gt_info.get("processed_gt_csv") or "")
            if not processed_gt_csv_str:
                baseline_by_video_key[vid.video_key] = {
                    mk: {"metrics": {"error": "processed GT from pipeline inference was not found for this video"}, "out_root": "", "rendered_video": ""}
                    for mk, enabled in (("lab", run_lab_for_video), ("raphael", run_raphael_for_video))
                    if enabled
                }
            elif not dry_run:
                baseline_by_video_key[vid.video_key] = _run_baselines_for_video(
                    run_root=run_root,
                    video=vid,
                    processed_gt_csv_path=Path(processed_gt_csv_str),
                    logs_dir=logs_dir,
                    run_lab=run_lab_for_video,
                    run_raphael=run_raphael_for_video,
                    dry_run=bool(dry_run),
                )
            lab_metrics, lab_path, raphael_metrics, raphael_path = _baseline_fields(vid.video_key)
            baseline_methods_used = []
            if run_lab_for_video:
                baseline_methods_used.append("lab")
            if run_raphael_for_video:
                baseline_methods_used.append("raphael")
            _record_row(
                {
                    "run_id": run_id,
                    "route": vid.route,
                    "species_name": vid.species_name,
                    "video_name": vid.video_name,
                    "eval_type": "baseline",
                    "model_used": "+".join(baseline_methods_used),
                    "results": "",
                    "inference_output_path": str(processed_gt_info.get("pipeline_output_root") or ""),
                    "lab_results": lab_metrics,
                    "lab_output_path": lab_path,
                    "raphael_results": raphael_metrics,
                    "raphael_output_path": raphael_path,
                    "gt_source": str(processed_gt_csv_str or vid.gt_source or ""),
                    "gt_rows": str(processed_gt_info.get("processed_gt_rows") or int(len(vid.gt_rows))),
                    "gt_max_t": str(vid.gt_max_t if vid.gt_max_t is not None else ""),
                }
            )
        if not dry_run:
            _write_baseline_species_results_and_registry(
                eval_videos=baseline_eval_videos,
                baseline_by_video_key=baseline_by_video_key,
                dry_run=dry_run,
            )

    if run_baselines:
        for row in rows_out:
            if str(row.get("eval_type") or "") not in {"global", "leaveout"}:
                continue
            video_key = _short_key(str(row.get("species_name") or "unknown_species"), str(row.get("video_name") or ""))
            processed_gt_info = processed_gt_by_video_key.get(video_key) or {}
            lab_metrics, lab_path, raphael_metrics, raphael_path = _baseline_fields(video_key)
            row["lab_results"] = lab_metrics
            row["lab_output_path"] = lab_path
            row["raphael_results"] = raphael_metrics
            row["raphael_output_path"] = raphael_path
            if processed_gt_info:
                row["gt_source"] = str(processed_gt_info.get("processed_gt_csv") or row.get("gt_source") or "")
                row["gt_rows"] = str(processed_gt_info.get("processed_gt_rows") or row.get("gt_rows") or "")

    if not dry_run:
        _write_final_csv(final_csv_path, rows_out)

    record: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "dry_run": dry_run,
        "paths": {
            "run_root": str(run_root),
            "raw_videos_root": str(raw_videos_root),
            "video_catalog": str(video_catalog_path),
            "combined_dataset_root": str(combined_dataset_root),
            "model_root": str(model_root),
            "inference_root": str(inference_root),
            "final_csv": str(final_csv_path),
        },
        "routing": {
            "route_by_species": dict(ROUTE_BY_SPECIES),
            "route_by_video_stem": dict(ROUTE_BY_VIDEO_STEM),
            "route_day_hint_tokens": list(ROUTE_NAME_HINT_DAY_TOKENS),
            "route_night_hint_tokens": list(ROUTE_NAME_HINT_NIGHT_TOKENS),
            "route_default": str(ROUTE_DEFAULT),
            "require_explicit_route": bool(REQUIRE_EXPLICIT_ROUTE),
            "require_gt_for_inference": bool(REQUIRE_GT_FOR_INFERENCE),
        },
        "baselines": {
            "enabled": bool(run_baselines),
            "lab_enabled": bool(run_lab_baseline),
            "raphael_enabled": bool(run_raphael_baseline),
            "lab_species_switches": dict(LAB_BASELINE_SPECIES_SWITCHES),
            "raphael_species_switches": dict(RAPHAEL_BASELINE_SPECIES_SWITCHES),
            "data_root": str(BASELINES_DATA_ROOT),
            "lab_data_root": str(LAB_BASELINE_DATA_ROOT),
            "raphael_method_data_root": str(RAPHAEL_METHOD_DATA_ROOT),
            "raphael_model_path": str(RAPHAEL_MODEL_PATH or ""),
        },
        "pipeline_inference": {
            "day_enabled": bool(RUN_DAY_PIPELINE_INFERENCE),
            "night_enabled": bool(RUN_NIGHT_PIPELINE_INFERENCE),
            "global_enabled": bool(run_global_model_inference),
            "leaveout_enabled": bool(run_leaveout_model_inference),
            "day_species_switches": dict(DAY_INFERENCE_SPECIES_SWITCHES),
            "night_species_switches": dict(NIGHT_INFERENCE_SPECIES_SWITCHES),
        },
        "model_training": {
            "run_model_training": bool(RUN_MODEL_TRAINING),
            "day_enabled": bool(RUN_DAY_MODEL_TRAINING),
            "night_enabled": bool(RUN_NIGHT_MODEL_TRAINING),
            "global_enabled": bool(TRAIN_GLOBAL_MODELS),
            "leaveout_enabled": bool(TRAIN_LEAVEOUT_MODELS),
            "reuse_existing_if_present": bool(REUSE_EXISTING_MODELS_IF_PRESENT),
        },
        "day_yolo_training": {
            "run_day_yolo_model_training": bool(RUN_DAY_YOLO_MODEL_TRAINING),
            "global_enabled": bool(TRAIN_DAY_YOLO_GLOBAL_MODEL),
            "leaveout_enabled": bool(TRAIN_DAY_YOLO_LEAVEOUT_MODELS),
            "species_switches": dict(DAY_YOLO_TRAINING_SPECIES_SWITCHES),
            "dataset_root": str(DAY_YOLO_DATASET_ROOT),
            "global_models_root": str(DAY_YOLO_GLOBAL_MODELS_ROOT),
            "leaveout_models_root": str(DAY_YOLO_LEAVEOUT_MODELS_ROOT),
            "legacy_models_root": str(DAY_YOLO_LEGACY_MODELS_ROOT),
            "trained_models": {k: str(v.ckpt_path) for k, v in day_yolo_models.items()},
            "dataset_summaries": day_yolo_dataset_summaries,
            "training_manifests": day_yolo_training_manifests,
        },
        "model_selection": {
            "auto_discover_latest_day_model_root": bool(AUTO_DISCOVER_LATEST_DAY_MODEL_ROOT),
            "auto_discover_latest_night_model_root": bool(AUTO_DISCOVER_LATEST_NIGHT_MODEL_ROOT),
            "auto_discover_latest_day_yolo_model": bool(AUTO_DISCOVER_LATEST_DAY_YOLO_MODEL),
            "manual_day_model_root": str(DAY_PRETRAINED_MODEL_ROOT),
            "manual_night_model_root": str(NIGHT_PRETRAINED_MODEL_ROOT),
            "manual_day_yolo_model_weights": str(DAY_YOLO_MODEL_WEIGHTS),
            "resolved_day_model_root": str(resolved_pretrained_model_dirs.get("day") or ""),
            "resolved_night_model_root": str(resolved_pretrained_model_dirs.get("night") or ""),
            "resolved_day_model_root_source": str(resolved_pretrained_model_sources.get("day") or ""),
            "resolved_night_model_root_source": str(resolved_pretrained_model_sources.get("night") or ""),
            "resolved_day_yolo_model_weights": str(resolved_day_yolo_model or ""),
            "resolved_day_yolo_model_source": str(resolved_day_yolo_source or ""),
            "resolved_day_yolo_leaveout_source": str(resolved_day_yolo_leaveout_source or ""),
            "resolved_day_yolo_leaveout_models": {
                k: str(v.ckpt_path) for k, v in resolved_day_yolo_leaveout_models.items()
            },
        },
        "training_sources_by_route": {
            r: [{"species": s.species_name, "path": str(s.path)} for s in srcs]
            for r, srcs in sources_by_route.items()
        },
        "training_inference_catalog_summary": catalog_summary,
        "dataset_summaries": dataset_summaries,
        "training_metrics": training_metrics,
        "models_by_route": {
            route: {k: str(m.ckpt_path) for k, m in (mods or {}).items()} for route, mods in models_by_route.items()
        },
        "n_discovered_videos": int(len(routed_videos)),
        "n_eval_videos": int(len(eval_videos)),
        "n_skipped_videos": int(len(skipped_videos)),
        "skipped_videos": skipped_videos,
        "processed_gt_by_video_key": dict(processed_gt_by_video_key),
        "n_rows_written": int(len(rows_out)),
    }

    if not dry_run:
        run_root.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(record, indent=2))
        print(f"[tmp-run] metadata: {metadata_path}")

    failed_eval_rows = [
        r
        for r in rows_out
        if str(r.get("eval_type")) in {"global", "leaveout"}
        and str(r.get("results") or "").startswith("ERROR:")
    ]
    if failed_eval_rows:
        print(f"[tmp-run] completed with failures: {len(failed_eval_rows)} eval row(s) failed.")
        for r in failed_eval_rows[:20]:
            print(
                f"  - route={r.get('route')} model={r.get('model_used')} "
                f"video={r.get('video_name')} :: {r.get('results')}"
            )
        return 1

    if skipped_videos:
        print(f"[tmp-run] completed with skips: {len(skipped_videos)} video(s) skipped for missing GT.")

    print("[tmp-run] done.")
    print(f"[tmp-run] final csv: {final_csv_path}")
    print("[tmp-run] models used:")
    if resolved_day_yolo_model is not None:
        print(f"  - day_yolo ({resolved_day_yolo_source}): {resolved_day_yolo_model}")
    if resolved_day_yolo_leaveout_models:
        print(f"  - day_yolo_leaveout ({resolved_day_yolo_leaveout_source}):")
        for species in sorted(resolved_day_yolo_leaveout_models.keys()):
            print(f"      {species} -> {resolved_day_yolo_leaveout_models[species].ckpt_path}")
    for route in ("day", "night"):
        route_models = models_by_route.get(route) or {}
        if not route_models:
            continue
        print(f"  - {route}:")
        for mk in sorted(route_models.keys()):
            print(f"      {mk} -> {route_models[mk].ckpt_path}")
    print(f"[tmp-run] inference root: {inference_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
