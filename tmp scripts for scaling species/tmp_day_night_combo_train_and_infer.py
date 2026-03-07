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
4) Runs legacy baselines (lab + Raphael) for each video.
5) Runs gateway inference (day v3 / night pipeline) with the trained models.
6) Appends each completed result row to CSV immediately (flush + fsync).

Routing uses folder/file/species names only (no brightness routing filter).
"""

import argparse
import csv
import hashlib
import json
import os
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
VAL_INDIVIDUAL_ROOT: Path = PATCH_DATA_ROOT / "Integrated_prototype_validation_datasets" / "individual species folder"

# Raw videos root to infer on (can be overridden via CLI).
RAW_VIDEOS_ROOT: Path = Path("/mnt/Samsung_SSD_2TB/integrated prototype raw videos")

# Final outputs for this temporary run.
# Keep all generated outputs under this single root.
RUNS_ROOT: Path = Path("/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run")
RUN_NAME_PREFIX: str = "tmp_day_night_combo_train_and_infer"
FINAL_RESULTS_FILENAME: str = "final_results.csv"

# -----------------------------------------------------------------------------
# Run Component Switches (edit these first)
# -----------------------------------------------------------------------------
# 1) Ingestion
# If False, this script skips ingestion and uses already-ingested species
# datasets under PATCH_DATA_ROOT directly.
RUN_INGESTION_FROM_SCRATCH: bool = False

# 2) Model training
RUN_MODEL_TRAINING: bool = False

# 3) Baseline methods (Lab + Raphael)
# Set this True to run only baseline methods if day/night pipeline toggles below
# are both False.
RUN_BASELINE_METHODS_INFERENCE: bool = True
RUN_LAB_BASELINE: bool = True
RUN_RAPHAEL_BASELINE: bool = True

# 4) Your pipeline inference (split by route)
RUN_DAY_PIPELINE_INFERENCE: bool = False
RUN_NIGHT_PIPELINE_INFERENCE: bool = True

# Optional explicit model root when RUN_MODEL_TRAINING=False.
# Expected layout:
#   <root>/day/global_all_species.pt
#   <root>/day/leaveout_<species>.pt
#   <root>/night/global_all_species.pt
#   <root>/night/leaveout_<species>.pt
# If unset/None, the script auto-discovers the newest prior run under RUNS_ROOT
# that has all required model files for discovered species/routes.
PRETRAINED_MODELS_ROOT: Optional[Path] = None

# Evaluation safeguard: require per-video GT and cap processing to the last
# annotated frame in that GT.
REQUIRE_GT_FOR_INFERENCE: bool = True

# Route assignment by species token.
ROUTE_BY_SPECIES: Dict[str, str] = {
    "bicellonycha-wickershamorum": "night",
    "forresti": "night",
    "frontalis": "night",
    "photinus-acuminatus": "day",
    "photinus-carolinus": "night",
    "photinus-knulli": "night",
    "photuris-bethaniensis": "night",
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

# Baselines
BASELINES_ONLY_FOR_NIGHT: bool = False

# Lab baseline (no model file required)
LAB_BASELINE_THRESHOLD: float = 0.12
LAB_BASELINE_BLUR_SIGMA: float = 1.0
LAB_BASELINE_BKGR_WINDOW_SEC: float = 2.0

# Raphael baseline (requires TorchScript ffnet model)
RAPHAEL_MODEL_PATH = (
    "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/"
    "model zoo/Raphael's model/ffnet_best.pth"
)
RAPHAEL_BW_THR: float = 0.2
RAPHAEL_CLASSIFY_THR: float = 0.98
RAPHAEL_BKGR_WINDOW_SEC: float = 2.0
RAPHAEL_BLUR_SIGMA: float = 0.0
RAPHAEL_PATCH_SIZE_PX: int = 33
RAPHAEL_BATCH_SIZE: int = 1000
RAPHAEL_GAUSS_CROP_SIZE: int = 10
RAPHAEL_DEVICE: str = "auto"

# Stage5 validator settings for baselines (10px only to keep outputs smaller).
BASELINE_DIST_THRESHOLDS_PX: List[float] = [10.0]
BASELINE_VALIDATE_CROP_W: int = 10
BASELINE_VALIDATE_CROP_H: int = 10

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
    orch_dir = repo_root / "integrated ingestor-trainer-tester orchestrator"
    for p in (repo_root, orch_dir):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _repo_root() -> Path:
    return Path(REPO_ROOT).expanduser().resolve()


def _gateway_py() -> Path:
    p = _repo_root() / "integrated pipeline" / "gateway.py"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _day_pipeline_dir() -> Path:
    p = _repo_root() / "day time pipeline v3 (yolo + patch classifier ensemble)"
    if not (p / "stage5_validate.py").exists():
        raise FileNotFoundError(p / "stage5_validate.py")
    return p


def _baseline_scripts() -> Dict[str, Path]:
    base = _repo_root() / "tools" / "legacy_baselines"
    lab = base / "nolan_mp4_to_predcsv.py"
    raphael = base / "raphael_oorb_detect_and_gauss.py"
    return {"lab": lab, "raphael": raphael}


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

    # 2) Species validation annotations store.
    if species_name:
        sp_dir = VAL_INDIVIDUAL_ROOT / str(species_name)
        latest = _latest_version_dir(sp_dir)
        if latest is not None:
            ann = latest / "annotations.csv"
            rows = _read_gt_rows_from_csv(ann, video_stem=video_stem)
            if rows:
                return rows, ann

    # 3) Last-resort scan across species validation annotations.
    if VAL_INDIVIDUAL_ROOT.exists():
        for sp_dir in sorted([p for p in VAL_INDIVIDUAL_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name):
            latest = _latest_version_dir(sp_dir)
            if latest is None:
                continue
            ann = latest / "annotations.csv"
            rows = _read_gt_rows_from_csv(ann, video_stem=video_stem)
            if rows:
                return rows, ann

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

    import stage2_trainer as tr  # type: ignore

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = tr.train_resnet_classifier(
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


def _run_stage5_validator(
    *,
    orig_video_path: Path,
    pred_csv_path: Path,
    gt_csv_path: Path,
    max_frames: int | None,
    out_dir: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    day_dir = _day_pipeline_dir()

    # Prevent Stage5 validator from auto-searching for FN-scoring weights.
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
            f"    dist_thresholds={list(float(x) for x in BASELINE_DIST_THRESHOLDS_PX)!r},",
            f"    crop_w={int(BASELINE_VALIDATE_CROP_W)},",
            f"    crop_h={int(BASELINE_VALIDATE_CROP_H)},",
            "    gt_t_offset=0,",
            f"    max_frames={max_frames_code},",
            "    only_firefly_rows=True,",
            "    show_per_frame=False,",
            f"    model_path=Path({repr(str(no_weights))}),",
            "    print_load_status=False,",
            ")",
        ]
    )

    _run_subprocess_logged([sys.executable, "-c", code], cwd=day_dir, log_path=log_path, dry_run=dry_run)


def _video_max_frames_from_gt(video: EvalVideo) -> int | None:
    if video.gt_max_t is None:
        return None
    return int(video.gt_max_t) + 1


def _run_baselines_for_video(
    *,
    run_root: Path,
    video: EvalVideo,
    logs_dir: Path,
    dry_run: bool,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    scripts = _baseline_scripts()
    max_frames_for_video = _video_max_frames_from_gt(video)

    if BASELINES_ONLY_FOR_NIGHT and str(video.route) != "night":
        return out

    # Lab baseline
    if RUN_LAB_BASELINE:
        out_root = run_root / "baselines" / "lab_method" / f"{video.route}_videos" / video.video_key
        gt_csv = out_root / "ground truth" / "gt.csv"
        pred_csv = out_root / "predictions.csv"
        stage5_dir = out_root / "stage5 validation" / video.video_name
        if not dry_run:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_gt_csv(gt_csv, video.gt_rows)

        if stage5_dir.exists():
            metrics = _parse_validation_metrics(stage5_dir)
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
                        log_path=logs_dir / f"baseline_lab__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    _run_stage5_validator(
                        orig_video_path=video.video_path,
                        pred_csv_path=pred_csv,
                        gt_csv_path=gt_csv,
                        max_frames=max_frames_for_video,
                        out_dir=stage5_dir,
                        log_path=logs_dir / f"baseline_lab_stage5__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    metrics = _parse_validation_metrics(stage5_dir)
                except Exception as e:
                    metrics = {"error": f"lab baseline failed: {e}"}

        out["lab"] = {"metrics": metrics, "out_root": str(out_root)}

    # Raphael baseline
    if RUN_RAPHAEL_BASELINE:
        out_root = run_root / "baselines" / "raphael_method" / f"{video.route}_videos" / video.video_key
        gt_csv = out_root / "ground truth" / "gt.csv"
        pred_csv = out_root / "predictions.csv"
        raw_csv = out_root / "raw.csv"
        gauss_csv = out_root / "gauss.csv"
        stage5_dir = out_root / "stage5 validation" / video.video_name
        if not dry_run:
            out_root.mkdir(parents=True, exist_ok=True)
            _write_gt_csv(gt_csv, video.gt_rows)

        if stage5_dir.exists():
            metrics = _parse_validation_metrics(stage5_dir)
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
                        log_path=logs_dir / f"baseline_raphael__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    _run_stage5_validator(
                        orig_video_path=video.video_path,
                        pred_csv_path=pred_csv,
                        gt_csv_path=gt_csv,
                        max_frames=max_frames_for_video,
                        out_dir=stage5_dir,
                        log_path=logs_dir / f"baseline_raphael_stage5__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    metrics = _parse_validation_metrics(stage5_dir)
                except Exception as e:
                    metrics = {"error": f"raphael baseline failed: {e}"}

        out["raphael"] = {"metrics": metrics, "out_root": str(out_root)}

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
    dry_run: bool,
) -> tuple[Dict[str, ModelSpec], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    models: Dict[str, ModelSpec] = {}
    dataset_summaries: Dict[str, Dict[str, Any]] = {}
    training_metrics: Dict[str, Dict[str, Any]] = {}

    if not sources:
        return models, dataset_summaries, training_metrics

    all_species = sorted({s.species_name for s in sources})
    jobs: List[Tuple[str, Optional[str], List[SourceSpec]]] = []
    jobs.append(("global_all_species", None, list(sources)))

    for species in all_species:
        keep = [s for s in sources if s.species_name != species]
        if not keep:
            continue
        jobs.append((f"leaveout_{species}", species, keep))

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


def _expected_model_keys_for_sources(sources: Sequence[SourceSpec]) -> List[str]:
    all_species = sorted({s.species_name for s in sources})
    keys: List[str] = ["global_all_species"]
    for species in all_species:
        # Match _train_models_for_route: leaveout model exists only if at least
        # one other species remains in the route-specific training pool.
        if any(s.species_name != species for s in sources):
            keys.append(f"leaveout_{species}")
    return keys


def _model_root_has_required_models(
    *, model_root: Path, sources_by_route: Dict[str, List[SourceSpec]], dry_run: bool
) -> bool:
    for route in ("day", "night"):
        srcs = sources_by_route.get(route) or []
        if not srcs:
            continue
        for mk in _expected_model_keys_for_sources(srcs):
            ckpt = model_root / route / f"{mk}.pt"
            if (not dry_run) and (not ckpt.exists()):
                return False
    return True


def _auto_discover_pretrained_model_root(
    *, runs_root: Path, current_run_root: Path, sources_by_route: Dict[str, List[SourceSpec]], dry_run: bool
) -> Optional[Path]:
    candidates: List[Path] = []
    for run_dir in sorted(
        [p for p in runs_root.glob(f"{RUN_NAME_PREFIX}__*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if run_dir.resolve() == current_run_root.resolve():
            continue
        mroot = run_dir / "models"
        if not mroot.exists() and (not dry_run):
            continue
        candidates.append(mroot)

    for mroot in candidates:
        if _model_root_has_required_models(model_root=mroot, sources_by_route=sources_by_route, dry_run=dry_run):
            return mroot
    return None


def _load_models_for_route(
    *,
    route: str,
    sources: Sequence[SourceSpec],
    model_root: Path,
    dry_run: bool,
) -> Dict[str, ModelSpec]:
    models: Dict[str, ModelSpec] = {}
    if not sources:
        return models

    missing: List[Path] = []
    for model_key in _expected_model_keys_for_sources(sources):
        ckpt = model_root / route / f"{model_key}.pt"
        if (not dry_run) and (not ckpt.exists()):
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
            "Either enable RUN_MODEL_TRAINING=True or point PRETRAINED_MODELS_ROOT to a complete model set."
        )

    return models


def _has_thr_subdirs(dir_path: Path) -> bool:
    if not dir_path.exists() or not dir_path.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("thr_") for p in dir_path.iterdir())


def _day_analysis_outputs_ready(day_root: Path, video_name: str) -> bool:
    stage7 = day_root / "stage7 fn analysis" / video_name
    stage8 = day_root / "stage8 fp analysis" / video_name
    return _has_thr_subdirs(stage7) and _has_thr_subdirs(stage8)


def _run_gateway_for_video(
    *,
    out_root: Path,
    video: EvalVideo,
    model_path: Path,
    log_path: Path,
    dry_run: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns: (route_used, validation_metrics)
    """
    day_root = out_root / "day_pipeline_v3"
    night_root = out_root / "night_time_pipeline"
    stage_day = day_root / "stage5 validation" / video.video_name
    stage_night = night_root / "stage9 validation" / video.video_name

    # Resume-friendly cache: reuse existing valid outputs.
    if stage_day.exists():
        m = _parse_validation_metrics(stage_day)
        if not m.get("error") and _day_analysis_outputs_ready(day_root, video.video_name):
            return "day", m
        if not m.get("error"):
            print(
                f"[tmp-run][eval] cache incomplete for day video={video.video_name}: "
                "stage5 exists but stage7/stage8 analysis missing; rerunning gateway."
            )
    if stage_night.exists():
        m = _parse_validation_metrics(stage_night)
        if not m.get("error"):
            return "night", m

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
            return "day", m
        if not m.get("error"):
            return "day", {
                "error": (
                    f"day validation exists but analysis outputs are incomplete for {video.video_name}: "
                    f"missing stage7/stage8 under {day_root}"
                )
            }
        return "day", m
    if stage_night.exists():
        return "night", _parse_validation_metrics(stage_night)

    expected = stage_day if video.route == "day" else stage_night
    return video.route, {"error": f"no validation outputs found (expected under: {expected})"}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Temporary runner: route-split global+leaveout training + baselines + gateway inference "
            "with crash-safe CSV appends."
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

    if bool(RUN_INGESTION_FROM_SCRATCH):
        raise SystemExit(
            "RUN_INGESTION_FROM_SCRATCH=True is not implemented in this script. "
            "Set it to False to reuse existing ingested patch datasets."
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
    run_any_pipeline = bool(enabled_pipeline_routes)

    print(
        f"[tmp-run] baselines: run={run_baselines} lab={run_lab_baseline} "
        f"raphael={run_raphael_baseline} only_for_night={bool(BASELINES_ONLY_FOR_NIGHT)}"
    )
    print(
        f"[tmp-run] pipeline inference: day={bool(RUN_DAY_PIPELINE_INFERENCE)} "
        f"night={bool(RUN_NIGHT_PIPELINE_INFERENCE)}"
    )
    print("[tmp-run] ingestion stage: disabled (using existing ingested datasets)")
    print(f"[tmp-run] run_model_training={bool(RUN_MODEL_TRAINING)}")
    if (not run_baselines) and (not run_any_pipeline) and (not bool(RUN_MODEL_TRAINING)):
        raise SystemExit(
            "Nothing to run: RUN_MODEL_TRAINING=False, RUN_BASELINE_METHODS_INFERENCE=False, "
            "RUN_DAY_PIPELINE_INFERENCE=False, RUN_NIGHT_PIPELINE_INFERENCE=False."
        )

    sources_by_route = _discover_training_sources_by_route()
    print("[tmp-run] discovered training sources:")
    for route in ("day", "night"):
        srcs = sources_by_route.get(route) or []
        print(f"  - {route}: {len(srcs)} species")
        for s in srcs:
            print(f"      {s.species_name} -> {s.path}")

    models_by_route: Dict[str, Dict[str, ModelSpec]] = {"day": {}, "night": {}}
    dataset_summaries: Dict[str, Dict[str, Dict[str, Any]]] = {"day": {}, "night": {}}
    training_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {"day": {}, "night": {}}

    if bool(RUN_MODEL_TRAINING):
        for route in ("day", "night"):
            srcs = sources_by_route.get(route) or []
            if not srcs:
                print(f"[tmp-run] WARNING: no training sources for route={route}; skipping model training.")
                continue

            route_models, route_ds, route_metrics = _train_models_for_route(
                route=route,
                sources=srcs,
                combined_dataset_root=combined_dataset_root,
                model_root=model_root,
                dry_run=dry_run,
            )
            models_by_route[route] = route_models
            dataset_summaries[route] = route_ds
            training_metrics[route] = route_metrics
    elif run_any_pipeline:
        sources_for_model_lookup: Dict[str, List[SourceSpec]] = {
            "day": list(sources_by_route.get("day") or []) if "day" in enabled_pipeline_routes else [],
            "night": list(sources_by_route.get("night") or []) if "night" in enabled_pipeline_routes else [],
        }
        explicit_model_root = PRETRAINED_MODELS_ROOT
        resolved_model_root: Optional[Path] = None
        if explicit_model_root is not None:
            resolved_model_root = Path(str(explicit_model_root)).expanduser().resolve()
            if (not dry_run) and (not resolved_model_root.exists()):
                raise SystemExit(f"PRETRAINED_MODELS_ROOT does not exist: {resolved_model_root}")
        else:
            resolved_model_root = _auto_discover_pretrained_model_root(
                runs_root=runs_root,
                current_run_root=run_root,
                sources_by_route=sources_for_model_lookup,
                dry_run=dry_run,
            )
            if resolved_model_root is None:
                raise SystemExit(
                    "RUN_MODEL_TRAINING=False but no compatible pretrained model root was found.\n"
                    "Set PRETRAINED_MODELS_ROOT to an existing models directory, or set RUN_MODEL_TRAINING=True."
                )

        model_root = resolved_model_root
        print(f"[tmp-run] reusing pretrained models from: {model_root}")
        for route in enabled_pipeline_routes:
            srcs = sources_by_route.get(route) or []
            if not srcs:
                continue
            models_by_route[route] = _load_models_for_route(
                route=route,
                sources=srcs,
                model_root=model_root,
                dry_run=dry_run,
            )
    else:
        print("[tmp-run] model prep skipped (training disabled and both day/night pipeline inference toggles are off).")

    known_species = sorted(
        set(list(ROUTE_BY_SPECIES.keys()) + [s.species_name for vv in sources_by_route.values() for s in vv])
    )
    routed_videos = _discover_inference_videos(raw_videos_root, known_species)

    max_videos = int(args.max_videos or 0)
    if max_videos > 0:
        routed_videos = routed_videos[:max_videos]

    print(f"[tmp-run] videos discovered: {len(routed_videos)}")
    by_route_counts = {"day": 0, "night": 0}
    for v in routed_videos:
        by_route_counts[v.route] = by_route_counts.get(v.route, 0) + 1
    print(f"[tmp-run] video split: day={by_route_counts.get('day', 0)} night={by_route_counts.get('night', 0)}")
    print(f"[tmp-run] require_gt_for_inference={bool(REQUIRE_GT_FOR_INFERENCE)}")

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
        print(f"[baselines] enabled={run_baselines} only_for_night={BASELINES_ONLY_FOR_NIGHT}")
        for i, vid in enumerate(eval_videos, start=1):
            print(f"[baselines] {i}/{len(eval_videos)} {vid.species_name} :: {vid.video_path.name} (route={vid.route})")
            if not dry_run:
                baseline_by_video_key[vid.video_key] = _run_baselines_for_video(
                    run_root=run_root,
                    video=vid,
                    logs_dir=logs_dir,
                    dry_run=bool(dry_run),
                )
            # Persist baseline progress immediately per video so crashes do not
            # lose completed baseline results.
            lab_metrics, lab_path, raphael_metrics, raphael_path = _baseline_fields(vid.video_key)
            _record_row(
                {
                    "run_id": run_id,
                    "route": vid.route,
                    "species_name": vid.species_name,
                    "video_name": vid.video_name,
                    "eval_type": "baseline",
                    "model_used": "lab+raphael",
                    "results": "",
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
    if run_baselines and (not run_any_pipeline):
        print("[eval] pipeline inference disabled; baseline rows already written.")

    videos_by_route: Dict[str, List[EvalVideo]] = {"day": [], "night": []}
    for v in eval_videos:
        videos_by_route.setdefault(v.route, []).append(v)

    # Global model on all videos in each route.
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
                    _, metrics = _run_gateway_for_video(
                        out_root=out_root,
                        video=vid,
                        model_path=global_model.ckpt_path,
                        log_path=log_path,
                        dry_run=bool(dry_run),
                    )
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

    # Leaveout models on their left-out species videos (within same route only).
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

                if dry_run:
                    metrics = {"error": "dry_run"}
                else:
                    try:
                        _, metrics = _run_gateway_for_video(
                            out_root=out_root,
                            video=vid,
                            model_path=lm.ckpt_path,
                            log_path=log_path,
                            dry_run=bool(dry_run),
                        )
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

    record: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "dry_run": dry_run,
        "paths": {
            "run_root": str(run_root),
            "raw_videos_root": str(raw_videos_root),
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
            "only_for_night": bool(BASELINES_ONLY_FOR_NIGHT),
            "lab_enabled": bool(run_lab_baseline),
            "raphael_enabled": bool(run_raphael_baseline),
            "raphael_model_path": str(RAPHAEL_MODEL_PATH or ""),
        },
        "pipeline_inference": {
            "day_enabled": bool(RUN_DAY_PIPELINE_INFERENCE),
            "night_enabled": bool(RUN_NIGHT_PIPELINE_INFERENCE),
        },
        "model_training": {
            "run_model_training": bool(RUN_MODEL_TRAINING),
            "reuse_existing_if_present": bool(REUSE_EXISTING_MODELS_IF_PRESENT),
            "pretrained_models_root": str(PRETRAINED_MODELS_ROOT) if PRETRAINED_MODELS_ROOT is not None else "",
        },
        "training_sources_by_route": {
            r: [{"species": s.species_name, "path": str(s.path)} for s in srcs]
            for r, srcs in sources_by_route.items()
        },
        "dataset_summaries": dataset_summaries,
        "training_metrics": training_metrics,
        "models_by_route": {
            route: {k: str(m.ckpt_path) for k, m in (mods or {}).items()} for route, mods in models_by_route.items()
        },
        "n_discovered_videos": int(len(routed_videos)),
        "n_eval_videos": int(len(eval_videos)),
        "n_skipped_videos": int(len(skipped_videos)),
        "skipped_videos": skipped_videos,
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
