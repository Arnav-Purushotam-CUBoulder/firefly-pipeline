#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import errno
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import stage1_ingestor
from codex_change_log import ChangeLogRun, SnapshotConfig, build_ingestion_index
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

# Global-model evaluation:
# If True, when evaluating the GLOBAL model we will re-run inference/validation on *all*
# validation videos accumulated so far across all species (from the combined validation annotations.csv),
# not just the held-out videos from the current OBSERVED_DATA_DIR.
EVAL_GLOBAL_MODEL_ON_ALL_VALIDATION_VIDEOS: bool = True

# Validation video store (for visibility + stable automation input paths).
# The orchestrator will maintain:
#   <ROOT>/validation videos/<species_name>/*.mp4
VALIDATION_VIDEOS_DIRNAME: str = "validation videos"
AUTO_SYNC_VALIDATION_VIDEOS_STORE: bool = True
VALIDATION_VIDEOS_STORE_MODE: str = "hardlink"  # "hardlink" | "copy"

# Training video metadata store (no video files, just metadata).
# The orchestrator will maintain versioned snapshots under:
#   <ROOT>/training videos/training_videos__vN_<timestamp>.json
TRAINING_VIDEOS_DIRNAME: str = "training videos"
TRAINING_VIDEOS_SNAPSHOT_PREFIX: str = "training_videos"

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

# If True, write *all* terminal output (including subprocess output) to a per-run log file.
# The log file is ultimately placed under: <ROOT>/inference outputs/<run_id>/
SAVE_RUN_LOG: bool = True
RUN_LOG_FILENAME: str = "run.log"

# Append-only change log (rollback-focused)
# Writes to: <LOG_ROOT>/codex_change_log.jsonl
ENABLE_CODEX_CHANGE_LOG: bool = True
LOG_ROOT_PATH: str | Path = "/mnt/Samsung_SSD_2TB/integrated prototype data"
CHANGE_LOG_FILENAME: str = "codex_change_log.jsonl"

# -----------------------------------------------------------------------------
# Legacy baseline evaluation (optional)
# -----------------------------------------------------------------------------

# If True, also run legacy/lab baseline detectors on the same eval videos and score
# them using the same point-distance validation logic (Stage5 validator).
RUN_BASELINE_EVAL: bool = True

# If True, only run baselines on videos routed to the night pipeline.
BASELINES_ONLY_FOR_NIGHT: bool = True

# Baseline validator settings.
# Validate only at 10px to reduce compute + output size.
BASELINE_DIST_THRESHOLDS_PX: List[float] = [10.0]
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

MODEL_ZOO_RESULTS_HISTORY_DIRNAME = "results_history"
MODEL_ZOO_RESULTS_HISTORY_SNAPSHOT_PREFIX = "results_history"
MODEL_ZOO_VIDEO_REGISTRY_DIRNAME = "video_registry"
MODEL_ZOO_VIDEO_REGISTRY_SNAPSHOT_PREFIX = "video_registry"

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


class _RunLogTee(io.TextIOBase):
    def __init__(self, *, orig: io.TextIOBase, logger: "_RunOutputLogger"):
        super().__init__()
        self._orig = orig
        self._logger = logger

    def write(self, s: str) -> int:  # type: ignore[override]
        self._logger._write_text(s, self._orig)
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        self._logger._flush(self._orig)

    def isatty(self) -> bool:  # pragma: no cover
        try:
            return bool(self._orig.isatty())
        except Exception:
            return False

    @property
    def encoding(self) -> str:  # pragma: no cover
        return getattr(self._orig, "encoding", "utf-8")

    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        return getattr(self._orig, name)


class _RunOutputLogger:
    """
    Best-effort tee of:
      1) Python prints (stdout/stderr), and
      2) subprocess output invoked by this orchestrator
    into a single per-run log file.

    The log file starts in: <ROOT>/inference outputs/_run_logs/
    and is moved into: <ROOT>/inference outputs/<run_id>/run.log
    once the run_id is known.
    """

    def __init__(self, *, inference_root: Path, species_name: str, time_of_day: str, enabled: bool, dry_run: bool):
        self._enabled = bool(enabled)
        self._dry_run = bool(dry_run)
        self._lock = threading.Lock()

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._tee_stdout = _RunLogTee(orig=self._orig_stdout, logger=self)
        self._tee_stderr = _RunLogTee(orig=self._orig_stderr, logger=self)

        self._fh: io.BufferedWriter | None = None
        self._path: Path | None = None

        if self._enabled and not self._dry_run:
            log_dir = Path(inference_root).expanduser().resolve() / "_run_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = "__".join(
                p
                for p in (_safe_name(species_name), _safe_name(time_of_day))
                if str(p or "").strip()
            )
            name = f"{ts}__{base}.log" if base else f"{ts}.log"
            self._path = log_dir / name
            self._fh = self._path.open("ab", buffering=0)

    @property
    def path(self) -> Path | None:
        return self._path

    def install(self) -> None:
        if not self._enabled:
            return
        sys.stdout = self._tee_stdout  # type: ignore[assignment]
        sys.stderr = self._tee_stderr  # type: ignore[assignment]

    def close(self) -> None:
        if not self._enabled:
            return
        try:
            sys.stdout = self._orig_stdout  # type: ignore[assignment]
            sys.stderr = self._orig_stderr  # type: ignore[assignment]
        finally:
            try:
                if self._fh is not None:
                    self._fh.flush()
                    self._fh.close()
            except Exception:
                pass

    def move_into_run_dir(self, run_dir: Path) -> None:
        if not (self._enabled and self._fh is not None and self._path is not None):
            return
        try:
            run_dir = Path(run_dir).expanduser().resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            final = run_dir / str(RUN_LOG_FILENAME)
            os.replace(str(self._path), str(final))
            self._path = final
        except Exception:
            # Best-effort only; keep writing to the existing path.
            pass

    def _write_text(self, s: str, orig: io.TextIOBase) -> None:
        if not self._enabled:
            orig.write(s)
            return
        with self._lock:
            try:
                orig.write(s)
                orig.flush()
            except Exception:
                pass
            try:
                if self._fh is not None:
                    self._fh.write(s.encode("utf-8", errors="replace"))
            except Exception:
                pass

    def _flush(self, orig: io.TextIOBase) -> None:
        if not self._enabled:
            try:
                orig.flush()
            except Exception:
                pass
            return
        with self._lock:
            try:
                orig.flush()
            except Exception:
                pass
            try:
                if self._fh is not None:
                    self._fh.flush()
            except Exception:
                pass

    def write_subprocess_bytes(self, b: bytes) -> None:
        if not b:
            return
        with self._lock:
            try:
                # Prefer raw bytes to preserve carriage returns/progress output.
                buf = getattr(self._orig_stdout, "buffer", None)
                if buf is not None:
                    buf.write(b)
                    buf.flush()
                else:
                    self._orig_stdout.write(b.decode("utf-8", errors="replace"))
                    self._orig_stdout.flush()
            except Exception:
                pass
            try:
                if self._fh is not None:
                    self._fh.write(b)
            except Exception:
                pass

    def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path,
        check: bool,
    ) -> None:
        if not self._enabled or self._dry_run:
            subprocess.run(list(cmd), cwd=str(cwd), check=bool(check))
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                self.write_subprocess_bytes(chunk)
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

        rc = int(proc.wait())
        if check and rc != 0:
            raise subprocess.CalledProcessError(rc, list(cmd))


_RUN_OUTPUT_LOGGER: _RunOutputLogger | None = None


def _subprocess_run(cmd: Sequence[str], *, cwd: Path, check: bool = True) -> None:
    logger = _RUN_OUTPUT_LOGGER
    if logger is not None:
        logger.run(cmd, cwd=cwd, check=bool(check))
    else:
        subprocess.run(list(cmd), cwd=str(cwd), check=bool(check))

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

    # Legacy (single-file) paths. These are no longer mutated in-place.
    legacy_history = model_root / "results_history.jsonl"
    legacy_registry = model_root / "video_registry.json"

    # New (append-only via snapshots) metadata dirs.
    history_dir = model_root / MODEL_ZOO_RESULTS_HISTORY_DIRNAME
    registry_dir = model_root / MODEL_ZOO_VIDEO_REGISTRY_DIRNAME
    history_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    # Seed snapshot dirs from legacy files on first run.
    _migrate_legacy_results_history_to_snapshots(legacy_path=legacy_history, history_dir=history_dir)
    _migrate_legacy_registry_to_snapshots(legacy_path=legacy_registry, registry_dir=registry_dir)

    return {
        "day_global": day_global,
        "night_global": night_global,
        "single_root": single_root,
        "results_history_dir": history_dir,
        "video_registry_dir": registry_dir,
        "legacy_results_history_file": legacy_history,
        "legacy_video_registry_file": legacy_registry,
    }


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _snapshot_num_from_name(*, prefix: str, name: str) -> Optional[int]:
    m = re.match(rf"^{re.escape(prefix)}__v(?P<n>\\d+)(?:_|$)", str(name))
    if not m:
        return None
    try:
        return int(m.group("n"))
    except Exception:
        return None


def _latest_snapshot_file(*, root: Path, prefix: str, suffix: str) -> Tuple[int, Path] | None:
    if not root.exists():
        return None
    best: Tuple[int, str, Path] | None = None
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != suffix.lower():
            continue
        n = _snapshot_num_from_name(prefix=prefix, name=p.name)
        if n is None:
            continue
        key = (n, p.name, p)
        if best is None or key[0] > best[0] or (key[0] == best[0] and key[1] > best[1]):
            best = key
    return (best[0], best[2]) if best else None


def _migrate_legacy_registry_to_snapshots(*, legacy_path: Path, registry_dir: Path) -> None:
    """
    One-time migration helper: if a legacy `video_registry.json` exists but there are no snapshots yet,
    create `video_registry__v1_<timestamp>.json` under `registry_dir`.
    """
    if not legacy_path.exists():
        return
    if _latest_snapshot_file(root=registry_dir, prefix=MODEL_ZOO_VIDEO_REGISTRY_SNAPSHOT_PREFIX, suffix=".json") is not None:
        return
    try:
        reg = _load_registry(legacy_path)
    except Exception:
        reg = {}
    out = registry_dir / f"{MODEL_ZOO_VIDEO_REGISTRY_SNAPSHOT_PREFIX}__v1_{_run_tag()}.json"
    _save_registry(out, reg)


def _migrate_legacy_results_history_to_snapshots(*, legacy_path: Path, history_dir: Path) -> None:
    """
    One-time migration helper: if a legacy `results_history.jsonl` exists but there are no snapshots yet,
    create `results_history__v1_<timestamp>.jsonl` under `history_dir`.
    """
    if _latest_snapshot_file(root=history_dir, prefix=MODEL_ZOO_RESULTS_HISTORY_SNAPSHOT_PREFIX, suffix=".jsonl") is not None:
        return
    # If legacy exists, copy; else write an empty v1.
    out = history_dir / f"{MODEL_ZOO_RESULTS_HISTORY_SNAPSHOT_PREFIX}__v1_{_run_tag()}.jsonl"
    if legacy_path.exists():
        try:
            out.write_text(legacy_path.read_text())
            return
        except Exception:
            pass
    out.write_text("")


def _load_latest_video_registry(*, registry_dir: Path, legacy_path: Path) -> Tuple[Dict[str, str], int, Path | None]:
    """
    Load latest registry snapshot if present; else fall back to legacy file.
    Returns: (registry_dict, prev_version_number, prev_path)
    """
    latest = _latest_snapshot_file(root=registry_dir, prefix=MODEL_ZOO_VIDEO_REGISTRY_SNAPSHOT_PREFIX, suffix=".json")
    if latest is not None:
        n, p = latest
        return _load_registry(p), int(n), p
    if legacy_path.exists():
        return _load_registry(legacy_path), 0, legacy_path
    return {}, 0, None


def _write_video_registry_snapshot(
    *,
    registry_dir: Path,
    registry: Dict[str, str],
    prev_version: int,
    dry_run: bool,
) -> Path | None:
    new_n = int(prev_version) + 1
    out = registry_dir / f"{MODEL_ZOO_VIDEO_REGISTRY_SNAPSHOT_PREFIX}__v{new_n}_{_run_tag()}.json"
    if dry_run:
        print(f"[dry-run] Would write video registry snapshot → {out}")
        return None
    _save_registry(out, registry)
    return out


def _append_results_history_snapshot(
    *,
    history_dir: Path,
    legacy_path: Path,
    record: Dict[str, Any],
    dry_run: bool,
) -> Path | None:
    """
    Append a run record by creating a NEW snapshot file:
      results_history__vN_<timestamp>.jsonl
    which is a copy of the latest snapshot + one appended JSONL line.
    """
    latest = _latest_snapshot_file(root=history_dir, prefix=MODEL_ZOO_RESULTS_HISTORY_SNAPSHOT_PREFIX, suffix=".jsonl")
    if latest is not None:
        prev_n, prev_path = latest
    elif legacy_path.exists():
        prev_n, prev_path = 0, legacy_path
    else:
        prev_n, prev_path = 0, None

    new_n = int(prev_n) + 1
    out = history_dir / f"{MODEL_ZOO_RESULTS_HISTORY_SNAPSHOT_PREFIX}__v{new_n}_{_run_tag()}.jsonl"
    if dry_run:
        print(f"[dry-run] Would append results history snapshot → {out}")
        return None

    history_dir.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as wf:
        if prev_path is not None and Path(prev_path).exists():
            with Path(prev_path).open("rb") as rf:
                shutil.copyfileobj(rf, wf)

    # Ensure file ends with newline before appending.
    needs_nl = False
    try:
        if out.stat().st_size > 0:
            with out.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                last = f.read(1)
            if last not in {b"\n"}:
                needs_nl = True
    except Exception:
        needs_nl = True

    with out.open("ab") as f:
        if needs_nl:
            f.write(b"\n")
        f.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))

    return out


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


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    """
    Create dst as either a hardlink to src (preferred) or a full copy.
    Falls back to copy if hardlink fails (e.g. cross-device).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    mode = (mode or "hardlink").strip().lower()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError as e:
            if e.errno in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EEXIST}:
                shutil.copy2(src, dst)
                return
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def _ensure_validation_video_store(root: Path) -> Path:
    store = Path(root) / VALIDATION_VIDEOS_DIRNAME
    store.mkdir(parents=True, exist_ok=True)
    return store


def _resolve_validation_video_path(
    *,
    video_name: str,
    species_name: str,
    store_root: Path,
    individual_validation_root: Path,
    registry: Dict[str, str],
    extra_search_dirs: Sequence[Path],
) -> Tuple[Path | None, str]:
    """
    Resolve a validation video path for (species_name, video_name).
    Returns (path, source) where source is a short string for run-record/debugging.
    """
    vn = _safe_name(video_name)
    sp = _safe_name(species_name)
    if not vn or not sp:
        return None, "invalid_names"

    # 1) Preferred: stable visible store under ROOT/validation videos/<species>/...
    sp_store = Path(store_root) / sp
    for ext in sorted(VIDEO_EXTS):
        cand = sp_store / f"{vn}{ext}"
        if cand.exists():
            return cand, "validation_videos_store"

    # 2) Existing individual validation datasets (if present on disk).
    sp_ind = Path(individual_validation_root) / sp
    if sp_ind.exists():
        for ext in sorted(VIDEO_EXTS):
            for cand in sp_ind.glob(f"**/videos/{vn}{ext}"):
                if cand.exists():
                    return cand, "individual_validation_dataset"

    # 3) Registry (may point to original observed/raw location).
    reg_path = registry.get(vn) or registry.get(video_name)
    if reg_path:
        try:
            rp = Path(reg_path).expanduser().resolve()
            if rp.exists():
                return rp, "video_registry"
        except Exception:
            pass

    # 4) Fallback: user-provided search roots.
    found = _find_video_by_stem(vn, extra_search_dirs)
    if found is not None and found.exists():
        return found, "extra_search_dirs"

    return None, "not_found"


def _sync_validation_videos_for_pairs(
    *,
    pairs: Sequence[Any],
    species_name: str,
    store_root: Path,
    mode: str,
    dry_run: bool,
) -> None:
    """
    Ensure held-out validation videos from the *current* run are visible under:
      <ROOT>/validation videos/<species>/<video_name>.<ext>
    """
    sp_dir = Path(store_root) / str(species_name)
    if not dry_run:
        sp_dir.mkdir(parents=True, exist_ok=True)

    for p in pairs:
        try:
            video_name = str(p.video_name)
            video_path = Path(p.video_path)
        except Exception:
            continue
        if not video_name or not video_path.exists():
            continue

        ext = video_path.suffix.lower()
        if ext not in VIDEO_EXTS:
            ext = ".mp4"
        dst = sp_dir / f"{video_name}{ext}"
        if dst.exists():
            continue
        if dry_run:
            print(f"[dry-run] Would sync validation video -> {dst}  (src={video_path})")
            continue
        _link_or_copy(video_path, dst, mode=mode)


def _ensure_training_videos_root(root: Path) -> Path:
    d = Path(root) / TRAINING_VIDEOS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _training_snapshot_num_from_name(name: str) -> Optional[int]:
    """
    Parse: training_videos__v12_20260219__235959.json -> 12
    """
    m = re.match(rf"^{re.escape(TRAINING_VIDEOS_SNAPSHOT_PREFIX)}__v(?P<n>\\d+)(?:_|$)", str(name))
    if not m:
        return None
    try:
        return int(m.group("n"))
    except Exception:
        return None


def _latest_training_snapshot(training_root: Path) -> Tuple[int, Path] | None:
    if not training_root.exists():
        return None
    best: Tuple[int, str, Path] | None = None
    for p in training_root.iterdir():
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        n = _training_snapshot_num_from_name(p.name)
        if n is None:
            continue
        key = (n, p.name, p)
        if best is None or key[0] > best[0] or (key[0] == best[0] and key[1] > best[1]):
            best = key
    return (best[0], best[2]) if best else None


def _collect_legacy_species_imports(data_root: Path) -> List[Dict[str, Any]]:
    """
    Best-effort reconstruction of "training sources" for the initial legacy species.
    These legacy imports do not have per-video provenance; we store their import_manifest.json info.
    """
    out: List[Dict[str, Any]] = []
    single_root = data_root / "Integrated_prototype_datasets" / "single species datasets"
    if not single_root.exists():
        return out
    for sp_dir in sorted([p for p in single_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        for mf in sorted(sp_dir.glob("v*/import_manifest.json")):
            try:
                data = json.loads(mf.read_text() or "{}")
            except Exception:
                continue
            if str(data.get("kind") or "") != "single_species":
                continue
            notes = data.get("notes") if isinstance(data.get("notes"), list) else []
            notes_txt = " ".join(str(x) for x in notes)
            # Only treat these as legacy if they explicitly say so.
            if "Imported legacy dataset" not in notes_txt:
                continue
            out.append(
                {
                    "species_name": str(data.get("species") or sp_dir.name),
                    "import_manifest_path": str(mf),
                    "imported_at": data.get("imported_at"),
                    "source": data.get("source"),
                    "dest": data.get("dest"),
                    "counts": data.get("counts"),
                    "split": data.get("split"),
                    "notes": notes,
                }
            )
    return out


def _bootstrap_training_videos_payload(data_root: Path) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "legacy_species_imports": _collect_legacy_species_imports(data_root),
        "runs": [],
    }


def _update_training_videos_history(
    *,
    root: Path,
    data_root: Path,
    run_id: str,
    species_name: str,
    observed_dir: Path,
    time_of_day: str,
    expected_route: str,
    train_pair_fraction: float,
    train_pairs: Sequence[Any],
    val_pairs: Sequence[Any],
    integrated_dataset_version: Path | None,
    single_species_dataset_version: Path | None,
    did_train: bool,
    dry_run: bool,
) -> Path | None:
    """
    Maintain versioned JSON snapshots under:
      <ROOT>/training videos/training_videos__vN_<timestamp>.json

    Each new snapshot contains all previous runs + the new run appended.
    """
    training_root = _ensure_training_videos_root(root)
    prev = _latest_training_snapshot(training_root)
    if prev is None:
        payload: Dict[str, Any] = _bootstrap_training_videos_payload(data_root)
        prev_n = 0
        prev_path = None
    else:
        prev_n, prev_path = prev
        try:
            payload = json.loads(Path(prev_path).read_text() or "{}")
            if not isinstance(payload, dict):
                payload = _bootstrap_training_videos_payload(data_root)
        except Exception:
            payload = _bootstrap_training_videos_payload(data_root)

    runs = payload.get("runs")
    if not isinstance(runs, list):
        runs = []

    # Per-video metadata for training split.
    train_items: List[Dict[str, Any]] = []
    train_rows_firefly_total = 0
    train_rows_background_total = 0
    for p in train_pairs:
        try:
            vn = str(p.video_name)
            vp = Path(p.video_path)
            ff_csv = Path(p.firefly_csv)
            bg_csv = Path(p.background_csv) if getattr(p, "background_csv", None) else None
        except Exception:
            continue

        n_ff = 0
        n_bg = 0
        try:
            n_ff = len(_read_annotator_csv(ff_csv)) if ff_csv.exists() else 0
        except Exception:
            n_ff = 0
        if bg_csv is not None:
            try:
                n_bg = len(_read_annotator_csv(bg_csv)) if bg_csv.exists() else 0
            except Exception:
                n_bg = 0

        train_rows_firefly_total += int(n_ff)
        train_rows_background_total += int(n_bg)

        st: os.stat_result | None = None
        try:
            st = vp.stat() if vp.exists() else None
        except Exception:
            st = None
        train_items.append(
            {
                "video_name": vn,
                "video_path": str(vp),
                "exists": bool(vp.exists()),
                "bytes": int(st.st_size) if st is not None else None,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds") if st is not None else None,
                "firefly_csv": str(ff_csv),
                "background_csv": str(bg_csv) if bg_csv is not None else None,
                "n_firefly_rows": int(n_ff),
                "n_background_rows": int(n_bg),
            }
        )

    entry = {
        "run_id": str(run_id),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "species_name": str(species_name),
        "observed_dir": str(observed_dir),
        "time_of_day": str(time_of_day),
        "expected_route": str(expected_route),
        "train_pair_fraction": float(train_pair_fraction),
        "n_pairs_total": int(len(train_pairs) + len(val_pairs)),
        "n_pairs_train": int(len(train_pairs)),
        "n_pairs_validation": int(len(val_pairs)),
        "did_train_models": bool(did_train),
        "dataset_versions": {
            "integrated": integrated_dataset_version.name if integrated_dataset_version else None,
            "integrated_path": str(integrated_dataset_version) if integrated_dataset_version else None,
            "single_species": single_species_dataset_version.name if single_species_dataset_version else None,
            "single_species_path": str(single_species_dataset_version) if single_species_dataset_version else None,
        },
        "train_rows": {"firefly": int(train_rows_firefly_total), "background": int(train_rows_background_total)},
        "train_videos": train_items,
        "notes": [
            "This file stores training-video METADATA only (no video files) to avoid high storage usage.",
            "Legacy imported datasets may not have per-video provenance; see legacy_species_imports.",
        ],
    }
    runs.append(entry)
    payload["runs"] = runs
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["last_run_id"] = str(run_id)
    if prev_path is not None:
        payload["previous_snapshot"] = str(prev_path)

    new_n = int(prev_n) + 1
    out_path = training_root / f"{TRAINING_VIDEOS_SNAPSHOT_PREFIX}__v{new_n}_{_run_tag()}.json"

    if dry_run:
        print(f"[dry-run] Would write training videos history → {out_path}")
        return None

    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _group_validation_rows_by_species_and_video(
    rows: Sequence[Dict[str, str]]
) -> Dict[Tuple[str, str], List[Dict[str, int]]]:
    """
    Group combined validation CSV rows by (species_name, video_name).
    Returns rows in the gt-row format used by _write_gt_csv (x,y,t,...).
    """
    out: Dict[Tuple[str, str], List[Dict[str, int]]] = {}
    for r in rows:
        video_name = _safe_name(r.get("video_name", "") or "")
        species_name = _safe_name(r.get("species_name", "") or "")
        if not video_name or not species_name:
            continue
        try:
            x = int(round(float(r.get("x", "") or 0)))
            y = int(round(float(r.get("y", "") or 0)))
            t = int(round(float(r.get("t", "") or 0)))
        except Exception:
            continue
        out.setdefault((species_name, video_name), []).append({"x": x, "y": y, "t": t, "w": 0, "h": 0})
    return out


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

    _subprocess_run(cmd, cwd=repo_root, check=True)


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
    _subprocess_run(cmd, cwd=repo_root, check=True)


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
    _subprocess_run(cmd, cwd=repo_root, check=True)


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
    _subprocess_run([sys.executable, "-c", code], cwd=day_dir, check=True)


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
    _subprocess_run([sys.executable, "-c", code], cwd=day_dir, check=True)


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
    p.add_argument(
        "--log-root",
        type=str,
        default=None,
        help=(
            "Where to write the append-only Codex changelog (JSONL). "
            "Defaults to LOG_ROOT_PATH (or <root>/.. if LOG_ROOT_PATH is empty)."
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
    # stage1_ingestor_core parses identity from staged CSV name split on underscores,
    # so species tokens must NOT contain underscores (use hyphens instead).
    species_norm = str(species_name).strip().lower().replace("_", "-")
    species_norm = re.sub(r"[^a-z0-9-]+", "-", species_norm)
    species_norm = re.sub(r"-+", "-", species_norm).strip("-")
    if not species_norm:
        raise SystemExit(f"Invalid species name after normalization: {species_arg!r}")
    if species_norm != species_name:
        print(f"[orchestrator] NOTE: normalized species_name {species_name!r} -> {species_norm!r}")
    species_name = species_norm

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

    # ---------------------------------------------------------------------
    # Append-only change log (best-effort)
    # ---------------------------------------------------------------------
    log_root_arg = (str(args.log_root).strip() if args.log_root else "") or (str(LOG_ROOT_PATH).strip() if LOG_ROOT_PATH else "")
    if not log_root_arg:
        log_root = root.parent
    else:
        log_root = Path(log_root_arg).expanduser().resolve()
    log_path = (log_root / str(CHANGE_LOG_FILENAME)).expanduser().resolve()

    changelog_meta: Dict[str, Any] = {
        "actor": "orchestrator",
        "argv": [str(x) for x in sys.argv],
        "root": str(root),
        "log_path": str(log_path),
        "species_name": species_name,
        "observed_dir": str(observed_dir),
        "type_of_video": str(video_type_norm),
        "time_of_day": str(time_of_day),
        "expected_route": str(expected_route),
        "dry_run": bool(args.dry_run),
        "skip_ingest": bool(getattr(args, "skip_ingest", False)),
        "skip_train": bool(getattr(args, "skip_train", False)),
        "skip_test": bool(getattr(args, "skip_test", False)),
    }
    if bool(ENABLE_CODEX_CHANGE_LOG) and (not bool(args.dry_run)):
        try:
            log_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    changelog_run: ChangeLogRun | None = None
    if bool(ENABLE_CODEX_CHANGE_LOG):
        try:
            scopes = [
                (root / DEFAULT_DATA_SUBDIR).expanduser().resolve(),
                (root / DEFAULT_MODEL_ZOO_SUBDIR).expanduser().resolve(),
                (root / DEFAULT_INFERENCE_OUTPUT_SUBDIR).expanduser().resolve(),
                (root / VALIDATION_VIDEOS_DIRNAME).expanduser().resolve(),
                (root / TRAINING_VIDEOS_DIRNAME).expanduser().resolve(),
            ]
            cfg = SnapshotConfig(root=root, scopes=scopes)
            changelog_run = ChangeLogRun(cfg=cfg, log_path=log_path, meta=changelog_meta, enabled=True)
            changelog_run.__enter__()
            print(f"[orchestrator] Change log → {log_path}")
            try:
                import atexit

                atexit.register(changelog_run.__exit__, None, None, None)
            except Exception:
                pass
        except Exception:
            changelog_run = None

    root.mkdir(parents=True, exist_ok=True)

    # Visible validation-video store under ROOT (for human inspection + stable automation input paths).
    validation_videos_store_root = _ensure_validation_video_store(root)

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

    run_logger: _RunOutputLogger | None = None
    if bool(SAVE_RUN_LOG):
        run_logger = _RunOutputLogger(
            inference_root=inference_root,
            species_name=species_name,
            time_of_day=time_of_day,
            enabled=bool(SAVE_RUN_LOG),
            dry_run=bool(args.dry_run),
        )
        run_logger.install()
        global _RUN_OUTPUT_LOGGER
        _RUN_OUTPUT_LOGGER = run_logger
        try:
            import atexit

            atexit.register(run_logger.close)
        except Exception:
            pass
        if run_logger.path is not None:
            print(f"[orchestrator] Run log (temp) → {run_logger.path}")

    # Best-effort backfill: make legacy validation videos visible under ROOT/validation videos/<species>/...
    # (Hardlink preferred; copy fallback for cross-device.)
    if AUTO_SYNC_VALIDATION_VIDEOS_STORE:
        legacy_val_individual_root = data_root / "Integrated_prototype_validation_datasets" / "individual species folder"
        if legacy_val_individual_root.exists():
            for sp_dir in sorted(legacy_val_individual_root.iterdir(), key=lambda p: p.name):
                if not sp_dir.is_dir():
                    continue
                for vp in sp_dir.glob("**/videos/*"):
                    if not vp.is_file() or vp.suffix.lower() not in VIDEO_EXTS:
                        continue
                    dst = validation_videos_store_root / sp_dir.name / vp.name
                    if dst.exists():
                        continue
                    if bool(args.dry_run):
                        print(f"[dry-run] Would backfill validation video -> {dst}  (src={vp})")
                        continue
                    try:
                        _link_or_copy(vp, dst, mode=str(VALIDATION_VIDEOS_STORE_MODE))
                    except Exception:
                        # Best-effort only.
                        pass

    # -------------------------------------------------------------------------
    # Stage 1: Ingest (extract patches + version datasets)
    # -------------------------------------------------------------------------

    # Discover and split observed (video,csv) pairs.
    pairs_all = stage1_ingestor.discover_observed_pairs(observed_dir)

    # Skip videos that were already ingested (centralized ingestion metadata lives in the change log).
    already_ingested_video_names: set[str] = set()
    try:
        ing_idx = build_ingestion_index(log_path)
        already_ingested_video_names = set(ing_idx.get(species_name, {}).keys())
    except Exception:
        already_ingested_video_names = set()

    skipped_existing = [p for p in pairs_all if p.video_name in already_ingested_video_names]
    pairs = [p for p in pairs_all if p.video_name not in already_ingested_video_names]
    if skipped_existing:
        preview = ", ".join([p.video_name for p in skipped_existing[:10]])
        more = "" if len(skipped_existing) <= 10 else f" (+{len(skipped_existing) - 10} more)"
        print(f"[orchestrator] NOTE: skipping already-ingested videos: {preview}{more}")
    try:
        changelog_meta["raw_pairs_total"] = int(len(pairs_all))
        changelog_meta["raw_pairs_to_ingest"] = int(len(pairs))
        changelog_meta["raw_pairs_skipped_existing"] = int(len(skipped_existing))
        if skipped_existing:
            changelog_meta["skipped_existing_video_names"] = [p.video_name for p in skipped_existing[:200]]
            changelog_meta["skipped_existing_video_names_truncated"] = bool(len(skipped_existing) > 200)
    except Exception:
        pass

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
        f"total={len(pairs_all)} new={len(pairs)} skipped_existing={len(skipped_existing)} "
        f"train={len(train_pairs)} val={len(val_pairs)} train_fraction={float(args.train_pair_fraction):.3f}",
    )

    # Count rows for run naming + train decision.
    train_firefly_rows_total = sum(len(_read_annotator_csv(p.firefly_csv)) for p in train_pairs)
    train_background_rows_total = sum(
        len(_read_annotator_csv(p.background_csv)) for p in train_pairs if p.background_csv is not None
    )
    val_firefly_rows_total = sum(len(_read_annotator_csv(p.firefly_csv)) for p in val_pairs)

    has_train_rows = bool(train_firefly_rows_total or train_background_rows_total)

    # Sync this run's held-out validation videos into the visible store under ROOT.
    if AUTO_SYNC_VALIDATION_VIDEOS_STORE and val_pairs:
        _sync_validation_videos_for_pairs(
            pairs=val_pairs,
            species_name=species_name,
            store_root=validation_videos_store_root,
            mode=str(VALIDATION_VIDEOS_STORE_MODE),
            dry_run=bool(args.dry_run),
        )

    skip_ingest = bool(getattr(args, "skip_ingest", False))
    skip_train = bool(getattr(args, "skip_train", False))
    skip_test = bool(getattr(args, "skip_test", False))
    if skip_ingest:
        print("[orchestrator] Skipping Stage 1 ingestion (--skip-ingest).")
    if skip_train:
        print("[orchestrator] Skipping Stage 2 training (--skip-train).")
    if skip_test:
        print("[orchestrator] Skipping Stage 3 testing (--skip-test).")

    # Ingestion metadata for downstream incremental runs (stored in the centralized change log).
    if not skip_ingest:
        try:
            ingested_pairs_meta: List[Dict[str, Any]] = []
            for p in train_pairs:
                ingested_pairs_meta.append(
                    {
                        "species_token": str(species_name),
                        "video_name": str(p.video_name),
                        "video_path": str(p.video_path),
                        "firefly_csv": str(p.firefly_csv),
                        "background_csv": str(p.background_csv) if p.background_csv else None,
                        "split": "train",
                    }
                )
            for p in val_pairs:
                ingested_pairs_meta.append(
                    {
                        "species_token": str(species_name),
                        "video_name": str(p.video_name),
                        "video_path": str(p.video_path),
                        "firefly_csv": str(p.firefly_csv),
                        "background_csv": str(p.background_csv) if p.background_csv else None,
                        "split": "validation",
                    }
                )
            changelog_meta["ingested_pairs"] = ingested_pairs_meta
        except Exception:
            pass

    zoo = _ensure_model_zoo_scaffold(model_root)
    results_history_dir = zoo["results_history_dir"]
    legacy_results_history_file = zoo["legacy_results_history_file"]
    video_registry_dir = zoo["video_registry_dir"]
    legacy_video_registry_file = zoo["legacy_video_registry_file"]

    registry, registry_prev_version, _registry_prev_path = _load_latest_video_registry(
        registry_dir=video_registry_dir, legacy_path=legacy_video_registry_file
    )
    for p in pairs:
        registry[p.video_name] = str(p.video_path)
    # Important: Do NOT mutate registry metadata in-place. We'll write a new snapshot at the end of the run.

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

    # Optional: GLOBAL eval sweep uses the *combined* validation annotations.csv to evaluate
    # the global model on all validation videos accumulated so far across all species.
    global_eval_items: List[Dict[str, Any]] = []
    global_eval_source: Dict[str, Any] = {}
    global_eval_error: str | None = None
    if EVAL_GLOBAL_MODEL and EVAL_GLOBAL_MODEL_ON_ALL_VALIDATION_VIDEOS and (not skip_test):
        combined_csv = (validation_ver / "annotations.csv") if validation_ver is not None else None
        if combined_csv is None or not combined_csv.exists():
            global_eval_error = "combined_validation_annotations_not_found"
        else:
            rows = _read_validation_combined_csv(combined_csv)
            grouped = _group_validation_rows_by_species_and_video(rows)

            individual_validation_root = (
                data_root / "Integrated_prototype_validation_datasets" / "individual species folder"
            )
            extra_dirs: List[Path] = []
            # Common-case: current observed dir often contains validation videos for this run.
            extra_dirs.append(observed_dir)
            for d in VALIDATION_VIDEO_SEARCH_DIRS:
                try:
                    extra_dirs.append(Path(d).expanduser().resolve())
                except Exception:
                    continue

            missing: List[Dict[str, Any]] = []
            skipped_route: List[Dict[str, Any]] = []
            used_sources: Counter[str] = Counter()

            for (sp, vn), gt_rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                vp, src = _resolve_validation_video_path(
                    video_name=vn,
                    species_name=sp,
                    store_root=validation_videos_store_root,
                    individual_validation_root=individual_validation_root,
                    registry=registry,
                    extra_search_dirs=extra_dirs,
                )
                if vp is None or (not vp.exists()):
                    missing.append({"species_name": sp, "video_name": vn, "reason": src})
                    continue

                # Filter by route so we don't accidentally evaluate with the wrong time-of-day model.
                try:
                    route = _route_for_video(
                        vp, thr=float(GATEWAY_BRIGHTNESS_THRESHOLD), frames=int(GATEWAY_BRIGHTNESS_NUM_FRAMES)
                    )
                except Exception as e:
                    missing.append({"species_name": sp, "video_name": vn, "reason": f"route_failed: {e}"})
                    continue
                if expected_route == "day" and route != "day":
                    skipped_route.append({"species_name": sp, "video_name": vn, "routed": route, "expected": "day"})
                    continue
                if expected_route == "night" and route != "night":
                    skipped_route.append(
                        {"species_name": sp, "video_name": vn, "routed": route, "expected": "night"}
                    )
                    continue

                used_sources[str(src)] += 1
                registry[vn] = str(vp)

                # Keep the store populated even when videos come from legacy validation-dataset dirs.
                if AUTO_SYNC_VALIDATION_VIDEOS_STORE:
                    dst_dir = validation_videos_store_root / sp
                    dst = dst_dir / f"{vn}{vp.suffix.lower()}"
                    if not dst.exists():
                        if bool(args.dry_run):
                            print(f"[dry-run] Would backfill validation video store -> {dst}  (src={vp})")
                        else:
                            try:
                                _link_or_copy(vp, dst, mode=str(VALIDATION_VIDEOS_STORE_MODE))
                            except Exception:
                                pass

                global_eval_items.append(
                    {
                        "species_name": sp,
                        "video_name": vn,
                        "video_key": f"{sp}__{vn}",
                        "video_path": vp,
                        "gt_rows": gt_rows,
                        "video_source": src,
                        "route": route,
                    }
                )

            # Do not persist mid-run; we write a new registry snapshot at the end.

            global_eval_source = {
                "kind": "combined_validation_sweep",
                "combined_version": validation_ver.name if validation_ver else None,
                "combined_csv": str(combined_csv),
                "expected_route": expected_route,
                "time_of_day": time_of_day,
                "n_groups_total": len(grouped),
                "n_videos_selected": len(global_eval_items),
                "n_missing": len(missing),
                "n_skipped_route": len(skipped_route),
                "video_sources_used": dict(used_sources),
                "missing_videos": missing[:50],  # cap to keep run_record reasonable
                "skipped_route": skipped_route[:50],
            }
            if not global_eval_items:
                global_eval_error = "no_global_eval_videos_selected"

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
    try:
        changelog_meta["run_id"] = str(run_id)
    except Exception:
        pass

    # Ensure the run dir exists early, and move the run log into it now that run_id is known.
    run_dir = inference_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if run_logger is not None:
        run_logger.move_into_run_dir(run_dir)
        if run_logger.path is not None:
            print(f"[orchestrator] Run log → {run_logger.path}")

    # Write/update training-video metadata BEFORE running any heavy eval work, so we keep provenance
    # even if evaluation crashes mid-run.
    training_videos_snapshot_path = _update_training_videos_history(
        root=root,
        data_root=data_root,
        run_id=run_id,
        species_name=species_name,
        observed_dir=observed_dir,
        time_of_day=time_of_day,
        expected_route=expected_route,
        train_pair_fraction=float(args.train_pair_fraction),
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        integrated_dataset_version=integrated_ver,
        single_species_dataset_version=single_ver,
        did_train=bool(do_train),
        dry_run=bool(args.dry_run),
    )
    if training_videos_snapshot_path is not None:
        print(f"[orchestrator] Training videos metadata → {training_videos_snapshot_path}")

    results: List[Dict[str, Any]] = []

    def _eval_one_model(model_key: str, *, day_patch_model: Path | None, night_cnn_model: Path | None) -> None:
        # GLOBAL model can optionally evaluate on the full, accumulated validation set
        # across all species (from the combined annotations.csv).
        if model_key == "global_model" and global_eval_items:
            items: List[Tuple[str, Path, List[Dict[str, int]], Dict[str, Any] | None]] = [
                (str(it["video_key"]), Path(it["video_path"]), list(it["gt_rows"]), dict(it)) for it in global_eval_items
            ]
        else:
            items = [(vn, vp, gt, None) for (vn, vp, gt) in eval_items]

        pbar = (
            tqdm(items, desc=f"[stage3] {model_key}", unit="video", dynamic_ncols=True)
            if tqdm is not None and items
            else None
        )
        it = pbar if pbar is not None else items
        for video_key, vp, gt_rows, meta in it:
            out_root = inference_root / run_id / model_key / video_key
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
                    pbar.set_postfix(video=str(video_key), route=str(route), exit=int(e.returncode), dt_s=f"{dt:.1f}")
                results.append(
                    {
                        "model_key": model_key,
                        "video_name": video_key,
                        "source_video_name": meta.get("video_name") if isinstance(meta, dict) else None,
                        "species_name": meta.get("species_name") if isinstance(meta, dict) else None,
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
                        video=str(video_key),
                        route=str(route),
                        tp=int(best.get("tp") or 0),
                        fp=int(best.get("fp") or 0),
                        fn=int(best.get("fn") or 0),
                        f1=float(best.get("f1") or 0.0),
                        dt_s=f"{dt:.1f}",
                    )
                else:
                    pbar.set_postfix(video=str(video_key), route=str(route), dt_s=f"{dt:.1f}")

            results.append(
                {
                    "model_key": model_key,
                    "video_name": video_key,
                    "source_video_name": meta.get("video_name") if isinstance(meta, dict) else None,
                    "species_name": meta.get("species_name") if isinstance(meta, dict) else None,
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
    combined_thresholds_px = [10.0]
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
    combined_txt_path = run_dir / "combined_results__thr10.txt"
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
            "global_model_eval_source": global_eval_source,
            "n_global_model_eval_videos": len(global_eval_items),
            "global_model_eval_error": global_eval_error,
            "validation_videos_store_root": str(validation_videos_store_root),
            "validation_videos_store_mode": str(VALIDATION_VIDEOS_STORE_MODE),
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
        "training_videos_metadata": {
            "root": str(root / TRAINING_VIDEOS_DIRNAME),
            "snapshot_path": str(training_videos_snapshot_path) if training_videos_snapshot_path else None,
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

    # Model-zoo metadata: write new snapshots (append-only behavior).
    registry_snapshot_path = _write_video_registry_snapshot(
        registry_dir=video_registry_dir,
        registry=registry,
        prev_version=int(registry_prev_version),
        dry_run=bool(args.dry_run),
    )
    if registry_snapshot_path is not None:
        print(f"[orchestrator] Video registry snapshot → {registry_snapshot_path}")

    history_snapshot_path = _append_results_history_snapshot(
        history_dir=results_history_dir,
        legacy_path=legacy_results_history_file,
        record=record,
        dry_run=bool(args.dry_run),
    )
    if history_snapshot_path is not None:
        print(f"[orchestrator] Results history snapshot → {history_snapshot_path}")

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

    if run_logger is not None:
        try:
            run_logger.close()
        finally:
            _RUN_OUTPUT_LOGGER = None

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
