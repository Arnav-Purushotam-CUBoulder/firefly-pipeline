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

# -----------------------------------------------------------------------------
# User-configurable globals (defaults; CLI args can override)
# -----------------------------------------------------------------------------

# Optional single root folder. If set (or passed via --root), the orchestrator will
# create/use:
#   <ROOT>/patch training datasets and pipeline validation data
#   <ROOT>/model zoo
#   <ROOT>/inference outputs
ROOT_PATH: str | Path = "/Volumes/DL Project SSD/integrated prototype data"

# Folder containing *many* observed videos + their annotator CSVs (and potentially other files).
# The orchestrator will:
# - match annotation CSVs to .mp4 videos by filename
# - split matched video/csv pairs: half → training ingestion, half → validation/testing
OBSERVED_DATA_DIR: str | Path = ""

# Species name for this run. This is the source of truth; the orchestrator does not
# infer species from CSV filenames.
SPECIES_NAME: str = ""

# Global video type for this run (used for dataset routing + model-zoo selection).
# Allowed values: "day" | "night"
TYPE_OF_VIDEO: str = "day"

# Search dirs for locating existing validation videos by stem (used when no new held-out validation rows).
VALIDATION_VIDEO_SEARCH_DIRS: List[str | Path] = []

# Species scaler config overrides (passed into the scaler module)
AUTO_LOAD_SIBLING_CLASS_CSV: bool = True

# Training config (ResNet patch/CNN classifier)
TRAIN_EPOCHS: int = 50
TRAIN_BATCH_SIZE: int = 128
TRAIN_LR: float = 3e-4
TRAIN_NUM_WORKERS: int = 2
TRAIN_RESNET: str = "resnet18"
TRAIN_SEED: int = 1337

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

# Set True to do everything except actually call species scaler / training / gateway.
DRY_RUN: bool = False

# -----------------------------------------------------------------------------
# Legacy baseline evaluation (optional)
# -----------------------------------------------------------------------------

# If True, also run legacy/lab baseline detectors on the same eval videos and score
# them using the same point-distance validation logic (Stage5 validator).
RUN_BASELINE_EVAL: bool = True

# If True, only run baselines on videos routed to the night pipeline.
BASELINES_ONLY_FOR_NIGHT: bool = True

# Baseline validator settings (match day pipeline defaults)
BASELINE_DIST_THRESHOLDS_PX: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
BASELINE_VALIDATE_CROP_W: int = 10
BASELINE_VALIDATE_CROP_H: int = 10
BASELINE_MAX_FRAMES: int | None = None

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

GATEWAY_REL_PATH = Path("test1/integrated pipeline/gateway.py")
SPECIES_SCALER_REL_DIR = Path("test1/species scaler")
TRAINING_SCRIPT_REL_PATH = Path("test1/tools/model training tools/training resnet18 model.py")

DAY_OUTPUT_SUBDIR = "day_pipeline_v3"
NIGHT_OUTPUT_SUBDIR = "night_time_pipeline"

DAY_PIPELINE_DIR_REL_PATH = Path("test1/day time pipeline v3 (yolo + patch classifier ensemble)")
LAB_BASELINE_SCRIPT_REL_PATH = Path("test1/tools/legacy_baselines/nolan_mp4_to_predcsv.py")
RAPHAEL_BASELINE_SCRIPT_REL_PATH = Path("test1/tools/legacy_baselines/raphael_oorb_detect_and_gauss.py")
BASELINE_OUTPUT_SUBDIR = "baselines"
BASELINE_MODELS_SUBDIR = "baseline models"


@dataclass(frozen=True)
class RunIdentity:
    dataset_name: str
    species_name: str
    time_of_day: str  # "day_time" | "night_time"


@dataclass(frozen=True)
class ObservedPair:
    video_name: str
    video_path: Path
    firefly_csv: Path
    background_csv: Path | None


@dataclass(frozen=True)
class StagedPair:
    pair: ObservedPair
    staged_firefly_csv: Path
    staged_background_csv: Path | None


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


def _is_annotator_csv(csv_path: Path) -> bool:
    try:
        with csv_path.open(newline="") as fh:
            r = csv.DictReader(fh)
            fieldnames = [str(x or "").strip().lower() for x in (r.fieldnames or [])]
    except Exception:
        return False
    if not fieldnames:
        return False
    cols = set(fieldnames)
    if not {"x", "y", "w", "h"}.issubset(cols):
        return False
    return bool({"t", "frame", "time"} & cols)


def _parse_label_from_csv_name(csv_path: Path, *, video_key: str) -> str:
    """
    Parse label from an annotator CSV filename.

    Expected: <video_stem>_..._[firefly|background].csv
    - label defaults to "firefly" if omitted
    - species is intentionally NOT parsed here (SPECIES_NAME controls this run)
    """
    vk = (video_key or "").strip().lower()

    remainder = ""
    raw_key = csv_path.stem.strip().lower()
    if vk and (raw_key == vk or raw_key.startswith(vk + "_")):
        remainder = raw_key[len(vk) :].lstrip("_")
    else:
        stem = csv_path.stem.strip()
        stem = stem.replace("-", "_").replace("(", "_").replace(")", "_")
        stem = re.sub(r"_+", "_", stem).strip("_")
        key = stem.lower()

        vk_norm = vk.replace("-", "_").replace("(", "_").replace(")", "_")
        vk_norm = re.sub(r"_+", "_", vk_norm).strip("_")

        if vk_norm and (key == vk_norm or key.startswith(vk_norm + "_")):
            remainder = key[len(vk_norm) :].lstrip("_")
        else:
            # Fall back to parsing by suffix only.
            remainder = key

    parts = [p for p in remainder.split("_") if p]
    if parts and parts[-1].lower() in {"firefly", "background"}:
        return parts[-1].lower()
    return "firefly"


def _discover_observed_pairs(observed_dir: Path) -> List[ObservedPair]:
    observed_dir = Path(observed_dir)
    if not observed_dir.exists():
        raise FileNotFoundError(observed_dir)
    if not observed_dir.is_dir():
        raise NotADirectoryError(observed_dir)

    videos = sorted([p for p in observed_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"])
    if not videos:
        raise SystemExit(f"No .mp4 videos found under: {observed_dir}")

    videos_by_stem: Dict[str, Path] = {}
    for v in videos:
        key = v.stem.strip().lower()
        if not key:
            continue
        if key in videos_by_stem and str(videos_by_stem[key].resolve()) != str(v.resolve()):
            raise SystemExit(f"Duplicate video stem {v.stem!r} found in: {videos_by_stem[key]} and {v}")
        videos_by_stem[key] = v

    stem_keys = sorted(videos_by_stem.keys(), key=len, reverse=True)

    csv_candidates = sorted([p for p in observed_dir.rglob("*.csv") if p.is_file() and _is_annotator_csv(p)])
    if not csv_candidates:
        raise SystemExit(f"No annotator CSVs found under: {observed_dir} (need x,y,w,h and t/frame columns).")

    grouped: Dict[str, Dict[str, Any]] = {}
    unmatched: List[Path] = []
    for c in csv_candidates:
        c_stem = c.stem.strip()
        c_key = c_stem.lower()
        match: str | None = None
        for vk in stem_keys:
            if c_key == vk or c_key.startswith(vk + "_"):
                match = vk
                break
        if match is None:
            unmatched.append(c)
            continue

        vp = videos_by_stem[match]
        video_name = _safe_name(vp.stem)
        try:
            label = _parse_label_from_csv_name(c, video_key=match)
        except Exception as e:
            raise SystemExit(f"Failed parsing CSV identity for {c}: {e}") from e

        g = grouped.setdefault(
            match,
            {
                "video_path": vp,
                "video_name": video_name,
                "firefly_csv": None,
                "background_csv": None,
            },
        )

        if label == "firefly":
            if g["firefly_csv"] is not None and str(Path(g["firefly_csv"]).resolve()) != str(c.resolve()):
                raise SystemExit(f"Multiple firefly CSVs matched to video {vp.name}: {g['firefly_csv']} and {c}")
            g["firefly_csv"] = c
        elif label == "background":
            if g["background_csv"] is not None and str(Path(g["background_csv"]).resolve()) != str(c.resolve()):
                raise SystemExit(f"Multiple background CSVs matched to video {vp.name}: {g['background_csv']} and {c}")
            g["background_csv"] = c

    if unmatched:
        preview = "\n".join([f"  - {p}" for p in unmatched[:20]])
        more = "" if len(unmatched) <= 20 else f"\n  ... and {len(unmatched) - 20} more"
        raise SystemExit(
            "Found annotator CSVs that did not match any .mp4 by filename prefix:\n"
            f"{preview}{more}\n"
            "Ensure each CSV name starts with the corresponding video stem."
        )

    pairs: List[ObservedPair] = []
    for _, g in sorted(grouped.items(), key=lambda kv: str(kv[1].get("video_name") or "")):
        if g.get("firefly_csv") is None:
            vp = g["video_path"]
            raise SystemExit(f"Missing firefly CSV for video: {vp} (need ..._firefly.csv or unlabeled CSV).")
        pairs.append(
            ObservedPair(
                video_name=str(g["video_name"]),
                video_path=Path(g["video_path"]),
                firefly_csv=Path(g["firefly_csv"]),
                background_csv=Path(g["background_csv"]) if g.get("background_csv") else None,
            )
        )

    if not pairs:
        raise SystemExit(f"No valid (video, firefly_csv) pairs found under: {observed_dir}")

    return pairs


def _split_pairs_train_vs_val(pairs: Sequence[ObservedPair]) -> Tuple[List[ObservedPair], List[ObservedPair]]:
    """
    Split matched (video,csv) pairs into TRAIN vs VALIDATION at the *video* level.

    Requirement:
    - half of pairs → training
    - half of pairs → validation/testing
    - if odd: validation gets the larger half
    """
    ordered = sorted(pairs, key=lambda p: (p.video_name, str(p.video_path)))
    n_train = int(len(ordered) // 2)
    return ordered[:n_train], ordered[n_train:]


def _stage_pairs_for_species_scaler(
    pairs: Sequence[ObservedPair],
    *,
    species_name: str,
    staging_dir: Path,
    time_of_day: str,
    dry_run: bool,
) -> List[StagedPair]:
    """
    Species-scaler parses identity from CSV filename, so we stage/copy CSVs into a
    canonical naming scheme:
      <video_name>_<species_name>_<day_time|night_time>_<firefly|background>.csv
    """
    staging_dir = Path(staging_dir)
    if not dry_run:
        staging_dir.mkdir(parents=True, exist_ok=True)

    staged: List[StagedPair] = []
    seen: set[str] = set()
    safe_species = _safe_name(species_name)
    if not safe_species:
        raise SystemExit(f"Invalid species_name for staging: {species_name!r}")
    for p in pairs:
        base = f"{_safe_name(p.video_name)}_{safe_species}_{_safe_name(time_of_day)}"
        dst_firefly = staging_dir / f"{base}_firefly.csv"
        if str(dst_firefly) in seen:
            raise SystemExit(f"Staging name collision (firefly): {dst_firefly}")
        seen.add(str(dst_firefly))

        if dry_run:
            print(f"[dry-run] Would stage CSV → {dst_firefly}  (src={p.firefly_csv})")
        else:
            dst_firefly.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p.firefly_csv, dst_firefly)

        dst_bg: Path | None = None
        if p.background_csv is not None:
            dst_bg = staging_dir / f"{base}_background.csv"
            if str(dst_bg) in seen:
                raise SystemExit(f"Staging name collision (background): {dst_bg}")
            seen.add(str(dst_bg))
            if dry_run:
                print(f"[dry-run] Would stage CSV → {dst_bg}  (src={p.background_csv})")
            else:
                dst_bg.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p.background_csv, dst_bg)

        staged.append(StagedPair(pair=p, staged_firefly_csv=dst_firefly, staged_background_csv=dst_bg))

    return staged


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


def _run_species_scaler(
    *,
    repo_root: Path,
    annotations_csv: Path,
    video_path: Path,
    data_root: Path,
    train_fraction: float,
    train_val_seed: int,
    auto_load_sibling: bool,
    dry_run: bool,
) -> None:
    scaler_dir = repo_root / SPECIES_SCALER_REL_DIR
    scaler_py = scaler_dir / "species_scaler.py"
    if not scaler_py.exists():
        raise FileNotFoundError(scaler_py)

    if dry_run:
        print(f"[dry-run] Would run species scaler on: {annotations_csv}")
        return

    code = "\n".join(
        [
            "from pathlib import Path",
            "import species_scaler as ss",
            f"ss.ANNOTATIONS_CSV_PATH = Path({repr(str(annotations_csv))})",
            f"ss.VIDEO_PATH = Path({repr(str(video_path))})",
            f"ss.DATA_ROOT = Path({repr(str(data_root))})",
            f"ss.TRAIN_FRACTION_OF_BATCH = {float(train_fraction)}",
            f"ss.TRAIN_VAL_SPLIT_SEED = {int(train_val_seed)}",
            f"ss.AUTO_LOAD_SIBLING_CLASS_CSV = {bool(auto_load_sibling)}",
            "ss.main()",
        ]
    )
    subprocess.run([sys.executable, "-c", code], cwd=str(scaler_dir), check=True)


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
    trainer = repo_root / TRAINING_SCRIPT_REL_PATH
    if not trainer.exists():
        raise FileNotFoundError(trainer)

    cmd = [
        sys.executable,
        str(trainer),
        "--data-dir",
        str(data_dir),
        "--best-model-path",
        str(out_model_path),
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--lr",
        str(float(lr)),
        "--num-workers",
        str(int(num_workers)),
        "--resnet",
        str(resnet),
        "--seed",
        str(int(seed)),
        "--no-gui",
        "--metrics-out",
        str(metrics_out_path),
    ]

    if dry_run:
        print("[dry-run] Would run training:", " ".join(cmd))
        return {}

    subprocess.run(cmd, cwd=str(repo_root), check=True)
    try:
        return json.loads(metrics_out_path.read_text())
    except Exception:
        return {}


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

    repo_root = Path(__file__).resolve().parents[2]
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

    # Discover and split observed (video,csv) pairs.
    pairs = _discover_observed_pairs(observed_dir)

    train_pairs, val_pairs = _split_pairs_train_vs_val(pairs)
    train_video_names = {p.video_name for p in train_pairs}
    val_video_names = {p.video_name for p in val_pairs}

    # Count rows for run naming + train decision.
    train_firefly_rows_total = sum(len(_read_annotator_csv(p.firefly_csv)) for p in train_pairs)
    train_background_rows_total = sum(
        len(_read_annotator_csv(p.background_csv)) for p in train_pairs if p.background_csv is not None
    )
    val_firefly_rows_total = sum(len(_read_annotator_csv(p.firefly_csv)) for p in val_pairs)

    has_train_rows = bool(train_firefly_rows_total or train_background_rows_total)

    # Stage CSVs into canonical names expected by species scaler.
    staging_dir = (
        data_root
        / "batch_exports"
        / "orchestrator_observed_dir_staging"
        / f"{_run_tag()}__{_safe_name(observed_dir.name)}"
    )
    staged_pairs = _stage_pairs_for_species_scaler(
        pairs,
        species_name=species_name,
        staging_dir=staging_dir,
        time_of_day=time_of_day,
        dry_run=bool(args.dry_run),
    )

    zoo = _ensure_model_zoo_scaffold(model_root)
    history_path = zoo["history"]
    registry_path = zoo["registry"]
    registry = _load_registry(registry_path)
    for p in pairs:
        registry[p.video_name] = str(p.video_path)
    _save_registry(registry_path, registry)

    staged_by_video: Dict[str, StagedPair] = {sp.pair.video_name: sp for sp in staged_pairs}

    # Run species scaler (ingestion) for training pairs (train_fraction=1.0) and validation pairs (train_fraction=0.0).
    for p in pairs:
        sp = staged_by_video[p.video_name]
        if p.video_name in train_video_names:
            _run_species_scaler(
                repo_root=repo_root,
                annotations_csv=sp.staged_firefly_csv,
                video_path=p.video_path,
                data_root=data_root,
                train_fraction=1.0,
                train_val_seed=1337,
                auto_load_sibling=bool(AUTO_LOAD_SIBLING_CLASS_CSV),
                dry_run=bool(args.dry_run),
            )
        elif p.video_name in val_video_names:
            _run_species_scaler(
                repo_root=repo_root,
                annotations_csv=sp.staged_firefly_csv,
                video_path=p.video_path,
                data_root=data_root,
                train_fraction=0.0,
                train_val_seed=1337,
                auto_load_sibling=False,
                dry_run=bool(args.dry_run),
            )

    # Resolve dataset roots produced by species scaler
    integrated_root = data_root / "Integrated_prototype_datasets" / "integrated pipeline datasets"
    single_root = data_root / "Integrated_prototype_datasets" / "single species datasets" / species_name
    validation_combined_root = data_root / "Integrated_prototype_validation_datasets" / "combined species folder"
    validation_species_root = (
        data_root
        / "Integrated_prototype_validation_datasets"
        / "individual species folder"
        / species_name
    )

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

    # Train models (if this batch added training rows)
    global_model_path: Path | None = None
    species_model_path: Path | None = None
    global_train_metrics: Dict[str, Any] | None = None
    species_train_metrics: Dict[str, Any] | None = None

    do_train = bool(TRAIN_MODELS_IF_TRAIN_ROWS_PRESENT) and has_train_rows

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

            species_summary = _species_summary_from_patch_locations(
                integrated_ver / dataset_time_dir / "patch_locations_train.csv"
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
                        "trained_species_source_csv": species_summary.get("source_csv"),
                        "trained_species_error": species_summary.get("error"),
                        "train_metrics": global_train_metrics,
                    },
                    indent=2,
                )
            )

        if EVAL_SINGLE_SPECIES_MODEL:
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
        for video_name, vp, gt_rows in eval_items:
            out_root = inference_root / run_id / model_key / video_name
            out_root.mkdir(parents=True, exist_ok=True)

            # Write GT for both pipeline roots (only one will be used depending on route)
            day_gt_dir = out_root / DAY_OUTPUT_SUBDIR / "ground truth"
            night_gt_dir = out_root / NIGHT_OUTPUT_SUBDIR / "ground truth"
            _write_gt_csv(day_gt_dir / f"gt_{vp.stem}.csv", gt_rows)
            _write_gt_csv(day_gt_dir / "gt.csv", gt_rows)
            _write_gt_csv(night_gt_dir / "gt.csv", gt_rows)

            t0 = time.time()
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
            dt = time.time() - t0

            # Parse validation metrics from whichever pipeline was chosen
            route = _route_for_video(vp, thr=float(GATEWAY_BRIGHTNESS_THRESHOLD), frames=int(GATEWAY_BRIGHTNESS_NUM_FRAMES))
            if route == "day":
                stage_dir = out_root / DAY_OUTPUT_SUBDIR / "stage5 validation" / vp.stem
            else:
                stage_dir = out_root / NIGHT_OUTPUT_SUBDIR / "stage9 validation" / vp.stem
            metrics = _parse_validation_metrics(stage_dir)

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

        for video_name, vp, gt_rows in eval_items:
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
                baseline_results.append(
                    {
                        "model_key": "baseline_lab_method",
                        "video_name": video_name,
                        "video_path": str(vp),
                        "route": route,
                        "output_root": str(out_root),
                        "duration_s": float(time.time() - t0),
                        "validation_metrics": metrics,
                    }
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
                baseline_results.append(
                    {
                        "model_key": "baseline_raphael_method",
                        "video_name": video_name,
                        "video_path": str(vp),
                        "route": route,
                        "output_root": str(out_root),
                        "duration_s": float(time.time() - t0),
                        "validation_metrics": metrics,
                    }
                )

    baseline_summary = _summarize_results(baseline_results) if baseline_results else {}

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
        "results": results,
        "results_summary": results_summary,
    }
    # Save a full run record alongside inference outputs for easy inspection.
    run_dir = inference_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
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
