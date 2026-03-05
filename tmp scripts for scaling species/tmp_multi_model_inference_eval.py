#!/usr/bin/env python3
from __future__ import annotations

"""
Temporary one-off automation (safe to delete after use).

Runs evaluation on the integrated gateway pipelines using:
  1) Global patch classifier (trained on all species)
  2) Leave-one-species-out models (evaluate each leaveout_* on its left-out species)
and also runs legacy baselines (lab + Raphael) per validation video.

Outputs:
  - Per-run pipeline outputs under: ACTUAL_OUTPUTS_ROOT/<RUN_ID>/
  - Consolidated CSV under: INFERENCE_ROOT/final_results.csv

This script is designed to be run locally (e.g. VSCode "Run" button).
Edit the CONFIG section below for your machine if needed.
"""

import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =========================
# CONFIG (EDIT THESE)
# =========================

# Trained model outputs root (from tools/train_multi_species_patch_classifiers.py)
MODEL_OUTPUTS_ROOT = Path("/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/outputs")

# Validation annotations (per species) live here:
VAL_INDIVIDUAL_ROOT = Path(
    "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/"
    "patch training datasets and pipeline validation data/"
    "Integrated_prototype_validation_datasets/individual species folder"
)

# Validation videos store (organized by species subfolders)
VAL_VIDEOS_ROOT = Path("/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/validation videos")

# Where to write all inference outputs (pipelines + baselines) for this run
INFERENCE_ROOT = Path("/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/inference output")
ACTUAL_OUTPUTS_ROOT = INFERENCE_ROOT / "actual outputs"

# Consolidated results CSV (single place to view everything)
FINAL_RESULTS_CSV = INFERENCE_ROOT / "final_results.csv"

# Route assignment for videos (brightness routing disabled in this script).
# Routing is derived from species/video names + optional explicit overrides below.
#
# Priority:
#   1) ROUTE_BY_VIDEO_NAME (exact video stem)
#   2) ROUTE_BY_SPECIES
#   3) route token hints found in species/video names
#   4) ROUTE_DEFAULT (unless REQUIRE_EXPLICIT_ROUTE=True)
ROUTE_BY_VIDEO_NAME: Dict[str, str] = {}
ROUTE_BY_SPECIES: Dict[str, str] = {
    "bicellonycha-wickershamorum": "night",
    "forresti": "night",
    "frontalis": "night",
    # Requested override: treat acuminatus validation video(s) as day-routed.
    "photinus-acuminatus": "day",
    "photinus-carolinus": "night",
    "photinus-knulli": "night",
    "photuris-bethaniensis": "night",
    "tremulans": "night",
}
ROUTE_NAME_HINT_DAY_TOKENS: Tuple[str, ...] = ("day", "daytime", "day_time")
ROUTE_NAME_HINT_NIGHT_TOKENS: Tuple[str, ...] = ("night", "nighttime", "night_time")
ROUTE_DEFAULT: str = "night"
REQUIRE_EXPLICIT_ROUTE: bool = True

GATEWAY_MAX_CONCURRENT = 1
FORCE_GATEWAY_TESTS = True

# Baselines
RUN_BASELINES = True
BASELINES_ONLY_FOR_NIGHT = True

# Lab baseline (no model file required)
RUN_LAB_BASELINE = True
LAB_BASELINE_THRESHOLD = 0.12
LAB_BASELINE_BLUR_SIGMA = 1.0
LAB_BASELINE_BKGR_WINDOW_SEC = 2.0

# Raphael baseline (requires TorchScript ffnet model)
RUN_RAPHAEL_BASELINE = True
RAPHAEL_MODEL_PATH = (
    "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/"
    "model zoo/Raphael's model/ffnet_best.pth"
)
RAPHAEL_BW_THR = 0.2
RAPHAEL_CLASSIFY_THR = 0.98
RAPHAEL_BKGR_WINDOW_SEC = 2.0
RAPHAEL_BLUR_SIGMA = 0.0
RAPHAEL_PATCH_SIZE_PX = 33
RAPHAEL_BATCH_SIZE = 1000
RAPHAEL_GAUSS_CROP_SIZE = 10
RAPHAEL_DEVICE = "auto"

# Stage5 validator settings for baselines (10px only to keep outputs small)
BASELINE_DIST_THRESHOLDS_PX: List[float] = [10.0]
BASELINE_VALIDATE_CROP_W = 10
BASELINE_VALIDATE_CROP_H = 10
BASELINE_MAX_FRAMES: int | None = None  # set to an int for quick smoke tests

# If True, only prints planned work and exits.
DRY_RUN = False

# =========================


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class EvalVideo:
    species_name: str
    video_name: str  # stem without extension (from annotations.csv)
    video_path: Path
    video_key: str  # short safe key used for folder names
    route: str  # "day" | "night"
    gt_rows: List[Dict[str, int]]  # x,y,t rows


@dataclass(frozen=True)
class ModelSpec:
    key: str  # e.g. global_all_species, leaveout_photinus-carolinus
    ckpt_path: Path


def _q(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _safe_name(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _short_key(species_name: str, video_name: str) -> str:
    sp = _safe_name(species_name)
    vn = _safe_name(video_name)
    h = hashlib.sha1(vn.encode("utf-8", errors="ignore")).hexdigest()[:10]
    # Keep folder names comfortably under per-segment limits.
    if len(vn) > 120:
        vn = vn[:120].rstrip("_")
    return f"{sp}__{vn}__{h}"


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


def _read_validation_annotations(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        if not fieldnames:
            return []
        return [{k: (row.get(k) or "") for k in fieldnames} for row in r]


def _group_gt_rows_by_video(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, int]]]:
    out: Dict[str, List[Dict[str, int]]] = {}
    for r in rows:
        vn = str(r.get("video_name") or "").strip()
        if vn.lower().endswith(".mp4"):
            vn = vn[: -len(".mp4")]
        if not vn:
            continue
        try:
            x = int(round(float(r.get("x") or 0)))
            y = int(round(float(r.get("y") or 0)))
            t = int(round(float(r.get("t") or 0)))
        except Exception:
            continue
        out.setdefault(vn, []).append({"x": x, "y": y, "t": t})
    return out


def _build_video_index(roots: Sequence[Path]) -> Dict[str, List[Path]]:
    """
    Map file stem -> [paths...] across one or more roots.
    """
    idx: Dict[str, List[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            idx.setdefault(p.stem, []).append(p)
    return idx


def _resolve_video_path(
    *,
    species_name: str,
    video_name: str,
    video_index: Dict[str, List[Path]],
) -> Optional[Path]:
    """
    Prefer the expected validation-videos store location if present; fall back to index.
    """
    expected_dir = VAL_VIDEOS_ROOT / str(species_name)
    if expected_dir.exists():
        for ext in sorted(VIDEO_EXTS):
            cand = expected_dir / f"{video_name}{ext}"
            if cand.exists():
                return cand
        # some files might include extra suffixes; try startswith search
        for p in sorted(expected_dir.glob(f"{video_name}*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                return p

    matches = video_index.get(video_name) or []
    if not matches:
        return None
    # Prefer an exact species-folder match if possible.
    for p in matches:
        try:
            if p.parent.name == str(species_name):
                return p
        except Exception:
            continue
    return matches[0]


_ROUTE_WARNED_DEFAULT_KEYS: set[str] = set()


def _normalize_route(value: str) -> str:
    v = (value or "").strip().lower()
    if v not in {"day", "night"}:
        raise ValueError(f"Invalid route '{value}'. Expected 'day' or 'night'.")
    return v


def _route_hint_from_names(*, species_name: str, video_name: str) -> Optional[str]:
    token_src = f"{species_name} {video_name}".lower()
    has_day = any(tok in token_src for tok in ROUTE_NAME_HINT_DAY_TOKENS)
    has_night = any(tok in token_src for tok in ROUTE_NAME_HINT_NIGHT_TOKENS)
    if has_day and has_night:
        raise ValueError(f"Conflicting day/night tokens for '{species_name}/{video_name}'.")
    if has_day:
        return "day"
    if has_night:
        return "night"
    return None


def _route_for_video(*, species_name: str, video_name: str) -> str:
    r_video = ROUTE_BY_VIDEO_NAME.get(video_name)
    if r_video is not None:
        return _normalize_route(r_video)

    r_species = ROUTE_BY_SPECIES.get(species_name)
    if r_species is not None:
        return _normalize_route(r_species)

    r_hint = _route_hint_from_names(species_name=species_name, video_name=video_name)
    if r_hint is not None:
        return r_hint

    if REQUIRE_EXPLICIT_ROUTE:
        raise ValueError(
            f"No explicit route for '{species_name}/{video_name}'. "
            "Set ROUTE_BY_VIDEO_NAME or ROUTE_BY_SPECIES, or add route tokens in names."
        )

    key = f"{species_name}/{video_name}"
    if key not in _ROUTE_WARNED_DEFAULT_KEYS:
        print(f"[discover] WARNING: route unresolved for {key}; defaulting to '{ROUTE_DEFAULT}'.")
        _ROUTE_WARNED_DEFAULT_KEYS.add(key)
    return _normalize_route(ROUTE_DEFAULT)


def _write_gt_csv(path: Path, rows: Sequence[Dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "t"])
        w.writeheader()
        for r in rows:
            w.writerow({"x": int(r["x"]), "y": int(r["y"]), "t": int(r["t"])})


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


def _discover_models() -> Dict[str, ModelSpec]:
    root = Path(MODEL_OUTPUTS_ROOT).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"MODEL_OUTPUTS_ROOT not found: {root}")

    models: Dict[str, ModelSpec] = {}
    for p in sorted([x for x in root.iterdir() if x.is_dir()], key=lambda x: x.name):
        key = p.name
        if key.startswith("_"):
            continue
        best = p / "checkpoints" / "best.pt"
        last = p / "checkpoints" / "last.pt"
        alt = p / "best.pt"
        if best.exists():
            ckpt = best
        elif last.exists():
            ckpt = last
        elif alt.exists():
            ckpt = alt
        else:
            continue
        models[key] = ModelSpec(key=key, ckpt_path=ckpt)
    if "global_all_species" not in models:
        raise SystemExit(
            f"Global model not found under {MODEL_OUTPUTS_ROOT}. Expected folder 'global_all_species/'."
        )
    return models


def _discover_validation_videos() -> List[EvalVideo]:
    ind_root = Path(VAL_INDIVIDUAL_ROOT).expanduser().resolve()
    if not ind_root.exists():
        raise SystemExit(f"VAL_INDIVIDUAL_ROOT not found: {ind_root}")

    # Build an index of all validation videos once (stem -> [paths...])
    print("[discover] indexing validation videos (this may take a moment)…")
    video_index = _build_video_index([Path(VAL_VIDEOS_ROOT).expanduser().resolve(), ind_root])

    out: List[EvalVideo] = []
    missing: List[Tuple[str, str]] = []

    for sp_dir in sorted([p for p in ind_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        species_name = sp_dir.name
        latest = _latest_version_dir(sp_dir)
        if latest is None:
            continue
        ann = latest / "annotations.csv"
        if not ann.exists():
            continue

        rows = _read_validation_annotations(ann)
        by_video = _group_gt_rows_by_video(rows)
        for video_name, gt_rows in sorted(by_video.items(), key=lambda kv: kv[0]):
            vp = _resolve_video_path(species_name=species_name, video_name=video_name, video_index=video_index)
            if vp is None or (not vp.exists()):
                missing.append((species_name, video_name))
                continue
            route = _route_for_video(species_name=species_name, video_name=video_name)
            out.append(
                EvalVideo(
                    species_name=species_name,
                    video_name=video_name,
                    video_path=vp,
                    video_key=_short_key(species_name, video_name),
                    route=route,
                    gt_rows=list(gt_rows),
                )
            )

    if missing:
        preview = ", ".join([f"{sp}/{vn}" for sp, vn in missing[:10]])
        print(f"[discover] WARNING: {len(missing)} video(s) missing (showing up to 10): {preview}")

    if not out:
        raise SystemExit("No validation videos discovered. Check VAL_INDIVIDUAL_ROOT / VAL_VIDEOS_ROOT.")
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    # If outputs already exist and look valid, reuse them (resume-friendly).
    day_root = out_root / "day_pipeline_v3"
    night_root = out_root / "night_time_pipeline"
    stage_day = day_root / "stage5 validation" / video.video_path.stem
    stage_night = night_root / "stage9 validation" / video.video_path.stem
    if stage_day.exists():
        m = _parse_validation_metrics(stage_day)
        if not m.get("error"):
            return "day", m
    if stage_night.exists():
        m = _parse_validation_metrics(stage_night)
        if not m.get("error"):
            return "night", m

    # Ensure GT exists for whichever pipeline gets routed.
    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        _write_gt_csv(day_root / "ground truth" / f"gt_{video.video_path.stem}.csv", video.gt_rows)
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
    ]
    if bool(FORCE_GATEWAY_TESTS):
        cmd.append("--force-tests")

    _run_subprocess_logged(cmd, cwd=_repo_root(), log_path=log_path, dry_run=dry_run)

    # Determine which validation folder exists (don't rely purely on precomputed route).
    if stage_day.exists():
        return "day", _parse_validation_metrics(stage_day)
    if stage_night.exists():
        return "night", _parse_validation_metrics(stage_night)

    # Fallback to expected route to produce a helpful error.
    exp = str(video.route)
    stage = stage_day if exp == "day" else stage_night
    return exp, {"error": f"no validation outputs found (expected under: {stage})"}


def _run_stage5_validator(
    *,
    orig_video_path: Path,
    pred_csv_path: Path,
    gt_csv_path: Path,
    out_dir: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    day_dir = _day_pipeline_dir()
    stage5_py = day_dir / "stage5_validate.py"
    if not stage5_py.exists():
        raise FileNotFoundError(stage5_py)

    # Prevent Stage5 validator from auto-searching for FN-scoring weights.
    no_weights = out_dir / "__no_fn_scoring_weights__.pt"
    max_frames_code = str(int(BASELINE_MAX_FRAMES)) if BASELINE_MAX_FRAMES is not None else "None"

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


def _run_baselines_for_video(
    *,
    run_dir: Path,
    video: EvalVideo,
    logs_dir: Path,
    dry_run: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        "lab": {"metrics":..., "out_root":...},
        "raphael": {"metrics":..., "out_root":...},
      }
    """
    out: Dict[str, Dict[str, Any]] = {}
    scripts = _baseline_scripts()

    if BASELINES_ONLY_FOR_NIGHT and str(video.route) != "night":
        return out

    # Lab baseline
    if RUN_LAB_BASELINE:
        out_root = run_dir / "baselines" / "lab_method" / f"{video.route}_videos" / video.video_key
        gt_csv = out_root / "ground truth" / "gt.csv"
        pred_csv = out_root / "predictions.csv"
        stage5_dir = out_root / "stage5 validation" / video.video_path.stem
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
                        + (["--max-frames", str(int(BASELINE_MAX_FRAMES))] if BASELINE_MAX_FRAMES is not None else []),
                        cwd=_repo_root(),
                        log_path=logs_dir / f"baseline_lab__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    _run_stage5_validator(
                        orig_video_path=video.video_path,
                        pred_csv_path=pred_csv,
                        gt_csv_path=gt_csv,
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
        out_root = run_dir / "baselines" / "raphael_method" / f"{video.route}_videos" / video.video_key
        gt_csv = out_root / "ground truth" / "gt.csv"
        pred_csv = out_root / "predictions.csv"
        raw_csv = out_root / "raw.csv"
        gauss_csv = out_root / "gauss.csv"
        stage5_dir = out_root / "stage5 validation" / video.video_path.stem
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
                        + (["--max-frames", str(int(BASELINE_MAX_FRAMES))] if BASELINE_MAX_FRAMES is not None else []),
                        cwd=_repo_root(),
                        log_path=logs_dir / f"baseline_raphael__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    _run_stage5_validator(
                        orig_video_path=video.video_path,
                        pred_csv_path=pred_csv,
                        gt_csv_path=gt_csv,
                        out_dir=stage5_dir,
                        log_path=logs_dir / f"baseline_raphael_stage5__{video.video_key}.log",
                        dry_run=dry_run,
                    )
                    metrics = _parse_validation_metrics(stage5_dir)
                except Exception as e:
                    metrics = {"error": f"raphael baseline failed: {e}"}

        out["raphael"] = {"metrics": metrics, "out_root": str(out_root)}

    return out


FINAL_RESULTS_FIELDNAMES = [
    "val_video_name",
    "model_used",
    "results",
    "inference_output_path",
    "run_id",
    "raphael_results",
    "raphael_output_path",
    "lab_results",
    "lab_output_path",
]


def _write_final_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_RESULTS_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "") for k in FINAL_RESULTS_FIELDNAMES})


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


def main() -> int:
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"tmp_infer_eval_{run_ts}"

    ACTUAL_OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = ACTUAL_OUTPUTS_ROOT / run_id
    logs_dir = run_dir / "logs"

    print(f"[run] run_id={run_id}")
    print(f"[run] models_root={MODEL_OUTPUTS_ROOT}")
    print(f"[run] val_individual_root={VAL_INDIVIDUAL_ROOT}")
    print(f"[run] val_videos_root={VAL_VIDEOS_ROOT}")
    print(f"[run] outputs={run_dir}")
    print(f"[run] final_csv={FINAL_RESULTS_CSV}")

    models = _discover_models()
    global_model = models["global_all_species"]

    videos = _discover_validation_videos()
    videos = sorted(videos, key=lambda v: (v.species_name, v.video_name))

    print(f"[discover] {len(models)} model(s) discovered (incl. global + leaveout_*).")
    print(f"[discover] {len(videos)} validation video(s) discovered.")

    # Run baselines once per video (cached and reused for all model rows)
    baseline_by_video_key: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if RUN_BASELINES:
        print(f"[baselines] enabled={RUN_BASELINES} only_for_night={BASELINES_ONLY_FOR_NIGHT}")
        for i, vid in enumerate(videos, start=1):
            print(f"[baselines] {i}/{len(videos)} {vid.species_name} :: {vid.video_path.name} (route={vid.route})")
            if DRY_RUN:
                continue
            baseline_by_video_key[vid.video_key] = _run_baselines_for_video(
                run_dir=run_dir,
                video=vid,
                logs_dir=logs_dir,
                dry_run=bool(DRY_RUN),
            )

    rows_out: List[Dict[str, str]] = []

    if not DRY_RUN:
        # Crash safety: initialize/truncate once, then append+fsync after each completed row.
        _init_final_csv(FINAL_RESULTS_CSV)

    def _record_row(row: Dict[str, str]) -> None:
        rows_out.append(row)
        if not DRY_RUN:
            _append_final_csv_row(FINAL_RESULTS_CSV, row)

    def _baseline_fields(video_key: str) -> Tuple[str, str, str, str]:
        base = baseline_by_video_key.get(video_key) or {}
        raphael = base.get("raphael") or {}
        lab = base.get("lab") or {}
        raphael_metrics = _metrics_str(raphael.get("metrics")) if raphael else ""
        raphael_path = str(raphael.get("out_root") or "") if raphael else ""
        lab_metrics = _metrics_str(lab.get("metrics")) if lab else ""
        lab_path = str(lab.get("out_root") or "") if lab else ""
        return raphael_metrics, raphael_path, lab_metrics, lab_path

    videos_by_route: Dict[str, List[EvalVideo]] = {"day": [], "night": []}
    for v in videos:
        videos_by_route.setdefault(v.route, []).append(v)

    # 1) Global model on ALL videos (separated by route)
    print(f"[eval] global model on all videos: {global_model.ckpt_path}")
    for route_name in ("day", "night"):
        route_videos = videos_by_route.get(route_name) or []
        if not route_videos:
            continue
        print(f"[eval][global] route={route_name} n_videos={len(route_videos)}")
        for i, vid in enumerate(route_videos, start=1):
            out_root = run_dir / "pipelines" / f"{route_name}_videos" / global_model.key / vid.video_key
            log_path = logs_dir / f"gateway__{global_model.key}__{vid.video_key}.log"
            print(f"[eval][global][{route_name}] {i}/{len(route_videos)} {vid.species_name} :: {vid.video_path.name}")

            if DRY_RUN:
                route_used = vid.route
                metrics = {"error": "dry_run"}
            else:
                try:
                    route_used, metrics = _run_gateway_for_video(
                        out_root=out_root,
                        video=vid,
                        model_path=global_model.ckpt_path,
                        log_path=log_path,
                        dry_run=bool(DRY_RUN),
                    )
                except Exception as e:
                    route_used, metrics = vid.route, {"error": str(e)}

            raphael_metrics, raphael_path, lab_metrics, lab_path = _baseline_fields(vid.video_key)
            _record_row(
                {
                    "val_video_name": vid.video_name,
                    "model_used": global_model.key,
                    "results": _metrics_str(metrics),
                    "inference_output_path": str(out_root),
                    "run_id": run_id,
                    "raphael_results": raphael_metrics,
                    "raphael_output_path": raphael_path,
                    "lab_results": lab_metrics,
                    "lab_output_path": lab_path,
                }
            )

    # 2) Leaveout models: evaluate each leaveout_* on its left-out species videos
    leaveout_models = [m for k, m in models.items() if k.startswith("leaveout_")]
    leaveout_models.sort(key=lambda m: m.key)

    by_species: Dict[str, List[EvalVideo]] = {}
    for v in videos:
        by_species.setdefault(v.species_name, []).append(v)

    # Combined per-species summary rows for the GLOBAL model (route-separated)
    for species_name, sel in sorted(by_species.items(), key=lambda kv: kv[0]):
        for route_name in ("day", "night"):
            sel_route = [v for v in sel if v.route == route_name]
            if not sel_route:
                continue
            tp = fp = fn = 0
            names = {v.video_name for v in sel_route}
            for r in rows_out:
                if r.get("model_used") != global_model.key:
                    continue
                if r.get("val_video_name") not in names:
                    continue
                if f"/pipelines/{route_name}_videos/" not in str(r.get("inference_output_path") or ""):
                    continue
                ms = str(r.get("results") or "")
                m_tp = re.search(r"\btp=(\d+)\b", ms)
                m_fp = re.search(r"\bfp=(\d+)\b", ms)
                m_fn = re.search(r"\bfn=(\d+)\b", ms)
                if not (m_tp and m_fp and m_fn):
                    continue
                tp += int(m_tp.group(1))
                fp += int(m_fp.group(1))
                fn += int(m_fn.group(1))

            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            acc = _accuracy(tp, fp, fn)
            combined_str = f"tp={tp} fp={fp} fn={fn} precision={prec:.6f} accuracy={acc:.6f} f1={f1:.6f}"
            _record_row(
                {
                    "val_video_name": f"{species_name}__{route_name}__COMBINED",
                    "model_used": global_model.key,
                    "results": combined_str,
                    "inference_output_path": str(run_dir / "pipelines" / f"{route_name}_videos" / global_model.key),
                    "run_id": run_id,
                    "raphael_results": "",
                    "raphael_output_path": "",
                    "lab_results": "",
                    "lab_output_path": "",
                }
            )

    for m in leaveout_models:
        left_out_species = m.key.replace("leaveout_", "", 1)
        sel = by_species.get(left_out_species) or []
        if not sel:
            print(f"[eval][leaveout] WARNING: no validation videos found for species: {left_out_species} (skip {m.key})")
            continue
        print(f"[eval] leaveout model {m.key} on species={left_out_species} (n_videos={len(sel)})")
        for route_name in ("day", "night"):
            sel_route = [v for v in sel if v.route == route_name]
            if not sel_route:
                continue
            print(f"[eval] leaveout model {m.key} route={route_name} (n_videos={len(sel_route)})")
            for i, vid in enumerate(sel_route, start=1):
                out_root = run_dir / "pipelines" / f"{route_name}_videos" / m.key / vid.video_key
                log_path = logs_dir / f"gateway__{m.key}__{vid.video_key}.log"
                print(f"[eval][{m.key}][{route_name}] {i}/{len(sel_route)} {vid.species_name} :: {vid.video_path.name}")

                if DRY_RUN:
                    metrics = {"error": "dry_run"}
                else:
                    try:
                        _, metrics = _run_gateway_for_video(
                            out_root=out_root,
                            video=vid,
                            model_path=m.ckpt_path,
                            log_path=log_path,
                            dry_run=bool(DRY_RUN),
                        )
                    except Exception as e:
                        metrics = {"error": str(e)}

                raphael_metrics, raphael_path, lab_metrics, lab_path = _baseline_fields(vid.video_key)
                _record_row(
                    {
                        "val_video_name": vid.video_name,
                        "model_used": m.key,
                        "results": _metrics_str(metrics),
                        "inference_output_path": str(out_root),
                        "run_id": run_id,
                        "raphael_results": raphael_metrics,
                        "raphael_output_path": raphael_path,
                        "lab_results": lab_metrics,
                        "lab_output_path": lab_path,
                    }
                )

            # Combined per-species summary row (route-separated).
            # Since pipelines use only thr_10.0px by default, summing "best" is OK here.
            tp = fp = fn = 0
            names = {v.video_name for v in sel_route}
            for r in rows_out:
                if r.get("model_used") != m.key:
                    continue
                if r.get("val_video_name") not in names:
                    continue
                if f"/pipelines/{route_name}_videos/" not in str(r.get("inference_output_path") or ""):
                    continue
                ms = str(r.get("results") or "")
                m_tp = re.search(r"\btp=(\d+)\b", ms)
                m_fp = re.search(r"\bfp=(\d+)\b", ms)
                m_fn = re.search(r"\bfn=(\d+)\b", ms)
                if not (m_tp and m_fp and m_fn):
                    continue
                tp += int(m_tp.group(1))
                fp += int(m_fp.group(1))
                fn += int(m_fn.group(1))

            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            acc = _accuracy(tp, fp, fn)
            combined_str = f"tp={tp} fp={fp} fn={fn} precision={prec:.6f} accuracy={acc:.6f} f1={f1:.6f}"
            _record_row(
                {
                    "val_video_name": f"{left_out_species}__{route_name}__COMBINED",
                    "model_used": m.key,
                    "results": combined_str,
                    "inference_output_path": str(run_dir / "pipelines" / f"{route_name}_videos" / m.key),
                    "run_id": run_id,
                    "raphael_results": "",
                    "raphael_output_path": "",
                    "lab_results": "",
                    "lab_output_path": "",
                }
            )

    # Write a manifest for reproducibility
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "model_outputs_root": str(Path(MODEL_OUTPUTS_ROOT).expanduser().resolve()),
            "val_individual_root": str(Path(VAL_INDIVIDUAL_ROOT).expanduser().resolve()),
            "val_videos_root": str(Path(VAL_VIDEOS_ROOT).expanduser().resolve()),
            "inference_root": str(Path(INFERENCE_ROOT).expanduser().resolve()),
            "actual_outputs_root": str(Path(ACTUAL_OUTPUTS_ROOT).expanduser().resolve()),
        },
        "gateway": {
            "routing_mode": "name_based_with_overrides",
            "route_by_species": dict(ROUTE_BY_SPECIES),
            "route_by_video_name": dict(ROUTE_BY_VIDEO_NAME),
            "route_default": str(ROUTE_DEFAULT),
            "require_explicit_route": bool(REQUIRE_EXPLICIT_ROUTE),
            "max_concurrent": int(GATEWAY_MAX_CONCURRENT),
            "force_tests": bool(FORCE_GATEWAY_TESTS),
        },
        "baselines": {
            "enabled": bool(RUN_BASELINES),
            "only_for_night": bool(BASELINES_ONLY_FOR_NIGHT),
            "lab_enabled": bool(RUN_LAB_BASELINE),
            "raphael_enabled": bool(RUN_RAPHAEL_BASELINE),
            "raphael_model_path": str(RAPHAEL_MODEL_PATH or ""),
        },
        "models": {k: str(v.ckpt_path) for k, v in models.items()},
        "n_videos": int(len(videos)),
        "dry_run": bool(DRY_RUN),
    }
    if not DRY_RUN:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Rewrite once at the end as a consistency pass (same content, deterministic order).
    if DRY_RUN:
        print(f"[dry-run] Would write final results CSV → {FINAL_RESULTS_CSV}")
    else:
        _write_final_csv(FINAL_RESULTS_CSV, rows_out)
        print(f"[done] Wrote CSV → {FINAL_RESULTS_CSV}")
        print(f"[done] Outputs → {run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
