#!/usr/bin/env python3
from __future__ import annotations

"""
Tester-only automation for the integrated ingest → train → test pipeline.

What it does
------------
- Scans the model zoo for trained models (global day/night + per-species).
- Checks whether each model already has a testing output recorded for the latest
  validation dataset version.
- Runs gateway + validation only for models that are missing tests, then exits.

All paths + params are globals at the top (edit them for your machine).
You can also override ROOT / DRY_RUN via CLI flags.
"""

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Change log (append-only)
from codex_change_log import ChangeLogRun, SnapshotConfig

# -----------------------------------------------------------------------------
# User-configurable globals
# -----------------------------------------------------------------------------

ROOT_PATH: str | Path = ""

DATA_SUBDIR: str = "patch training datasets and pipeline validation data"
MODEL_ZOO_SUBDIR: str = "model zoo"
INFERENCE_OUTPUT_SUBDIR: str = "inference outputs"

# Where to write the append-only change log (defaults to the outer integrated prototype data folder).
LOG_ROOT_PATH: str | Path = "/mnt/Samsung_SSD_2TB/integrated prototype data"
CHANGE_LOG_FILENAME: str = "codex_change_log.jsonl"
ENABLE_CODEX_CHANGE_LOG: bool = True

# Evaluation set
# - Uses the latest version under:
#     <ROOT>/<DATA_SUBDIR>/Integrated_prototype_validation_datasets/combined species folder/vN_*/annotations.csv
EVAL_USE_COMBINED_VALIDATION: bool = True

# What to test
TEST_GLOBAL_MODELS: bool = True
TEST_SINGLE_SPECIES_MODELS: bool = True
ONLY_SPECIES: Sequence[str] = ()  # restrict species models (folder names) or leave empty

# Extra places to search for validation videos (used when they are not in the visible store).
VALIDATION_VIDEO_SEARCH_DIRS: Sequence[str | Path] = ()

# If True, require that a model has been tested on the *latest* validation dataset version.
# If False, any prior test output counts as "tested".
RETEST_IF_VALIDATION_CHANGED: bool = True

# For single-species models, choose how to route evaluation:
# - "auto_from_model_card": use model_card.txt time_of_day if it is day_time/night_time; else test both routes
# - "both": pass model as BOTH day_patch_model and night_cnn_model and evaluate both day+night videos
# - "day_only": only evaluate videos routed to day (pass as day_patch_model)
# - "night_only": only evaluate videos routed to night (pass as night_cnn_model)
SPECIES_EVAL_MODE: str = "auto_from_model_card"

# Gateway routing + execution
GATEWAY_BRIGHTNESS_THRESHOLD: float = 60.0
GATEWAY_BRIGHTNESS_NUM_FRAMES: int = 5
GATEWAY_MAX_CONCURRENT: int = 1  # keep 1 for GT-backed validation runs
FORCE_GATEWAY_TESTS: bool = True

# Optional: cap number of videos evaluated per model (None = no cap)
MAX_VIDEOS_PER_MODEL: int | None = None

# If True, do everything except actually running gateway / writing files.
DRY_RUN: bool = False

# -----------------------------------------------------------------------------


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text() or "{}")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except Exception:
        return


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tester-only automation (evaluate missing model outputs).")
    p.add_argument("--root", type=str, default="", help="Override ROOT_PATH.")
    p.add_argument("--log-root", type=str, default="", help="Override LOG_ROOT_PATH (where changelog lives).")
    p.add_argument("--dry-run", action="store_true", default=False, help="Do not run gateway / write outputs.")
    p.add_argument("--only-species", action="append", default=[], help="Restrict to a species (repeatable).")
    p.add_argument("--skip-global", action="store_true", default=False, help="Skip global model tests.")
    p.add_argument("--skip-species", action="store_true", default=False, help="Skip single-species model tests.")
    return p.parse_args(argv)


def _collect_tested_versions_from_results_history(
    *,
    history_dir: Path,
    legacy_path: Path,
) -> Dict[str, set[str]]:
    """
    Returns: model_path_str -> {validation_dataset_version,...}
    based on the model-zoo results_history snapshots (or legacy file).
    """
    import orchestrator as orch  # local import (same folder)

    tested: Dict[str, set[str]] = {}

    latest = orch._latest_snapshot_file(  # type: ignore[attr-defined]
        root=history_dir,
        prefix=orch.MODEL_ZOO_RESULTS_HISTORY_SNAPSHOT_PREFIX,  # type: ignore[attr-defined]
        suffix=".jsonl",
    )
    src = latest[1] if latest is not None else (legacy_path if legacy_path.exists() else None)
    if src is None:
        return tested

    for rec in _iter_jsonl_records(Path(src)):
        paths = rec.get("paths") if isinstance(rec.get("paths"), dict) else {}
        val_ver = str(paths.get("validation_dataset_version") or "").strip()
        if not val_ver:
            continue
        results = rec.get("results") if isinstance(rec.get("results"), list) else []
        has_global = any(isinstance(r, dict) and str(r.get("model_key")) == "global_model" for r in results)
        has_species = any(isinstance(r, dict) and str(r.get("model_key")) == "single_species_model" for r in results)

        models = rec.get("models") if isinstance(rec.get("models"), dict) else {}
        global_path = None
        species_path = None
        if isinstance(models.get("global"), dict):
            global_path = models["global"].get("path")
        if isinstance(models.get("single_species"), dict):
            species_path = models["single_species"].get("path")

        if has_global and isinstance(global_path, str) and global_path.strip():
            tested.setdefault(global_path.strip(), set()).add(val_ver)
        if has_species and isinstance(species_path, str) and species_path.strip():
            tested.setdefault(species_path.strip(), set()).add(val_ver)

    return tested


def _collect_tested_versions_from_inference_outputs(inference_root: Path) -> Dict[str, set[str]]:
    """
    Fallback: scan <inference_root>/*/run_record.json to detect which models were tested.
    Returns: model_path_str -> {validation_dataset_version,...}
    """
    tested: Dict[str, set[str]] = {}
    if not inference_root.exists():
        return tested

    # Typical layout: <inference_root>/<run_id>/run_record.json
    for run_dir in sorted([p for p in inference_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        rec_path = run_dir / "run_record.json"
        rec = _load_json(rec_path)
        if not rec:
            continue
        paths = rec.get("paths") if isinstance(rec.get("paths"), dict) else {}
        val_ver = str(paths.get("validation_dataset_version") or "").strip()
        if not val_ver:
            continue

        results = rec.get("results") if isinstance(rec.get("results"), list) else []
        has_global = any(isinstance(r, dict) and str(r.get("model_key")) == "global_model" for r in results)
        has_species = any(isinstance(r, dict) and str(r.get("model_key")) == "single_species_model" for r in results)

        models = rec.get("models") if isinstance(rec.get("models"), dict) else {}
        global_path = None
        species_path = None
        if isinstance(models.get("global"), dict):
            global_path = models["global"].get("path")
        if isinstance(models.get("single_species"), dict):
            species_path = models["single_species"].get("path")

        if has_global and isinstance(global_path, str) and global_path.strip():
            tested.setdefault(global_path.strip(), set()).add(val_ver)
        if has_species and isinstance(species_path, str) and species_path.strip():
            tested.setdefault(species_path.strip(), set()).add(val_ver)

    return tested


@dataclass(frozen=True)
class EvalItem:
    species_name: str
    video_name: str
    video_key: str
    video_path: Path
    route: str  # "day" | "night"
    gt_rows: List[Dict[str, int]]
    video_source: str


def _build_eval_index(
    *,
    root: Path,
    data_root: Path,
    registry: Dict[str, str],
    extra_search_dirs: Sequence[Path],
    dry_run: bool,
) -> Tuple[str | None, Path | None, List[EvalItem], Dict[str, str], bool, List[Dict[str, Any]]]:
    """
    Returns:
      (validation_version_name, combined_csv_path, items, updated_registry, missing_preview)
    """
    import orchestrator as orch  # local import (same folder)

    combined_root = data_root / "Integrated_prototype_validation_datasets" / "combined species folder"
    val_ver = orch._latest_version_dir(combined_root)  # type: ignore[attr-defined]
    if val_ver is None:
        return None, None, [], registry, [{"reason": "no_validation_versions", "root": str(combined_root)}]

    combined_csv = val_ver / "annotations.csv"
    if not combined_csv.exists():
        return val_ver.name, combined_csv, [], registry, [{"reason": "combined_csv_missing", "path": str(combined_csv)}]

    rows = orch._read_validation_combined_csv(combined_csv)  # type: ignore[attr-defined]
    grouped = orch._group_validation_rows_by_species_and_video(rows)  # type: ignore[attr-defined]

    store_root = orch._ensure_validation_video_store(root)  # type: ignore[attr-defined]
    individual_validation_root = data_root / "Integrated_prototype_validation_datasets" / "individual species folder"

    items: List[EvalItem] = []
    missing: List[Dict[str, Any]] = []
    registry_dirty = False

    for (sp, vn), gt_rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        vp, src = orch._resolve_validation_video_path(  # type: ignore[attr-defined]
            video_name=vn,
            species_name=sp,
            store_root=store_root,
            individual_validation_root=individual_validation_root,
            registry=registry,
            extra_search_dirs=extra_search_dirs,
        )
        if vp is None or (not Path(vp).exists()):
            missing.append({"species_name": sp, "video_name": vn, "reason": src})
            continue

        # Compute route (needed to avoid evaluating global models on the wrong pipeline).
        try:
            route = orch._route_for_video(  # type: ignore[attr-defined]
                Path(vp),
                thr=float(GATEWAY_BRIGHTNESS_THRESHOLD),
                frames=int(GATEWAY_BRIGHTNESS_NUM_FRAMES),
            )
        except Exception as e:
            missing.append({"species_name": sp, "video_name": vn, "reason": f"route_failed: {e}"})
            continue

        prev = registry.get(str(vn))
        nxt = str(Path(vp))
        if prev != nxt:
            registry_dirty = True
            registry[str(vn)] = nxt
        items.append(
            EvalItem(
                species_name=str(sp),
                video_name=str(vn),
                video_key=f"{sp}__{vn}",
                video_path=Path(vp),
                route=str(route),
                gt_rows=list(gt_rows),
                video_source=str(src),
            )
        )

    # Keep run_record smaller
    missing_preview = missing[:50]
    if not dry_run and missing and len(missing) > 0:
        print(f"[tester-only] WARNING: {len(missing)} validation video(s) missing (showing up to 50 in run_record).")
    return val_ver.name, combined_csv, items, registry, bool(registry_dirty), missing_preview


def _species_eval_mode_for_model(model_dir: Path) -> str:
    mode = str(SPECIES_EVAL_MODE or "").strip().lower()
    if mode in {"both", "day_only", "night_only"}:
        return mode
    # auto_from_model_card
    card = _load_json(model_dir / "model_card.txt") or {}
    tod = str(card.get("time_of_day") or "").strip().lower()
    if tod == "day_time":
        return "day_only"
    if tod == "night_time":
        return "night_only"
    return "both"


def _eval_model(
    *,
    repo_root: Path,
    gateway_path: Path,
    inference_root: Path,
    run_id: str,
    model_key: str,
    model_path: Path,
    day_patch_model: Path | None,
    night_cnn_model: Path | None,
    items: Sequence[EvalItem],
    dry_run: bool,
) -> List[Dict[str, Any]]:
    import orchestrator as orch  # local import (same folder)

    results: List[Dict[str, Any]] = []

    sel = list(items)
    if MAX_VIDEOS_PER_MODEL is not None:
        sel = sel[: max(0, int(MAX_VIDEOS_PER_MODEL))]

    for it in sel:
        out_root = inference_root / run_id / model_key / it.video_key
        if not dry_run:
            out_root.mkdir(parents=True, exist_ok=True)

        # Write GT for both pipeline roots (only one will be used depending on route)
        day_root = out_root / orch.DAY_OUTPUT_SUBDIR  # type: ignore[attr-defined]
        night_root = out_root / orch.NIGHT_OUTPUT_SUBDIR  # type: ignore[attr-defined]
        day_gt_dir = day_root / "ground truth"
        night_gt_dir = night_root / "ground truth"
        if dry_run:
            print(f"[dry-run] Would write GT CSVs under: {out_root}")
        else:
            orch._write_gt_csv(day_gt_dir / f"gt_{it.video_path.stem}.csv", it.gt_rows)  # type: ignore[attr-defined]
            orch._write_gt_csv(day_gt_dir / "gt.csv", it.gt_rows)  # type: ignore[attr-defined]
            orch._write_gt_csv(day_root / "gt.csv", it.gt_rows)  # type: ignore[attr-defined]
            orch._write_gt_csv(night_gt_dir / "gt.csv", it.gt_rows)  # type: ignore[attr-defined]
            orch._write_gt_csv(night_root / "gt.csv", it.gt_rows)  # type: ignore[attr-defined]

        t0 = time.time()
        try:
            orch._run_gateway(  # type: ignore[attr-defined]
                repo_root=repo_root,
                gateway_path=gateway_path,
                video_path=it.video_path,
                output_root=out_root,
                day_patch_model=day_patch_model,
                night_cnn_model=night_cnn_model,
                thr=float(GATEWAY_BRIGHTNESS_THRESHOLD),
                frames=int(GATEWAY_BRIGHTNESS_NUM_FRAMES),
                max_concurrent=int(GATEWAY_MAX_CONCURRENT),
                force_tests=bool(FORCE_GATEWAY_TESTS),
                dry_run=bool(dry_run),
            )
        except Exception as e:
            dt = float(time.time() - t0)
            results.append(
                {
                    "model_key": model_key,
                    "video_name": it.video_key,
                    "source_video_name": it.video_name,
                    "species_name": it.species_name,
                    "video_path": str(it.video_path),
                    "route": it.route,
                    "output_root": str(out_root),
                    "duration_s": dt,
                    "validation_metrics": {"error": "gateway_failed", "message": str(e)},
                }
            )
            continue
        dt = float(time.time() - t0)

        # Parse validation metrics from whichever pipeline was routed
        if it.route == "day":
            stage_dir = out_root / orch.DAY_OUTPUT_SUBDIR / "stage5 validation" / it.video_path.stem  # type: ignore[attr-defined]
        else:
            stage_dir = out_root / orch.NIGHT_OUTPUT_SUBDIR / "stage9 validation" / it.video_path.stem  # type: ignore[attr-defined]
        metrics = orch._parse_validation_metrics(stage_dir)  # type: ignore[attr-defined]

        results.append(
            {
                "model_key": model_key,
                "video_name": it.video_key,
                "source_video_name": it.video_name,
                "species_name": it.species_name,
                "video_path": str(it.video_path),
                "route": it.route,
                "output_root": str(out_root),
                "duration_s": dt,
                "validation_metrics": metrics,
            }
        )

    return results


def _summarize_best_f1(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        mk = str(r.get("model_key") or "")
        if mk:
            by_model.setdefault(mk, []).append(r)

    out: Dict[str, Any] = {}
    for mk, rows in by_model.items():
        f1s: List[float] = []
        n_with = 0
        for r in rows:
            m = r.get("validation_metrics")
            if not isinstance(m, dict):
                continue
            best = m.get("best")
            if not isinstance(best, dict):
                continue
            try:
                f1 = float(best.get("f1"))
            except Exception:
                continue
            n_with += 1
            f1s.append(f1)
        out[mk] = {
            "n_videos": int(len(rows)),
            "n_with_metrics": int(n_with),
            "mean_best_f1": (sum(f1s) / len(f1s)) if f1s else None,
        }
    return out


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    root_arg = (str(args.root or "").strip() or str(ROOT_PATH or "").strip()).strip()
    if not root_arg:
        print("Set ROOT_PATH at top of this file or pass --root.", file=sys.stderr)
        return 2

    log_root_arg = (str(args.log_root or "").strip() or str(LOG_ROOT_PATH or "").strip()).strip()
    if not log_root_arg:
        log_root_arg = root_arg

    dry_run = bool(DRY_RUN) or bool(args.dry_run)
    only_species = tuple(str(s).strip() for s in (args.only_species or []) if str(s).strip()) or tuple(ONLY_SPECIES)
    skip_global = bool(args.skip_global) or (not bool(TEST_GLOBAL_MODELS))
    skip_species = bool(args.skip_species) or (not bool(TEST_SINGLE_SPECIES_MODELS))

    root = Path(root_arg).expanduser().resolve()
    log_root = Path(log_root_arg).expanduser().resolve()
    data_root = root / str(DATA_SUBDIR)
    model_root = root / str(MODEL_ZOO_SUBDIR)
    inference_root = root / str(INFERENCE_OUTPUT_SUBDIR)

    if not data_root.exists():
        print(f"[tester-only] Data root not found: {data_root} (nothing to test)")
        return 0
    if not model_root.exists():
        print(f"[tester-only] Model zoo root not found: {model_root} (nothing to test)")
        return 0

    import orchestrator as orch  # type: ignore

    # Locate gateway + repo root
    repo_root = orch._REPO_ROOT  # type: ignore[attr-defined]
    gateway_path = (repo_root / orch.GATEWAY_REL_PATH).resolve()  # type: ignore[attr-defined]
    if not gateway_path.exists():
        raise FileNotFoundError(gateway_path)

    zoo = orch._ensure_model_zoo_scaffold(model_root)  # type: ignore[attr-defined]
    history_dir = zoo["results_history_dir"]
    legacy_history = zoo["legacy_results_history_file"]
    registry_dir = zoo["video_registry_dir"]
    legacy_registry = zoo["legacy_video_registry_file"]

    registry, registry_prev_version, _ = orch._load_latest_video_registry(  # type: ignore[attr-defined]
        registry_dir=registry_dir, legacy_path=legacy_registry
    )

    # Build evaluation index (validation videos + GT)
    extra_dirs: List[Path] = []
    for d in VALIDATION_VIDEO_SEARCH_DIRS:
        try:
            extra_dirs.append(Path(d).expanduser().resolve())
        except Exception:
            continue

    val_ver_name, combined_csv, eval_items, registry, registry_dirty, missing_preview = _build_eval_index(
        root=root,
        data_root=data_root,
        registry=registry,
        extra_search_dirs=extra_dirs,
        dry_run=dry_run,
    )
    if val_ver_name is None or combined_csv is None:
        print("[tester-only] No validation dataset found; nothing to test.")
        return 0

    # Determine which models have already been tested.
    tested_hist = _collect_tested_versions_from_results_history(history_dir=history_dir, legacy_path=legacy_history)
    tested_inf = _collect_tested_versions_from_inference_outputs(inference_root)
    tested: Dict[str, set[str]] = {}
    for mp, vers in tested_hist.items():
        tested.setdefault(mp, set()).update(vers)
    for mp, vers in tested_inf.items():
        tested.setdefault(mp, set()).update(vers)

    def _already_tested(model_path: Path) -> bool:
        key = str(model_path)
        vers = tested.get(key, set())
        if not vers:
            return False
        if not bool(RETEST_IF_VALIDATION_CHANGED):
            return True
        return str(val_ver_name) in vers

    # Decide which models need testing.
    todo: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    if not skip_global:
        # latest global day
        day_latest = orch._latest_version_dir(Path(zoo["day_global"]))  # type: ignore[attr-defined]
        if day_latest is not None and (day_latest / "model.pt").exists():
            mp = day_latest / "model.pt"
            candidates.append({"kind": "global", "route": "day", "model_dir": day_latest, "model_path": mp})
            if not _already_tested(mp):
                todo.append({"kind": "global", "route": "day", "model_dir": day_latest, "model_path": mp})
        # latest global night
        night_latest = orch._latest_version_dir(Path(zoo["night_global"]))  # type: ignore[attr-defined]
        if night_latest is not None and (night_latest / "model.pt").exists():
            mp = night_latest / "model.pt"
            candidates.append({"kind": "global", "route": "night", "model_dir": night_latest, "model_path": mp})
            if not _already_tested(mp):
                todo.append({"kind": "global", "route": "night", "model_dir": night_latest, "model_path": mp})

    if not skip_species:
        allowed: set[str] | None = None
        if only_species:
            allowed = {orch._safe_name(s) for s in only_species if orch._safe_name(s)}  # type: ignore[attr-defined]

        sp_root = Path(zoo["single_root"])
        if sp_root.exists():
            for sp_dir in sorted([p for p in sp_root.iterdir() if p.is_dir()], key=lambda p: p.name):
                sp_name = orch._safe_name(sp_dir.name)  # type: ignore[attr-defined]
                if not sp_name:
                    continue
                if allowed is not None and sp_name not in allowed:
                    continue
                latest = orch._latest_version_dir(sp_dir)  # type: ignore[attr-defined]
                if latest is None:
                    continue
                mp = latest / "model.pt"
                if not mp.exists():
                    continue
                candidates.append({"kind": "single_species", "species": sp_name, "model_dir": latest, "model_path": mp})
                if _already_tested(mp):
                    continue
                todo.append({"kind": "single_species", "species": sp_name, "model_dir": latest, "model_path": mp})

    if not candidates:
        print(f"[tester-only] Nothing to test: no trained models found under: {model_root}")
        return 0

    if not todo:
        print(
            "[tester-only] Up to date: trained model outputs already have testing outputs "
            + (f"for validation={val_ver_name}." if RETEST_IF_VALIDATION_CHANGED else ".")
        )
        return 0

    log_path = (log_root / str(CHANGE_LOG_FILENAME)).expanduser().resolve()
    if not dry_run:
        log_root.mkdir(parents=True, exist_ok=True)

    print(f"[tester-only] Testing needed for {len(todo)} model(s). validation={val_ver_name} dry_run={dry_run}")

    new_records: List[Dict[str, Any]] = []
    tested_models: List[Dict[str, Any]] = []
    cfg = SnapshotConfig(root=root, scopes=[model_root, inference_root])
    meta = {"actor": "tester_only", "dry_run": bool(dry_run), "validation_dataset_version": str(val_ver_name)}
    with ChangeLogRun(cfg=cfg, log_path=log_path, meta=meta, enabled=bool(ENABLE_CODEX_CHANGE_LOG)):
        if not dry_run:
            inference_root.mkdir(parents=True, exist_ok=True)

        for job in todo:
            kind = str(job["kind"])
            model_dir = Path(job["model_dir"])
            model_path = Path(job["model_path"])
            model_ver = model_dir.name

            if kind == "global":
                route = str(job["route"])
                label = f"global_{route}"
                # Filter videos by route so we don't accidentally validate using the wrong pipeline/model.
                items = [it for it in eval_items if it.route == route]
                if not items:
                    print(f"[tester-only] Skipping {label}: no validation videos routed to {route}.")
                    continue
                if route == "day":
                    day_patch_model, night_cnn_model = model_path, None
                else:
                    day_patch_model, night_cnn_model = None, model_path
                model_key = "global_model"
                print(f"[tester-only] Running {label}: model={model_path} videos={len(items)}")
            else:
                sp = str(job["species"])
                label = f"species_{sp}"
                mode = _species_eval_mode_for_model(model_dir)
                items_all = [it for it in eval_items if orch._safe_name(it.species_name) == sp]  # type: ignore[attr-defined]
                if mode == "day_only":
                    items = [it for it in items_all if it.route == "day"]
                    day_patch_model, night_cnn_model = model_path, None
                elif mode == "night_only":
                    items = [it for it in items_all if it.route == "night"]
                    day_patch_model, night_cnn_model = None, model_path
                else:
                    items = list(items_all)
                    day_patch_model, night_cnn_model = model_path, model_path
                if not items:
                    print(f"[tester-only] Skipping {label}: no validation videos selected (mode={mode}).")
                    continue
                model_key = "single_species_model"
                print(
                    f"[tester-only] Running {label}: model={model_path} videos={len(items)} mode={mode}"
                )

            run_id = orch._safe_name(  # type: ignore[attr-defined]
                "__".join(["test_only", orch._run_tag(), label, model_ver, f"val_{val_ver_name}"])  # type: ignore[attr-defined]
            )

            results = _eval_model(
                repo_root=repo_root,
                gateway_path=gateway_path,
                inference_root=inference_root,
                run_id=run_id,
                model_key=model_key,
                model_path=model_path,
                day_patch_model=day_patch_model,
                night_cnn_model=night_cnn_model,
                items=items,
                dry_run=dry_run,
            )

            summary = _summarize_best_f1(results)
            if summary:
                for mk, s in summary.items():
                    mean_f1 = s.get("mean_best_f1")
                    mean_str = f"{float(mean_f1):.4f}" if isinstance(mean_f1, (int, float)) else "n/a"
                    print(
                        f"[tester-only] Summary {label}/{mk}: mean_best_f1={mean_str} n_videos={s.get('n_with_metrics', 0)}/{s.get('n_videos', 0)}"
                    )

            record: Dict[str, Any] = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "batch": {
                    "kind": "test_only",
                    "validation_dataset_version": str(val_ver_name),
                    "combined_csv": str(combined_csv),
                    "model_label": label,
                    "model_version_dir": str(model_dir),
                    "n_eval_videos": int(len(items)),
                    "missing_videos_preview": missing_preview,
                },
                "paths": {
                    "data_root": str(data_root),
                    "model_zoo_root": str(model_root),
                    "inference_output_root": str(inference_root),
                    "validation_dataset_version": str(val_ver_name),
                },
                "models": {
                    "global": {"path": str(model_path) if kind == "global" else None, "train_metrics": None},
                    "single_species": {"path": str(model_path) if kind != "global" else None, "train_metrics": None},
                },
                "gateway": {
                    "brightness_threshold": float(GATEWAY_BRIGHTNESS_THRESHOLD),
                    "brightness_frames": int(GATEWAY_BRIGHTNESS_NUM_FRAMES),
                    "max_concurrent": int(GATEWAY_MAX_CONCURRENT),
                    "force_tests": bool(FORCE_GATEWAY_TESTS),
                },
                "evaluation": {
                    "eval_source": {
                        "kind": "combined_validation_sweep",
                        "combined_version": str(val_ver_name),
                        "combined_csv": str(combined_csv),
                    }
                },
                "results": results,
                "results_summary": summary,
            }

            run_dir = inference_root / run_id
            rec_path = run_dir / "run_record.json"
            if dry_run:
                print(f"[dry-run] Would write run record → {rec_path}")
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                rec_path.write_text(json.dumps(record, indent=2))

            new_records.append(record)
            tested_models.append(
                {
                    "kind": kind,
                    "label": label,
                    "model_path": str(model_path),
                    "model_version_dir": str(model_dir),
                    "model_version": str(model_ver),
                    "validation_dataset_version": str(val_ver_name),
                    "run_id": str(run_id),
                    "n_eval_videos": int(len(items)),
                }
            )

        # Persist updated video registry snapshot (if we resolved any videos).
        try:
            meta["n_models_tested"] = int(len(tested_models))
            meta["tested_models"] = list(tested_models)
        except Exception:
            pass
        if registry_dirty:
            reg_snap = orch._write_video_registry_snapshot(  # type: ignore[attr-defined]
                registry_dir=registry_dir,
                registry=registry,
                prev_version=int(registry_prev_version),
                dry_run=bool(dry_run),
            )
            if reg_snap is not None:
                print(f"[tester-only] Video registry snapshot → {reg_snap}")

        # Optional: append all new run records into a single new results_history snapshot.
        if new_records:
            latest = orch._latest_snapshot_file(  # type: ignore[attr-defined]
                root=history_dir,
                prefix=orch.MODEL_ZOO_RESULTS_HISTORY_SNAPSHOT_PREFIX,  # type: ignore[attr-defined]
                suffix=".jsonl",
            )
            if latest is not None:
                prev_n, prev_path = latest
            elif legacy_history.exists():
                prev_n, prev_path = 0, legacy_history
            else:
                prev_n, prev_path = 0, None

            new_n = int(prev_n) + 1
            out = history_dir / f"{orch.MODEL_ZOO_RESULTS_HISTORY_SNAPSHOT_PREFIX}__v{new_n}_{orch._run_tag()}.jsonl"  # type: ignore[attr-defined]
            if dry_run:
                print(f"[dry-run] Would append results history snapshot → {out}")
            else:
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
                            f.seek(-1, 2)
                            last = f.read(1)
                        if last not in {b"\n"}:
                            needs_nl = True
                except Exception:
                    needs_nl = True

                with out.open("ab") as f:
                    if needs_nl:
                        f.write(b"\n")
                    for rec in new_records:
                        f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
                print(f"[tester-only] Results history snapshot → {out}")

    print(f"[tester-only] Change log → {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
