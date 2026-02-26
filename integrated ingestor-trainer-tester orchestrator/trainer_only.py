#!/usr/bin/env python3
from __future__ import annotations

"""
Trainer-only automation for the integrated ingest → train → test pipeline.

What it does
------------
- Scans the ingested dataset folder structure under:
    <ROOT>/<DATA_SUBDIR>/Integrated_prototype_datasets/
- Detects which ingested datasets do NOT yet have an equivalent trained model in:
    <ROOT>/<MODEL_ZOO_SUBDIR>/
- Trains only the missing models and exits.
- If everything is already up to date, prints that and exits cleanly.

All paths + hyperparameters are globals at the top (edit them for your machine).
You can also override ROOT / DRY_RUN via CLI flags.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Change log (append-only)
from codex_change_log import ChangeLogRun, SnapshotConfig, default_log_path

# -----------------------------------------------------------------------------
# User-configurable globals
# -----------------------------------------------------------------------------

# Root folder that contains the standard integrated layout:
#   <ROOT>/<DATA_SUBDIR>/...
#   <ROOT>/<MODEL_ZOO_SUBDIR>/...
ROOT_PATH: str | Path = ""

DATA_SUBDIR: str = "patch training datasets and pipeline validation data"
MODEL_ZOO_SUBDIR: str = "model zoo"

# Where to write the append-only change log (defaults to the outer integrated prototype data folder).
LOG_ROOT_PATH: str | Path = "/mnt/Samsung_SSD_2TB/integrated prototype data"
CHANGE_LOG_FILENAME: str = "codex_change_log.jsonl"
ENABLE_CODEX_CHANGE_LOG: bool = True

# What to train
TRAIN_GLOBAL_MODELS: bool = True
TRAIN_SINGLE_SPECIES_MODELS: bool = True

# Training config (ResNet patch/CNN classifier)
TRAIN_EPOCHS: int = 50
TRAIN_BATCH_SIZE: int = 128
TRAIN_LR: float = 3e-4
TRAIN_NUM_WORKERS: int = 2
TRAIN_RESNET: str = "resnet18"  # resnet18|34|50|101|152
TRAIN_SEED: int = 1337

# Dataset sanitization (preflight)
SANITIZE_DATASET_IMAGES: bool = True
SANITIZE_DATASET_MODE: str = "quarantine"  # "quarantine" | "delete"
SANITIZE_DATASET_VERIFY_WITH_PIL: bool = True
SANITIZE_DATASET_REPORT_MAX: int = 20

# If True, do everything except actual training writes.
DRY_RUN: bool = False

# Optional: restrict to these species (dataset folder names); empty = all.
ONLY_SPECIES: Sequence[str] = ()

# Single-species model_card field (metadata only). Choose:
# - "day_time" / "night_time" if your single-species datasets are time-specific
# - "mixed_time" if they can contain both.
SINGLE_SPECIES_MODEL_TIME_OF_DAY: str = "mixed_time"

# -----------------------------------------------------------------------------


def _lazy_import_stage2_trainer():
    try:
        import stage2_trainer as tr  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not import stage2_trainer (training dependency). "
            "Ensure torch/torchvision are installed in this environment."
        ) from e
    return tr


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text() or "{}")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _model_is_up_to_date(*, model_dir: Optional[Path], expected_dataset_version: str, expected_time_of_day: str | None) -> bool:
    if model_dir is None:
        return False
    model_pt = model_dir / "model.pt"
    if not model_pt.exists():
        return False
    card = _load_json_file(model_dir / "model_card.txt") or {}
    if str(card.get("dataset_version") or "") != str(expected_dataset_version):
        return False
    if expected_time_of_day is not None:
        if str(card.get("time_of_day") or "") != str(expected_time_of_day):
            return False
    return True


def _train_resnet_classifier(
    *,
    data_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    resnet: str,
    seed: int,
    dry_run: bool,
) -> Dict[str, Any]:
    model_path = out_dir / "model.pt"
    metrics_path = out_dir / "training_metrics.json"

    if dry_run:
        print(
            "[dry-run] Would train:",
            f"data_dir={data_dir} -> {model_path} (epochs={epochs} batch={batch_size} lr={lr} workers={num_workers} resnet={resnet} seed={seed})",
        )
        return {}

    tr = _lazy_import_stage2_trainer()
    metrics = tr.train_resnet_classifier(
        data_dir=Path(data_dir).expanduser().resolve(),
        best_model_path=Path(model_path).expanduser().resolve(),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        num_workers=int(num_workers),
        resnet_model=str(resnet),
        seed=int(seed),
        no_gui=True,
        metrics_out=str(metrics_path),
        sanitize_dataset=bool(SANITIZE_DATASET_IMAGES),
        sanitize_mode=str(SANITIZE_DATASET_MODE),
        sanitize_verify_with_pil=bool(SANITIZE_DATASET_VERIFY_WITH_PIL),
        sanitize_report_max=int(SANITIZE_DATASET_REPORT_MAX),
    )
    return metrics if isinstance(metrics, dict) else {}


def _count_train_images(dataset_final_dir: Path) -> Tuple[int, int]:
    import orchestrator as orch  # local import (same folder)

    ff_n = orch._count_images_in_dir(dataset_final_dir / "train" / "firefly")  # type: ignore[attr-defined]
    bg_n = orch._count_images_in_dir(dataset_final_dir / "train" / "background")  # type: ignore[attr-defined]
    return int(ff_n), int(bg_n)


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trainer-only automation (scan ingested datasets and train missing models).")
    p.add_argument("--root", type=str, default="", help="Override ROOT_PATH.")
    p.add_argument("--log-root", type=str, default="", help="Override LOG_ROOT_PATH (where changelog lives).")
    p.add_argument("--dry-run", action="store_true", default=False, help="Do not run actual training.")
    p.add_argument("--only-species", action="append", default=[], help="Restrict to a specific species (repeatable).")
    p.add_argument("--skip-global", action="store_true", default=False, help="Skip training global day/night models.")
    p.add_argument("--skip-species", action="store_true", default=False, help="Skip training single-species models.")
    return p.parse_args(argv)


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
    skip_global = bool(args.skip_global) or (not bool(TRAIN_GLOBAL_MODELS))
    skip_species = bool(args.skip_species) or (not bool(TRAIN_SINGLE_SPECIES_MODELS))

    root = Path(root_arg).expanduser().resolve()
    log_root = Path(log_root_arg).expanduser().resolve()
    data_root = root / str(DATA_SUBDIR)
    model_root = root / str(MODEL_ZOO_SUBDIR)

    if not data_root.exists():
        print(f"[trainer-only] Data root not found: {data_root} (nothing to train)")
        return 0

    # Import the integrated orchestrator helpers (folder conventions + versioning helpers).
    import orchestrator as orch  # type: ignore

    zoo = orch._ensure_model_zoo_scaffold(model_root)  # type: ignore[attr-defined]

    train_root = data_root / "Integrated_prototype_datasets"
    integrated_root = train_root / "integrated pipeline datasets"
    single_species_root = train_root / "single species datasets"

    work: List[Dict[str, Any]] = []
    skipped: List[str] = []
    trainable_datasets = 0
    untrainable_datasets = 0

    # ---------------------------------------------------------------------
    # Global models (day/night) — keyed off the latest integrated dataset version.
    # ---------------------------------------------------------------------
    integrated_ver = orch._latest_version_dir(integrated_root)  # type: ignore[attr-defined]
    if integrated_ver is None:
        skipped.append(f"No integrated dataset versions found under: {integrated_root}")
    elif not skip_global:
        for time_of_day, dataset_time_dir, global_parent in (
            ("day_time", "day_time_dataset", zoo["day_global"]),
            ("night_time", "night_time_dataset", zoo["night_global"]),
        ):
            dataset_final = integrated_ver / dataset_time_dir / "final dataset"
            if not dataset_final.exists():
                skipped.append(f"Global {time_of_day}: missing dataset dir: {dataset_final}")
                continue
            ff_n, bg_n = _count_train_images(dataset_final)
            if ff_n <= 0 or bg_n <= 0:
                untrainable_datasets += 1
                skipped.append(
                    f"Global {time_of_day}: skipping training (need both classes). train_firefly={ff_n} train_background={bg_n} dataset={dataset_final}"
                )
                continue
            trainable_datasets += 1

            latest_model_dir = orch._latest_version_dir(Path(global_parent))  # type: ignore[attr-defined]
            up_to_date = _model_is_up_to_date(
                model_dir=latest_model_dir,
                expected_dataset_version=str(integrated_ver.name),
                expected_time_of_day=str(time_of_day),
            )
            if up_to_date:
                continue

            work.append(
                {
                    "kind": "global",
                    "time_of_day": time_of_day,
                    "dataset_version": str(integrated_ver.name),
                    "dataset_final_dir": str(dataset_final),
                    "model_parent": str(global_parent),
                    "integrated_ver": str(integrated_ver),
                    "dataset_time_dir": dataset_time_dir,
                }
            )

    # ---------------------------------------------------------------------
    # Single-species models — keyed off each species' latest dataset version.
    # ---------------------------------------------------------------------
    if not skip_species:
        if not single_species_root.exists():
            skipped.append(f"No single-species datasets root found: {single_species_root}")
        else:
            allowed: set[str] | None = None
            if only_species:
                allowed = {orch._safe_name(s) for s in only_species if orch._safe_name(s)}  # type: ignore[attr-defined]

            for sp_dir in sorted([p for p in single_species_root.iterdir() if p.is_dir()], key=lambda p: p.name):
                sp_name = orch._safe_name(sp_dir.name)  # type: ignore[attr-defined]
                if not sp_name:
                    continue
                if allowed is not None and sp_name not in allowed:
                    continue

                latest_ds_dir = orch._latest_version_dir(sp_dir)  # type: ignore[attr-defined]
                if latest_ds_dir is None:
                    continue
                dataset_final = latest_ds_dir / "final dataset"
                if not dataset_final.exists():
                    skipped.append(f"Species {sp_name}: missing dataset dir: {dataset_final}")
                    continue

                ff_n, bg_n = _count_train_images(dataset_final)
                if ff_n <= 0 or bg_n <= 0:
                    untrainable_datasets += 1
                    skipped.append(
                        f"Species {sp_name}: skipping training (need both classes). train_firefly={ff_n} train_background={bg_n} dataset={dataset_final}"
                    )
                    continue
                trainable_datasets += 1

                model_parent = Path(zoo["single_root"]) / sp_name
                model_parent.mkdir(parents=True, exist_ok=True)
                latest_model_dir = orch._latest_version_dir(model_parent)  # type: ignore[attr-defined]
                up_to_date = _model_is_up_to_date(
                    model_dir=latest_model_dir,
                    expected_dataset_version=str(latest_ds_dir.name),
                    expected_time_of_day=None,
                )
                if up_to_date:
                    continue

                work.append(
                    {
                        "kind": "single_species",
                        "species": sp_name,
                        "time_of_day": str(SINGLE_SPECIES_MODEL_TIME_OF_DAY),
                        "dataset_version": str(latest_ds_dir.name),
                        "dataset_final_dir": str(dataset_final),
                        "model_parent": str(model_parent),
                    }
                )

    if not work:
        if trainable_datasets <= 0 and untrainable_datasets > 0:
            print("[trainer-only] Nothing trainable: ingested datasets are missing required classes (firefly/background).")
        elif trainable_datasets <= 0 and untrainable_datasets <= 0:
            print("[trainer-only] Nothing to do: no ingested datasets found.")
        else:
            print("[trainer-only] Up to date: all trainable ingested datasets already have trained model outputs.")
        for msg in skipped[:50]:
            print(f"[trainer-only] Note: {msg}")
        return 0

    log_path = (log_root / str(CHANGE_LOG_FILENAME)).expanduser().resolve()
    if not dry_run:
        log_root.mkdir(parents=True, exist_ok=True)

    print(f"[trainer-only] Training needed for {len(work)} model(s). dry_run={dry_run}")
    for w in work:
        if w.get("kind") == "global":
            print(
                f"  - global {w['time_of_day']}: dataset={w['dataset_final_dir']} (integrated_ver={w['dataset_version']})"
            )
        else:
            print(
                f"  - species {w['species']}: dataset={w['dataset_final_dir']} (dataset_ver={w['dataset_version']})"
            )

    trained: List[Dict[str, Any]] = []
    cfg = SnapshotConfig(root=root, scopes=[data_root, model_root])
    meta = {"actor": "trainer_only", "dry_run": bool(dry_run), "n_models_planned": int(len(work))}
    with ChangeLogRun(cfg=cfg, log_path=log_path, meta=meta, enabled=bool(ENABLE_CODEX_CHANGE_LOG)):
        for w in work:
            kind = str(w["kind"])
            model_parent = Path(str(w["model_parent"]))
            out_ver = orch._next_version_dir(model_parent)  # type: ignore[attr-defined]
            if dry_run:
                print(f"[dry-run] Would create model version dir: {out_ver}")
            else:
                out_ver.mkdir(parents=True, exist_ok=False)

            dataset_final = Path(str(w["dataset_final_dir"]))
            metrics = _train_resnet_classifier(
                data_dir=dataset_final,
                out_dir=out_ver,
                epochs=int(TRAIN_EPOCHS),
                batch_size=int(TRAIN_BATCH_SIZE),
                lr=float(TRAIN_LR),
                num_workers=int(TRAIN_NUM_WORKERS),
                resnet=str(TRAIN_RESNET),
                seed=int(TRAIN_SEED),
                dry_run=dry_run,
            )

            # Write model card (best-effort; keeps consistency with orchestrator conventions)
            model_card: Dict[str, Any] = {
                "kind": kind,
                "dataset_version": str(w.get("dataset_version") or ""),
                "dataset_dir": str(dataset_final),
                "train_metrics": metrics,
            }
            if kind == "global":
                time_of_day = str(w.get("time_of_day") or "")
                model_card["time_of_day"] = time_of_day
                try:
                    integrated_ver = Path(str(w.get("integrated_ver") or ""))
                    dataset_time_dir = str(w.get("dataset_time_dir") or "")
                    patch_locations_train_csv = integrated_ver / dataset_time_dir / "patch_locations_train.csv"
                    species_summary_csv = orch._species_summary_from_patch_locations(patch_locations_train_csv)  # type: ignore[attr-defined]
                    species_summary_dir = orch._species_summary_from_dataset_dir(dataset_final)  # type: ignore[attr-defined]
                    species_summary = species_summary_dir if species_summary_dir.get("counts") else species_summary_csv

                    warn_mismatch = None
                    if species_summary_dir.get("counts") and species_summary_csv.get("counts"):
                        if species_summary_dir.get("counts") != species_summary_csv.get("counts"):
                            warn_mismatch = (
                                "patch_locations_train.csv species counts do not match dataset_dir/train/firefly. "
                                "This usually means legacy-imported patches exist on disk but are missing from patch_locations*.csv."
                            )

                    model_card.update(
                        {
                            "trained_species": species_summary.get("species", []),
                            "trained_species_counts": species_summary.get("counts", {}),
                            "trained_species_source_csv": species_summary_csv.get("source_csv"),
                            "trained_species_error": species_summary_csv.get("error"),
                            "trained_species_source_train_dir": str(dataset_final / "train" / "firefly"),
                            "trained_species_counts_by_split": species_summary_dir.get("counts_by_split", {}),
                            "trained_species_counts_total": species_summary_dir.get("counts_total", {}),
                            "trained_species_unknown_counts_by_split": species_summary_dir.get(
                                "unknown_counts_by_split", {}
                            ),
                            "trained_species_warning": warn_mismatch,
                        }
                    )
                except Exception as e:
                    model_card["trained_species_error"] = f"summary_failed: {e}"
            elif kind == "single_species":
                model_card["species"] = str(w.get("species") or "")
                model_card["time_of_day"] = str(w.get("time_of_day") or "")

            card_path = out_ver / "model_card.txt"
            if dry_run:
                print(f"[dry-run] Would write model card → {card_path}")
            else:
                card_path.write_text(json.dumps(model_card, indent=2))

            trained.append(
                {
                    "kind": kind,
                    "out_dir": str(out_ver),
                    "model_path": str(out_ver / "model.pt"),
                    "dataset_version": str(w.get("dataset_version") or ""),
                }
            )
        try:
            meta["n_models_trained"] = int(len(trained))
            meta["trained_models"] = list(trained)
        except Exception:
            pass

    print(f"[trainer-only] Done. Trained {len(trained)} model(s).")
    for t in trained:
        print(f"  - {t['kind']}: {t['model_path']} (dataset_version={t['dataset_version']})")
    for msg in skipped[:50]:
        print(f"[trainer-only] Note: {msg}")
    print(f"[trainer-only] Change log → {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
