#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import orchestrator as orch


DEFAULT_OBSERVED_DIR = Path("/home/guest/Desktop/arnav's files/temp data/4k")
DEFAULT_LOCAL_DATA_ROOT = Path(
    "/home/guest/Desktop/arnav's files/firefly pipeline/.local_ingest_output/patch training datasets and pipeline validation data"
)


def _infer_species_name(pairs: list[orch.ObservedPair]) -> str:
    tokens: list[str] = []
    for p in pairs:
        parts = p.firefly_csv.stem.split("_")
        if parts:
            tokens.append(parts[-1].strip().lower())
    if not tokens:
        return "unknown_species"
    # Use the most frequent suffix token from CSV names.
    species = max(set(tokens), key=tokens.count)
    species = orch._safe_name(species)
    return species or "unknown_species"


def _ensure_scaffold(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "batch_exports").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_datasets" / "integrated pipeline datasets").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_datasets" / "single species datasets").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_validation_datasets" / "combined species folder").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_validation_datasets" / "individual species folder").mkdir(parents=True, exist_ok=True)


def _run_species_scaler_local(
    *,
    repo_root: Path,
    annotations_csv: Path,
    video_path: Path,
    data_root: Path,
    train_fraction: float,
    train_val_seed: int,
    auto_load_sibling: bool,
) -> None:
    scaler_dir = repo_root / "species scaler"
    scaler_py = scaler_dir / "species_scaler.py"
    if not scaler_py.exists():
        raise FileNotFoundError(scaler_py)

    # species_scaler.py defines derived roots at import time, so we must override
    # DATA_ROOT and all dependent roots before calling ss.main().
    code = "\n".join(
        [
            "from pathlib import Path",
            "import species_scaler as ss",
            f"ss.ANNOTATIONS_CSV_PATH = Path({repr(str(annotations_csv))})",
            f"ss.VIDEO_PATH = Path({repr(str(video_path))})",
            f"ss.DATA_ROOT = Path({repr(str(data_root))})",
            "ss.BATCH_EXPORT_ROOT = ss.DATA_ROOT / 'batch_exports'",
            "ss.TRAIN_DATASETS_ROOT = ss.DATA_ROOT / 'Integrated_prototype_datasets'",
            "ss.VALIDATION_DATASETS_ROOT = ss.DATA_ROOT / 'Integrated_prototype_validation_datasets'",
            f"ss.TRAIN_FRACTION_OF_BATCH = {float(train_fraction)}",
            f"ss.TRAIN_VAL_SPLIT_SEED = {int(train_val_seed)}",
            f"ss.AUTO_LOAD_SIBLING_CLASS_CSV = {bool(auto_load_sibling)}",
            "ss.main()",
        ]
    )
    subprocess.run([sys.executable, "-u", "-c", code], cwd=str(scaler_dir), check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ingestion only (no training/eval), writing to local output root.")
    p.add_argument("--observed-dir", type=Path, default=DEFAULT_OBSERVED_DIR)
    p.add_argument("--data-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT)
    p.add_argument("--time-of-day", choices=["day_time", "night_time"], default="night_time")
    p.add_argument("--species-name", type=str, default="", help="Optional override. If omitted, inferred from CSV suffix.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    observed_dir = args.observed_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    time_of_day = str(args.time_of_day).strip()

    repo_root = Path(__file__).resolve().parents[1]
    orch.SPECIES_SCALER_REL_DIR = Path("species scaler")

    _ensure_scaffold(data_root)

    pairs = orch._discover_observed_pairs(observed_dir)
    species_name = orch._safe_name(str(args.species_name).strip()) if str(args.species_name).strip() else _infer_species_name(pairs)

    train_pairs, val_pairs = orch._split_pairs_train_vs_val(pairs)
    train_video_names = {p.video_name for p in train_pairs}
    val_video_names = {p.video_name for p in val_pairs}

    staging_dir = (
        data_root
        / "batch_exports"
        / "orchestrator_observed_dir_staging"
        / f"{orch._run_tag()}__{orch._safe_name(observed_dir.name)}"
    )
    staged_pairs = orch._stage_pairs_for_species_scaler(
        pairs,
        species_name=species_name,
        staging_dir=staging_dir,
        time_of_day=time_of_day,
        dry_run=False,
    )
    staged_by_video = {sp.pair.video_name: sp for sp in staged_pairs}

    print(f"[ingest-only] observed_dir={observed_dir}")
    print(f"[ingest-only] data_root={data_root}")
    print(f"[ingest-only] species_name={species_name}")
    print(f"[ingest-only] time_of_day={time_of_day}")
    print(f"[ingest-only] matched_pairs={len(pairs)} train={len(train_pairs)} validation={len(val_pairs)}")
    print(f"[ingest-only] staging_dir={staging_dir}")

    for i, p in enumerate(pairs, start=1):
        sp = staged_by_video[p.video_name]
        if p.video_name in train_video_names:
            split = "train"
            train_fraction = 1.0
            auto_load_sibling = True
        elif p.video_name in val_video_names:
            split = "validation"
            train_fraction = 0.0
            auto_load_sibling = False
        else:
            continue

        print(
            f"\n[ingest-only] ({i}/{len(pairs)}) {p.video_path.name} split={split} train_fraction={train_fraction}"
        )
        _run_species_scaler_local(
            repo_root=repo_root,
            annotations_csv=sp.staged_firefly_csv,
            video_path=p.video_path,
            data_root=data_root,
            train_fraction=float(train_fraction),
            train_val_seed=1337,
            auto_load_sibling=bool(auto_load_sibling),
        )

    integrated_root = data_root / "Integrated_prototype_datasets" / "integrated pipeline datasets"
    single_root = data_root / "Integrated_prototype_datasets" / "single species datasets" / species_name
    val_comb_root = data_root / "Integrated_prototype_validation_datasets" / "combined species folder"
    val_species_root = data_root / "Integrated_prototype_validation_datasets" / "individual species folder" / species_name

    print("\n[ingest-only] completed")
    print(f"[ingest-only] latest integrated: {orch._latest_version_dir(integrated_root)}")
    print(f"[ingest-only] latest single-species: {orch._latest_version_dir(single_root)}")
    print(f"[ingest-only] latest validation combined: {orch._latest_version_dir(val_comb_root)}")
    print(f"[ingest-only] latest validation species: {orch._latest_version_dir(val_species_root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
