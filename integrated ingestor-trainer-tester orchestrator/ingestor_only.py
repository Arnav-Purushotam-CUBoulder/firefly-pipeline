#!/usr/bin/env python3
from __future__ import annotations

"""
Ingestor-only automation for the integrated ingest → train → test pipeline.

Goal
----
Scan a RAW videos root (with one folder per species, prefixed by day_/night_), and ingest ONLY the
videos that have not yet been ingested for each species (supports new species
and new videos added later to an existing species). No training. No testing.

Also writes an append-only change log (JSONL) that records folder/file changes
done by each ingestion job, with patch-image folders treated as "bulk" to keep
log size realistic.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import stage1_ingestor

from codex_change_log import ChangeLogRun, SnapshotConfig, build_ingestion_index, default_log_path

# -----------------------------------------------------------------------------
# User-configurable globals (edit these)
# -----------------------------------------------------------------------------

# RAW input root (contains one folder per species).
# Expected folder naming convention (case-insensitive): day_<species> / night_<species>
# Each folder contains .mp4 + annotator .csv files.
RAW_VIDEOS_ROOT: str | Path = "/mnt/Samsung_SSD_2TB/integrated prototype raw videos"

# Integrated pipeline ROOT (this is the same "ROOT_PATH" you use in the integrated orchestrator)
ROOT_PATH: str | Path = "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data"

# Where to write the append-only change log + rollback instructions.
# The user requested this outer folder:
LOG_ROOT_PATH: str | Path = "/mnt/Samsung_SSD_2TB/integrated prototype data"
CHANGE_LOG_FILENAME: str = "codex_change_log.jsonl"

# Subfolders under ROOT_PATH
DATA_SUBDIR: str = "patch training datasets and pipeline validation data"

# Ingestion settings
TRAIN_PAIR_FRACTION: float = 0.8
ONE_DATASET_VERSION_PER_BATCH: bool = True

# Stage1_ingestor_core config (overrides passed into stage1_ingestor_core)
AUTO_LOAD_SIBLING_CLASS_CSV: bool = True
DATASET_VERSION_COPY_MODE: str = "hardlink"  # "hardlink" | "copy"

# Auto background patch generation
AUTO_GENERATE_BACKGROUND_PATCHES: bool = True
AUTO_BACKGROUND_TO_FIREFLY_RATIO: float = 10.0
AUTO_BACKGROUND_PATCH_SIZE_PX: int = 10
AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME: int = 10
AUTO_BACKGROUND_MAX_FRAME_SAMPLES: int = 5000
AUTO_BACKGROUND_SEED: int = 1337
AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS: bool = True

# Safety/performance: blob detection can be very slow on high-res videos. If it is too slow,
# stage1_ingestor_core will disable blob detection and fall back to random centers only.
AUTO_BACKGROUND_BLOB_DETECT_SLOW_FRAME_SECONDS: float = 2.0
AUTO_BACKGROUND_BLOB_DETECT_DISABLE_AFTER_SLOW_FRAMES: int = 1

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

# Optional: also sync held-out validation videos to the stable store under ROOT/validation videos/<species>/...
AUTO_SYNC_VALIDATION_VIDEOS_STORE: bool = True
VALIDATION_VIDEOS_STORE_MODE: str = "hardlink"  # "hardlink" | "copy"

# If True, only ingest species that do not already have a single-species dataset version dir.
# If False (recommended), also ingest *new videos* added later to an existing species folder.
ONLY_INGEST_NOT_YET_INGESTED: bool = False

# Optional: restrict to certain species folder names (as they appear under RAW_VIDEOS_ROOT)
ONLY_RAW_SPECIES_DIRS: Sequence[str] = ()

# Optional: override raw folder name -> species token used by the pipeline.
# Overrides may be keyed by either:
#   - the full raw folder name (e.g. "night_Photinus Knulli"), or
#   - the base species name without the day/night prefix (e.g. "Photinus Knulli")
# IMPORTANT: species token must NOT contain underscores because stage1_ingestor_core parses
# identity from CSV filenames split on underscores.
SPECIES_NAME_OVERRIDES: Dict[str, str] = {
    # Example:
    # "Photinus Knulli": "photinus-knulli",
    # "Photinus Knulli": "s1806ik",
}

# Change log
ENABLE_CODEX_CHANGE_LOG: bool = True

# If True, do everything except the actual stage1 ingestion writes.
DRY_RUN: bool = False

# -----------------------------------------------------------------------------


def _parse_video_type_and_base_species_name(raw_dir_name: str) -> Tuple[str, str]:
    """
    Extract (video_type, base_species_name) from a raw species folder name.

    Expected convention (case-insensitive):
      - day_<species name>
      - night_<species name>
    """
    raw_dir_name = str(raw_dir_name or "").strip()
    m = re.match(r"^(day|night)[_\-\s]+(.+)$", raw_dir_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Raw species dir must start with 'day_' or 'night_' (got {raw_dir_name!r})")
    video_type = str(m.group(1) or "").strip().lower()
    base = str(m.group(2) or "").strip()
    if not base:
        raise ValueError(f"Raw species dir missing species name after prefix (got {raw_dir_name!r})")
    return video_type, base


def _normalize_species_token(raw_dir_name: str, *, base_name: str | None = None) -> str:
    raw_dir_name = str(raw_dir_name or "").strip()
    base_name = str(base_name or "").strip() or raw_dir_name

    # Allow overrides keyed by either:
    #   - the full raw dir name (e.g. "night_Photinus Knulli"), or
    #   - the base species name without the day/night prefix (e.g. "Photinus Knulli")
    override_key: str | None = None
    if raw_dir_name in SPECIES_NAME_OVERRIDES and str(SPECIES_NAME_OVERRIDES[raw_dir_name]).strip():
        override_key = raw_dir_name
    elif base_name in SPECIES_NAME_OVERRIDES and str(SPECIES_NAME_OVERRIDES[base_name]).strip():
        override_key = base_name

    if override_key is not None:
        token = str(SPECIES_NAME_OVERRIDES[override_key]).strip()
        token = token.lower()
        token = re.sub(r"[_\s]+", "-", token)
        token = re.sub(r"[^a-z0-9-]+", "-", token)
        token = re.sub(r"-+", "-", token).strip("-")
        if "_" in token or not token:
            raise ValueError(f"Invalid override species token for {override_key!r}: {SPECIES_NAME_OVERRIDES[override_key]!r}")
        return token

    s = base_name.lower()
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        raise ValueError(f"Could not normalize species folder name: {base_name!r}")
    if "_" in s:
        # Guardrail: stage1_ingestor_core identity parsing breaks if species token contains underscores.
        s = s.replace("_", "-")
    return s


def _iter_species_dirs(raw_root: Path) -> List[Path]:
    raw_root = Path(raw_root).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    if not raw_root.is_dir():
        raise NotADirectoryError(raw_root)

    dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    only = {str(x).strip() for x in ONLY_RAW_SPECIES_DIRS if str(x).strip()}
    if only:
        dirs = [d for d in dirs if d.name in only]
    return dirs


def _dir_has_any_files(d: Path) -> bool:
    """
    Return True if directory tree contains at least one regular file.
    Used to ignore placeholder/empty species folders.
    """
    d = Path(d)
    try:
        if not d.exists() or (not d.is_dir()):
            return False
    except Exception:
        return False

    try:
        stack = [d]
        while stack:
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    for e in it:
                        try:
                            if e.is_file(follow_symlinks=False):
                                return True
                            if e.is_dir(follow_symlinks=False):
                                stack.append(Path(e.path))
                        except Exception:
                            continue
            except Exception:
                continue
    except Exception:
        # If we can't scan, assume there might be data and let downstream discovery decide.
        return True
    return False


def _species_already_ingested(*, data_root: Path, species_token: str) -> bool:
    import orchestrator as orch  # local import (same folder)

    sp_root = data_root / "Integrated_prototype_datasets" / "single species datasets" / str(species_token)
    if not sp_root.exists():
        return False
    latest = orch._latest_version_dir(sp_root)  # type: ignore[attr-defined]
    return latest is not None


def _ensure_scaffolds(*, root: Path, data_root: Path) -> Path:
    """
    Match the integrated orchestrator's required folder scaffolds under the chosen ROOT.
    Returns the validation-video store root.
    """
    import orchestrator as orch  # local import (same folder)

    root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    (data_root / "batch_exports").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_datasets" / "integrated pipeline datasets").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_datasets" / "single species datasets").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_validation_datasets" / "combined species folder").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_validation_datasets" / "individual species folder").mkdir(parents=True, exist_ok=True)

    return orch._ensure_validation_video_store(root)  # type: ignore[attr-defined]


def _scaler_overrides() -> Dict[str, Any]:
    return {
        "VERSION_COPY_MODE": str(DATASET_VERSION_COPY_MODE),
        "AUTO_GENERATE_BACKGROUND_PATCHES": bool(AUTO_GENERATE_BACKGROUND_PATCHES),
        "AUTO_BACKGROUND_TO_FIREFLY_RATIO": float(AUTO_BACKGROUND_TO_FIREFLY_RATIO),
        "AUTO_BACKGROUND_PATCH_SIZE_PX": int(AUTO_BACKGROUND_PATCH_SIZE_PX),
        "AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME": int(AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME),
        "AUTO_BACKGROUND_MAX_FRAME_SAMPLES": int(AUTO_BACKGROUND_MAX_FRAME_SAMPLES),
        "AUTO_BACKGROUND_SEED": int(AUTO_BACKGROUND_SEED),
        "AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS": bool(AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS),
        "AUTO_BACKGROUND_BLOB_DETECT_SLOW_FRAME_SECONDS": float(AUTO_BACKGROUND_BLOB_DETECT_SLOW_FRAME_SECONDS),
        "AUTO_BACKGROUND_BLOB_DETECT_DISABLE_AFTER_SLOW_FRAMES": int(AUTO_BACKGROUND_BLOB_DETECT_DISABLE_AFTER_SLOW_FRAMES),
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


def _ingest_one_species(
    *,
    root: Path,
    data_root: Path,
    validation_videos_store_root: Path,
    raw_species_dir: Path,
    species_token: str,
    time_of_day: str,
    pairs: Sequence[stage1_ingestor.ObservedPair],
    dry_run: bool,
) -> Dict[str, Any]:
    import orchestrator as orch  # local import (same folder)

    time_of_day = str(time_of_day or "").strip().lower()
    if time_of_day not in {"day_time", "night_time"}:
        raise ValueError(f"time_of_day must be 'day_time' or 'night_time' (got {time_of_day!r})")

    pairs = list(pairs)
    if not pairs:
        raise SystemExit(f"No pairs to ingest for species={species_token} dir={raw_species_dir}")
    train_pairs, val_pairs = stage1_ingestor.split_pairs_train_vs_val(pairs, train_fraction=float(TRAIN_PAIR_FRACTION))

    print(
        "[ingestor-only] Discovered pairs:",
        f"species={species_token} time_of_day={time_of_day} dir={raw_species_dir} "
        f"total={len(pairs)} train={len(train_pairs)} val={len(val_pairs)}",
    )

    if AUTO_SYNC_VALIDATION_VIDEOS_STORE and val_pairs:
        orch._sync_validation_videos_for_pairs(  # type: ignore[attr-defined]
            pairs=val_pairs,
            species_name=species_token,
            store_root=validation_videos_store_root,
            mode=str(VALIDATION_VIDEOS_STORE_MODE),
            dry_run=bool(dry_run),
        )

    # Dataset roots produced by stage1_ingestor_core.
    integrated_root = data_root / "Integrated_prototype_datasets" / "integrated pipeline datasets"
    single_root = data_root / "Integrated_prototype_datasets" / "single species datasets" / species_token
    validation_combined_root = data_root / "Integrated_prototype_validation_datasets" / "combined species folder"
    validation_species_root = data_root / "Integrated_prototype_validation_datasets" / "individual species folder" / species_token

    # Stage CSVs into canonical names expected by stage1_ingestor_core.
    staging_dir = (
        data_root
        / "batch_exports"
        / "ingestor_only_observed_dir_staging"
        / f"{orch._run_tag()}__{orch._safe_name(raw_species_dir.name)}"  # type: ignore[attr-defined]
    )

    staged_pairs = stage1_ingestor.stage_pairs_for_ingestor(
        pairs,
        species_name=species_token,
        staging_dir=staging_dir,
        time_of_day=time_of_day,
        dry_run=bool(dry_run),
    )
    staged_by_video: Dict[str, stage1_ingestor.StagedPair] = {sp.pair.video_name: sp for sp in staged_pairs}

    overrides = _scaler_overrides()

    batch_integrated_target_ver: Path | None = None
    batch_single_target_ver: Path | None = None
    batch_val_comb_target_ver: Path | None = None
    batch_val_species_target_ver: Path | None = None

    if ONE_DATASET_VERSION_PER_BATCH:
        batch_integrated_target_ver = orch._next_version_dir(integrated_root)  # type: ignore[attr-defined]
        batch_single_target_ver = orch._next_version_dir(single_root)  # type: ignore[attr-defined]
        if val_pairs:
            batch_val_comb_target_ver = orch._next_version_dir(validation_combined_root)  # type: ignore[attr-defined]
            batch_val_species_target_ver = orch._next_version_dir(validation_species_root)  # type: ignore[attr-defined]

        print(
            "[ingestor-only] Batch ingest versions:",
            f"integrated={batch_integrated_target_ver.name} single_species={batch_single_target_ver.name}"
            + (
                f" validation_combined={batch_val_comb_target_ver.name} validation_species={batch_val_species_target_ver.name}"
                if val_pairs and batch_val_comb_target_ver is not None and batch_val_species_target_ver is not None
                else ""
            ),
        )

        # TRAIN ingestion (append into a single target version; finalize split once at the end).
        for i, p in enumerate(train_pairs):
            sp = staged_by_video[p.video_name]
            finalize = (i == (len(train_pairs) - 1))
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
                scaler_overrides=overrides,
                dry_run=bool(dry_run),
            )

        # FINAL-VALIDATION ingestion (append into a single target validation version).
        for p in val_pairs:
            sp = staged_by_video[p.video_name]
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
                scaler_overrides=overrides,
                dry_run=bool(dry_run),
            )
    else:
        # Legacy behavior: one dataset version per video ingested.
        train_video_names = {p.video_name for p in train_pairs}
        val_video_names = {p.video_name for p in val_pairs}
        for p in pairs:
            sp = staged_by_video[p.video_name]
            if p.video_name in train_video_names:
                stage1_ingestor.run_ingestor_core(
                    annotations_csv=sp.staged_firefly_csv,
                    video_path=p.video_path,
                    data_root=data_root,
                    train_fraction=1.0,
                    train_val_seed=1337,
                    auto_load_sibling=bool(AUTO_LOAD_SIBLING_CLASS_CSV),
                    scaler_overrides=overrides,
                    dry_run=bool(dry_run),
                )
            elif p.video_name in val_video_names:
                stage1_ingestor.run_ingestor_core(
                    annotations_csv=sp.staged_firefly_csv,
                    video_path=p.video_path,
                    data_root=data_root,
                    train_fraction=0.0,
                    train_val_seed=1337,
                    auto_load_sibling=False,
                    scaler_overrides=overrides,
                    dry_run=bool(dry_run),
                )

    # Return latest versions for logging/visibility.
    integrated_ver = orch._latest_version_dir(integrated_root)  # type: ignore[attr-defined]
    single_ver = orch._latest_version_dir(single_root)  # type: ignore[attr-defined]
    validation_ver = orch._latest_version_dir(validation_combined_root)  # type: ignore[attr-defined]
    validation_species_ver = orch._latest_version_dir(validation_species_root)  # type: ignore[attr-defined]

    ingested_pairs: List[Dict[str, Any]] = []
    for p in train_pairs:
        ingested_pairs.append(
            {
                "species_token": str(species_token),
                "video_name": str(p.video_name),
                "video_path": str(p.video_path),
                "firefly_csv": str(p.firefly_csv),
                "background_csv": str(p.background_csv) if p.background_csv else None,
                "split": "train",
            }
        )
    for p in val_pairs:
        ingested_pairs.append(
            {
                "species_token": str(species_token),
                "video_name": str(p.video_name),
                "video_path": str(p.video_path),
                "firefly_csv": str(p.firefly_csv),
                "background_csv": str(p.background_csv) if p.background_csv else None,
                "split": "validation",
            }
        )

    return {
        "species": species_token,
        "raw_species_dir": str(raw_species_dir),
        "time_of_day": str(time_of_day),
        "n_pairs_total": int(len(pairs)),
        "n_pairs_train": int(len(train_pairs)),
        "n_pairs_validation": int(len(val_pairs)),
        "ingested_pairs": ingested_pairs,
        "paths": {
            "root": str(root),
            "data_root": str(data_root),
            "integrated_dataset_version": integrated_ver.name if integrated_ver else None,
            "single_species_dataset_version": single_ver.name if single_ver else None,
            "validation_dataset_version": validation_ver.name if validation_ver else None,
            "validation_species_dataset_version": validation_species_ver.name if validation_species_ver else None,
        },
    }


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingestor-only automation (ingest missing videos per species).")
    p.add_argument("--raw-root", type=str, default="", help="Override RAW_VIDEOS_ROOT.")
    p.add_argument("--root", type=str, default="", help="Override ROOT_PATH.")
    p.add_argument("--log-root", type=str, default="", help="Override LOG_ROOT_PATH (where changelog lives).")
    p.add_argument("--dry-run", action="store_true", default=False, help="Do not run actual ingestion.")
    p.add_argument("--only-species", action="append", default=[], help="Restrict to a raw species folder name (repeatable).")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    raw_root_arg = str(args.raw_root or "").strip() or str(RAW_VIDEOS_ROOT or "").strip()
    root_arg = str(args.root or "").strip() or str(ROOT_PATH or "").strip()
    log_root_arg = str(args.log_root or "").strip() or str(LOG_ROOT_PATH or "").strip()
    dry_run = bool(DRY_RUN) or bool(args.dry_run)

    if not raw_root_arg:
        print("Set RAW_VIDEOS_ROOT at top of file or pass --raw-root.", file=sys.stderr)
        return 2
    if not root_arg:
        print("Set ROOT_PATH at top of file or pass --root.", file=sys.stderr)
        return 2
    if not log_root_arg:
        print("Set LOG_ROOT_PATH at top of file or pass --log-root.", file=sys.stderr)
        return 2

    raw_root = Path(raw_root_arg).expanduser().resolve()
    root = Path(root_arg).expanduser().resolve()
    log_root = Path(log_root_arg).expanduser().resolve()
    data_root = root / str(DATA_SUBDIR)

    # Optional CLI override for ONLY_RAW_SPECIES_DIRS
    cli_only = {str(x).strip() for x in (args.only_species or []) if str(x).strip()}
    global ONLY_RAW_SPECIES_DIRS
    if cli_only:
        ONLY_RAW_SPECIES_DIRS = tuple(sorted(cli_only))

    validation_videos_store_root = _ensure_scaffolds(root=root, data_root=data_root)

    log_path = (log_root / str(CHANGE_LOG_FILENAME)).expanduser().resolve()
    if not dry_run:
        log_root.mkdir(parents=True, exist_ok=True)

    # Build an ingestion index from the centralized change log.
    ingestion_index = build_ingestion_index(log_path)

    raw_species_dirs = _iter_species_dirs(raw_root)
    if not raw_species_dirs:
        print(f"[ingestor-only] No species folders found under: {raw_root}")
        return 0

    candidates: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for sp_dir in raw_species_dirs:
        if not _dir_has_any_files(sp_dir):
            skipped.append(f"Skipping raw species dir {sp_dir.name!r}: raw species dir is empty")
            continue

        try:
            video_type, base_species_name = _parse_video_type_and_base_species_name(sp_dir.name)
            token = _normalize_species_token(sp_dir.name, base_name=base_species_name)
            time_of_day = f"{video_type}_time"
        except Exception as e:
            skipped.append(f"Skipping raw species dir {sp_dir.name!r}: parse failed: {e}")
            continue

        if ONLY_INGEST_NOT_YET_INGESTED and _species_already_ingested(data_root=data_root, species_token=token):
            skipped.append(f"Skipping {sp_dir.name!r} -> {token}: already ingested (ONLY_INGEST_NOT_YET_INGESTED=True)")
            continue

        try:
            pairs = stage1_ingestor.discover_observed_pairs(sp_dir)
        except SystemExit as e:
            skipped.append(f"Skipping {sp_dir.name!r} -> {token}: discover pairs failed: {e}")
            continue
        except Exception as e:
            skipped.append(f"Skipping {sp_dir.name!r} -> {token}: discover pairs error: {e}")
            continue

        already_ingested = set(ingestion_index.get(token, {}).keys())
        pairs_to_ingest = [p for p in pairs if str(p.video_name) not in already_ingested]
        if not pairs_to_ingest:
            skipped.append(f"Skipping {sp_dir.name!r} -> {token}: all {len(pairs)} video(s) already ingested")
            continue

        candidates.append(
            {
                "raw_species_dir": sp_dir,
                "species_token": token,
                "time_of_day": time_of_day,
                "pairs_total": int(len(pairs)),
                "pairs_to_ingest": list(pairs_to_ingest),
                "already_ingested_video_names": sorted(already_ingested),
            }
        )

    if not candidates:
        print("[ingestor-only] Up to date: no new videos found to ingest (or all species were skipped).")
        for msg in skipped[:50]:
            print(f"[ingestor-only] Note: {msg}")
        return 0

    print(f"[ingestor-only] Found {len(candidates)} species with new videos to ingest. dry_run={dry_run}")
    for c in candidates:
        sp_dir = Path(c["raw_species_dir"])
        token = str(c["species_token"])
        time_of_day = str(c.get("time_of_day") or "")
        print(
            f"  - {sp_dir.name} -> {token} ({time_of_day}): ingest_videos={len(c['pairs_to_ingest'])}/{int(c['pairs_total'])} total",
        )

    results: List[Dict[str, Any]] = []
    for c in candidates:
        sp_dir = Path(c["raw_species_dir"])
        token = str(c["species_token"])
        time_of_day = str(c.get("time_of_day") or "")
        pairs_to_ingest = list(c["pairs_to_ingest"])
        already_ingested_names = list(c.get("already_ingested_video_names") or [])

        print(f"\n[ingestor-only] === Ingesting {sp_dir.name} -> {token} ({time_of_day}) ===")
        print(
            "[ingestor-only] Work:",
            f"total_pairs={int(c['pairs_total'])} already_ingested={len(already_ingested_names)} to_ingest={len(pairs_to_ingest)}",
        )

        cfg = SnapshotConfig(
            root=root,
            scopes=[data_root, validation_videos_store_root] if AUTO_SYNC_VALIDATION_VIDEOS_STORE else [data_root],
        )
        meta = {
            "actor": "ingestor_only",
            "raw_species_dir": str(sp_dir),
            "species_token": str(token),
            "time_of_day": str(time_of_day),
            "dry_run": bool(dry_run),
            "raw_pairs_total": int(c["pairs_total"]),
            "raw_pairs_to_ingest": int(len(pairs_to_ingest)),
            "raw_pairs_already_ingested": int(len(already_ingested_names)),
            # NOTE: we intentionally do not inline all existing ingested history here; only what we ingest now.
        }

        with ChangeLogRun(cfg=cfg, log_path=log_path, meta=meta, enabled=bool(ENABLE_CODEX_CHANGE_LOG)):
            try:
                out = _ingest_one_species(
                    root=root,
                    data_root=data_root,
                    validation_videos_store_root=validation_videos_store_root,
                    raw_species_dir=sp_dir,
                    species_token=token,
                    time_of_day=str(time_of_day),
                    pairs=pairs_to_ingest,
                    dry_run=dry_run,
                )
                meta["ingested_pairs"] = list(out.get("ingested_pairs") or [])
            except SystemExit as e:
                # stage1_ingestor uses SystemExit for "no data" type errors; treat as skip.
                skipped.append(f"Ingest skipped for {sp_dir.name!r} -> {token}: {e}")
                continue
            except Exception as e:
                print(f"[ingestor-only] ERROR ingesting {sp_dir.name} -> {token}: {e}", file=sys.stderr)
                raise

        results.append(out)
        print(
            "[ingestor-only] Done:",
            f"species={token} integrated_ver={out['paths'].get('integrated_dataset_version')} single_ver={out['paths'].get('single_species_dataset_version')}",
        )

    if results:
        print(f"\n[ingestor-only] Completed {len(results)} ingestion job(s).")
        print(f"[ingestor-only] Change log → {log_path}")
    else:
        print("\n[ingestor-only] No ingestion jobs were executed.")
        print(f"[ingestor-only] Change log → {log_path}")

    for msg in skipped[:50]:
        print(f"[ingestor-only] Note: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
