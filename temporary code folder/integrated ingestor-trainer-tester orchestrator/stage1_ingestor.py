#!/usr/bin/env python3
from __future__ import annotations

"""
Stage 1 (Ingestor)
------------------

Helpers for the integrated ingest-train-test orchestrator.

Goal: keep orchestrator.py focused on (1) params and (2) orchestration, while
this module holds the ingestion mechanics:
  - discover (video,csv) pairs under an observed folder
  - split pairs into train vs held-out validation
  - stage CSVs into the canonical naming expected by stage1_ingestor_core
  - run stage1_ingestor_core in-process (driven by globals set here)
"""

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


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


def safe_name(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


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


def discover_observed_pairs(observed_dir: Path) -> List[ObservedPair]:
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
        c_key = c.stem.strip().lower()
        match: str | None = None
        for vk in stem_keys:
            if c_key == vk or c_key.startswith(vk + "_"):
                match = vk
                break
        if match is None:
            unmatched.append(c)
            continue

        vp = videos_by_stem[match]
        video_name = safe_name(vp.stem)
        label = _parse_label_from_csv_name(c, video_key=match)

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


def split_pairs_train_vs_val(pairs: Sequence[ObservedPair], *, train_fraction: float) -> Tuple[List[ObservedPair], List[ObservedPair]]:
    ordered = sorted(pairs, key=lambda p: (p.video_name, str(p.video_path)))
    if not (0.0 < float(train_fraction) < 1.0):
        raise ValueError(f"train_fraction must be between 0 and 1 (got {train_fraction!r})")

    n = len(ordered)
    n_train = int(n * float(train_fraction))
    # Ensure both splits are non-empty when possible.
    if n >= 2:
        n_train = max(1, min(n - 1, n_train))
    return ordered[:n_train], ordered[n_train:]


def stage_pairs_for_ingestor(
    pairs: Sequence[ObservedPair],
    *,
    species_name: str,
    staging_dir: Path,
    time_of_day: str,
    dry_run: bool,
) -> List[StagedPair]:
    """
    stage1_ingestor_core parses identity from CSV filename, so we stage/copy CSVs into a
    canonical naming scheme:
      <video_name>_<species_name>_<day_time|night_time>_<firefly|background>.csv
    """
    staging_dir = Path(staging_dir)
    if not dry_run:
        staging_dir.mkdir(parents=True, exist_ok=True)

    staged: List[StagedPair] = []
    seen: set[str] = set()
    safe_species = safe_name(species_name)
    if not safe_species:
        raise SystemExit(f"Invalid species_name for staging: {species_name!r}")

    for p in pairs:
        base = f"{safe_name(p.video_name)}_{safe_species}_{safe_name(time_of_day)}"
        dst_firefly = staging_dir / f"{base}_firefly.csv"
        if str(dst_firefly) in seen:
            raise SystemExit(f"Staging name collision (firefly): {dst_firefly}")
        seen.add(str(dst_firefly))

        if dry_run:
            print(f"[dry-run] Would stage CSV -> {dst_firefly}  (src={p.firefly_csv})")
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
                print(f"[dry-run] Would stage CSV -> {dst_bg}  (src={p.background_csv})")
            else:
                dst_bg.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p.background_csv, dst_bg)

        staged.append(StagedPair(pair=p, staged_firefly_csv=dst_firefly, staged_background_csv=dst_bg))

    return staged


def run_ingestor_core(
    *,
    annotations_csv: Path,
    video_path: Path,
    data_root: Path,
    train_fraction: float,
    train_val_seed: int,
    auto_load_sibling: bool,
    single_species_target_version_dir: Path | None = None,
    integrated_target_version_dir: Path | None = None,
    validation_combined_target_version_dir: Path | None = None,
    validation_individual_target_version_dir: Path | None = None,
    skip_final_split_rebuild: bool = False,
    scaler_overrides: Dict[str, Any] | None = None,
    dry_run: bool = False,
) -> None:
    if dry_run:
        print(f"[dry-run] Would run stage1_ingestor_core on: {annotations_csv}")
        return

    core_path = (Path(__file__).resolve().parent / "stage1_ingestor_core.py").resolve()
    if not core_path.exists():
        raise FileNotFoundError(core_path)

    import stage1_ingestor_core as ss  # type: ignore

    ss.ANNOTATIONS_CSV_PATH = Path(annotations_csv).expanduser().resolve()
    ss.VIDEO_PATH = Path(video_path).expanduser().resolve()

    ss.DATA_ROOT = Path(data_root).expanduser().resolve()
    ss.BATCH_EXPORT_ROOT = ss.DATA_ROOT / "batch_exports"
    ss.TRAIN_DATASETS_ROOT = ss.DATA_ROOT / "Integrated_prototype_datasets"
    ss.VALIDATION_DATASETS_ROOT = ss.DATA_ROOT / "Integrated_prototype_validation_datasets"

    ss.TRAIN_FRACTION_OF_BATCH = float(train_fraction)
    ss.TRAIN_VAL_SPLIT_SEED = int(train_val_seed)
    ss.AUTO_LOAD_SIBLING_CLASS_CSV = bool(auto_load_sibling)

    ss.SINGLE_SPECIES_TARGET_VERSION_DIR = (
        Path(single_species_target_version_dir).expanduser().resolve() if single_species_target_version_dir else None
    )
    ss.INTEGRATED_TARGET_VERSION_DIR = (
        Path(integrated_target_version_dir).expanduser().resolve() if integrated_target_version_dir else None
    )
    ss.VALIDATION_COMBINED_TARGET_VERSION_DIR = (
        Path(validation_combined_target_version_dir).expanduser().resolve()
        if validation_combined_target_version_dir
        else None
    )
    ss.VALIDATION_INDIVIDUAL_TARGET_VERSION_DIR = (
        Path(validation_individual_target_version_dir).expanduser().resolve()
        if validation_individual_target_version_dir
        else None
    )
    ss.SKIP_FINAL_SPLIT_REBUILD = bool(skip_final_split_rebuild)

    if scaler_overrides:
        for k, v in scaler_overrides.items():
            setattr(ss, str(k), v)

    ss.main()


# Back-compat alias (older orchestrator revisions used this name)
run_species_scaler = run_ingestor_core

# Back-compat alias (older orchestrator revisions used this name)
stage_pairs_for_species_scaler = stage_pairs_for_ingestor
