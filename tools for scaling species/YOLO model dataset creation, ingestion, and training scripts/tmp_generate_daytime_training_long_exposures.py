#!/usr/bin/env python3
from __future__ import annotations

"""
Generate day-pipeline-v3 long-exposure images for Photinus greeni day-time
training videos only.

This script reuses the exact Stage 1 implementation from:
  Pipelines/day time pipeline v3 (yolo + patch classifier ensemble)/stage1_long_exposure.py

The training/inference split is loaded from the persistent raw-video catalog at
VIDEO_CATALOG_PATH, and only the training half is processed here so long-
exposure annotation data does not mix training and held-out inference clips.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple


# =============================================================================
# GLOBALS (edit here if needed)
# =============================================================================

REPO_ROOT: Path = Path("/home/guest/Desktop/arnav's files/firefly pipeline")
DAY_PIPELINE_DIR: Path = REPO_ROOT / "Pipelines" / "day time pipeline v3 (yolo + patch classifier ensemble)"
DAY_PIPELINE_PARAMS_PATH: Path = DAY_PIPELINE_DIR / "params.py"
DAY_PIPELINE_STAGE1_PATH: Path = DAY_PIPELINE_DIR / "stage1_long_exposure.py"

# Save long-exposure PNGs directly under this folder, grouped by video stem.
OUTPUT_ROOT: Path = Path(
    "/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data/generated long exposure images/day_Photinus greeni"
)

# If True, skip an entire video when its output folder already contains PNGs.
SKIP_VIDEOS_WITH_EXISTING_OUTPUTS: bool = True

RAW_VIDEOS_ROOT: Path = Path(
    "/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)"
)
VIDEO_CATALOG_PATH: Path = RAW_VIDEOS_ROOT / "tmp_scaling_species_training_inference_catalog.json"
SPECIES_TOKEN: str = "photinus-greeni"
ROUTE_NAME: str = "day"


def _load_day_pipeline_modules() -> Tuple[ModuleType, ModuleType, ModuleType | None]:
    original_params = sys.modules.get("params")

    params_spec = importlib.util.spec_from_file_location("params", DAY_PIPELINE_PARAMS_PATH)
    if params_spec is None or params_spec.loader is None:
        raise RuntimeError(f"Could not load params module from {DAY_PIPELINE_PARAMS_PATH}")
    params_mod = importlib.util.module_from_spec(params_spec)
    sys.modules["params"] = params_mod
    params_spec.loader.exec_module(params_mod)

    stage1_spec = importlib.util.spec_from_file_location("_day_v3_stage1_long_exposure", DAY_PIPELINE_STAGE1_PATH)
    if stage1_spec is None or stage1_spec.loader is None:
        raise RuntimeError(f"Could not load stage1 module from {DAY_PIPELINE_STAGE1_PATH}")
    stage1_mod = importlib.util.module_from_spec(stage1_spec)
    stage1_spec.loader.exec_module(stage1_mod)
    return params_mod, stage1_mod, original_params


def _restore_params_module(original_params: ModuleType | None) -> None:
    if original_params is None:
        sys.modules.pop("params", None)
    else:
        sys.modules["params"] = original_params


def _configure_stage1(params_mod: ModuleType) -> None:
    params_mod.ROOT = OUTPUT_ROOT
    params_mod.ORIGINAL_VIDEOS_DIR = OUTPUT_ROOT / "original videos"
    params_mod.STAGE1_DIR = OUTPUT_ROOT
    params_mod.MAX_FRAMES = None


def _video_output_dir(video_path: Path) -> Path:
    return OUTPUT_ROOT / video_path.stem


def _existing_png_count(video_path: Path) -> int:
    out_dir = _video_output_dir(video_path)
    if not out_dir.exists():
        return 0
    return sum(1 for _ in out_dir.glob("*.png"))


def _load_video_catalog() -> Dict[str, Any]:
    if not VIDEO_CATALOG_PATH.exists():
        raise FileNotFoundError(VIDEO_CATALOG_PATH)
    data = json.loads(VIDEO_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON in {VIDEO_CATALOG_PATH}")
    return data


def _species_video_entries_from_catalog(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for item in list(catalog.get("videos") or []):
        if str(item.get("species_name") or "").strip() != SPECIES_TOKEN:
            continue
        if str(item.get("route") or "").strip().lower() != ROUTE_NAME:
            continue
        entries.append(item)
    if not entries:
        raise ValueError(
            f"No {ROUTE_NAME} catalog entries found for species={SPECIES_TOKEN!r} in {VIDEO_CATALOG_PATH}"
        )
    return entries


def _training_video_paths_from_catalog(catalog: Dict[str, Any]) -> List[Path]:
    videos: List[Path] = []
    for item in _species_video_entries_from_catalog(catalog):
        if str(item.get("category") or "").strip().lower() != "training":
            continue
        p = Path(str(item.get("video_path") or "")).expanduser()
        if str(p):
            videos.append(p)
    return videos


def _inference_video_stems_from_catalog(catalog: Dict[str, Any]) -> List[str]:
    stems: List[str] = []
    for item in _species_video_entries_from_catalog(catalog):
        if str(item.get("category") or "").strip().lower() != "inference":
            continue
        stem = str(item.get("video_stem") or "").strip()
        if not stem:
            name = str(item.get("video_name") or "").strip()
            if name:
                stem = Path(name).stem
        if stem:
            stems.append(stem)
    return stems


def _validate_video_list(video_paths: List[Path]) -> None:
    missing = [p for p in video_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing configured training videos:\n" + "\n".join(str(p) for p in missing)
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate v3 day-pipeline long-exposure images for day-time training raw videos."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without generating images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N videos from the hardcoded list.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Do not skip videos that already have PNGs under the output root.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    video_catalog = _load_video_catalog()
    training_video_paths = _training_video_paths_from_catalog(video_catalog)
    inference_video_stems = _inference_video_stems_from_catalog(video_catalog)
    _validate_video_list(training_video_paths)
    if not args.dry_run:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    params_mod, stage1_mod, original_params = _load_day_pipeline_modules()
    try:
        _configure_stage1(params_mod)

        interval_frames = getattr(params_mod, "INTERVAL_FRAMES", None)
        long_exposure_mode = getattr(params_mod, "LONG_EXPOSURE_MODE", None)
        max_frames = getattr(params_mod, "MAX_FRAMES", None)

        videos = list(training_video_paths)
        if args.limit is not None:
            videos = videos[: max(0, int(args.limit))]

        print(f"[config] output_root={OUTPUT_ROOT}")
        print(f"[config] video_catalog={VIDEO_CATALOG_PATH}")
        print(f"[config] species={SPECIES_TOKEN} route={ROUTE_NAME}")
        print(f"[config] mode={long_exposure_mode} interval_frames={interval_frames} max_frames={max_frames}")
        print(f"[config] skip_existing={SKIP_VIDEOS_WITH_EXISTING_OUTPUTS and not args.rerun_existing}")
        print(f"[config] heldout_inference_videos={len(inference_video_stems)}")
        print(f"[config] training_videos={len(videos)}")

        summary: List[Dict[str, Any]] = []
        for i, video_path in enumerate(videos, start=1):
            existing_pngs = _existing_png_count(video_path)
            should_skip = (
                SKIP_VIDEOS_WITH_EXISTING_OUTPUTS
                and not args.rerun_existing
                and existing_pngs > 0
            )

            print(f"[run] {i}/{len(videos)} {video_path.name}")
            if should_skip:
                print(f"[skip] existing_pngs={existing_pngs} under {_video_output_dir(video_path)}")
                summary.append(
                    {
                        "video_path": str(video_path),
                        "status": "skipped_existing",
                        "existing_pngs": int(existing_pngs),
                        "output_dir": str(_video_output_dir(video_path)),
                    }
                )
                continue

            if args.dry_run:
                summary.append(
                    {
                        "video_path": str(video_path),
                        "status": "dry_run",
                        "existing_pngs": int(existing_pngs),
                        "output_dir": str(_video_output_dir(video_path)),
                    }
                )
                continue

            out_images = stage1_mod.run_for_video(video_path)
            summary.append(
                {
                    "video_path": str(video_path),
                    "status": "generated",
                    "generated_pngs": int(len(out_images)),
                    "output_dir": str(_video_output_dir(video_path)),
                }
            )

        if args.dry_run:
            print("[done] dry-run complete; no files were generated.")
            return 0

        manifest = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "output_root": str(OUTPUT_ROOT),
            "source_stage1": str(DAY_PIPELINE_STAGE1_PATH),
            "source_params": str(DAY_PIPELINE_PARAMS_PATH),
            "long_exposure_mode": long_exposure_mode,
            "interval_frames": interval_frames,
            "max_frames": max_frames,
            "video_catalog_path": str(VIDEO_CATALOG_PATH),
            "species_token": SPECIES_TOKEN,
            "route": ROUTE_NAME,
            "heldout_inference_video_stems": inference_video_stems,
            "training_day_video_paths": [str(p) for p in videos],
            "results": summary,
        }
        manifest_path = OUTPUT_ROOT / "generation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[done] manifest={manifest_path}")
        return 0
    finally:
        _restore_params_module(original_params)


if __name__ == "__main__":
    raise SystemExit(main())
