#!/usr/bin/env python3
from __future__ import annotations

"""
Temporary one-shot runner (safe to delete later).

What it does:
1) Discovers latest single-species final datasets.
2) Splits species into day/night groups from names + explicit route overrides.
3) Builds one combined dataset per route.
4) Trains one day model and one night model.
5) Scans a raw-videos root, splits videos into day/night groups.
6) Runs gateway inference on each video using the route-specific model.

Routing uses folder/file/species names (no brightness filter).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


# =============================================================================
# GLOBALS (edit these if needed)
# =============================================================================

REPO_ROOT: Path = Path("/home/guest/Desktop/arnav's files/firefly pipeline")

# Integrated data roots.
INTEGRATED_OUTER_ROOT: Path = Path("/mnt/Samsung_SSD_2TB/integrated prototype data")
INTEGRATED_INNER_ROOT: Path = INTEGRATED_OUTER_ROOT / "integrated prototype data"
PATCH_DATA_ROOT: Path = INTEGRATED_INNER_ROOT / "patch training datasets and pipeline validation data"

# Raw videos root to infer on (can be overridden via CLI).
RAW_VIDEOS_ROOT: Path = Path("/mnt/Samsung_SSD_2TB/integrated prototype raw videos")

# Final outputs for this temporary run.
RUNS_ROOT: Path = Path("/mnt/Samsung_SSD_2TB/night time pipeline inference output data")
RUN_NAME_PREFIX: str = "tmp_day_night_combo_train_and_infer"

# If False (default), this script skips any raw-video ingestion stage and uses
# already-ingested species datasets under PATCH_DATA_ROOT directly.
# Keep False for your requested flow: train + test only.
RUN_INGESTION_FROM_SCRATCH: bool = False

# Route assignment by species token.
# Used for both training-source grouping and inference-video grouping.
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
# orchestrator._run_gateway still accepts these args.
GATEWAY_BRIGHTNESS_THRESHOLD: float = 10.0
GATEWAY_BRIGHTNESS_FRAMES: int = 5
GATEWAY_MAX_CONCURRENT: int = 1
FORCE_GATEWAY_TESTS: bool = False

# If True, prints actions but does not train or run gateway.
DRY_RUN: bool = False


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
CLASSES = ("firefly", "background")


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


def _latest_version_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    best: tuple[int, str, Path] | None = None
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^v(?P<n>\d+)(?:_|$)", p.name)
        if not m:
            continue
        n = int(m.group("n"))
        key = (n, p.name, p)
        if best is None or key[0] > best[0] or (key[0] == best[0] and key[1] > best[1]):
            best = key
    return best[2] if best else None


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


def _run_inference_for_video(
    *,
    route: str,
    model_path: Path,
    video_path: Path,
    output_root: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    import orchestrator as orch  # type: ignore

    repo_root = Path(orch._REPO_ROOT).resolve()  # type: ignore[attr-defined]
    gateway_path = (repo_root / orch.GATEWAY_REL_PATH).resolve()  # type: ignore[attr-defined]
    if not gateway_path.exists():
        raise FileNotFoundError(gateway_path)

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    try:
        orch._run_gateway(  # type: ignore[attr-defined]
            repo_root=repo_root,
            gateway_path=gateway_path,
            video_path=video_path,
            output_root=output_root,
            day_patch_model=(model_path if route == "day" else None),
            night_cnn_model=(model_path if route == "night" else None),
            thr=float(GATEWAY_BRIGHTNESS_THRESHOLD),
            frames=int(GATEWAY_BRIGHTNESS_FRAMES),
            max_concurrent=int(GATEWAY_MAX_CONCURRENT),
            force_tests=bool(FORCE_GATEWAY_TESTS),
            dry_run=bool(dry_run),
            route_override=str(route),
        )
        status = "ok"
        err: str | None = None
    except subprocess.CalledProcessError as exc:
        status = "failed"
        err = f"gateway_exit={int(exc.returncode)}"

    day_root = output_root / "day_pipeline_v3"
    night_root = output_root / "night_time_pipeline"
    chosen_root = day_root if route == "day" else night_root
    return {
        "video_path": str(video_path),
        "route": route,
        "status": status,
        "error": err,
        "output_root": str(output_root),
        "pipeline_root": str(chosen_root),
        "xywh_csv": str(chosen_root / "csv files" / f"{video_path.stem}.csv"),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporary runner: train day+night combo models + route-based inference.")
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
    run_root = runs_root / f"{RUN_NAME_PREFIX}__{run_tag}"

    combined_dataset_root = run_root / "combined_training_datasets"
    model_root = run_root / "models"
    inference_root = run_root / "inference_outputs"
    metadata_path = run_root / "run_metadata.json"

    print(f"[tmp-run] run_root: {run_root}")
    print(f"[tmp-run] raw_videos_root: {raw_videos_root}")
    print(f"[tmp-run] dry_run={dry_run}")
    print("[tmp-run] ingestion stage: disabled (using existing ingested datasets)")

    sources_by_route = _discover_training_sources_by_route()
    print("[tmp-run] discovered training sources:")
    for route in ("day", "night"):
        srcs = sources_by_route.get(route) or []
        print(f"  - {route}: {len(srcs)} species")
        for s in srcs:
            print(f"      {s.species_name} -> {s.path}")

    dataset_summaries: Dict[str, Dict[str, Any]] = {}
    models_by_route: Dict[str, Path] = {}
    training_metrics: Dict[str, Dict[str, Any]] = {}

    for route in ("day", "night"):
        srcs = sources_by_route.get(route) or []
        if not srcs:
            print(f"[tmp-run] WARNING: no training sources for route={route}; skipping model training.")
            continue

        dst_final = combined_dataset_root / route / "final dataset"
        if not dry_run:
            dst_final.mkdir(parents=True, exist_ok=True)

        summary = _build_combined_dataset(dst_final_dataset_dir=dst_final, sources=srcs, dry_run=dry_run)
        dataset_summaries[route] = summary

        print(f"[tmp-run] combined dataset counts ({route}):")
        for split in SPLITS:
            ff = summary["counts"][split]["firefly"]
            bg = summary["counts"][split]["background"]
            print(f"  - {split}: firefly={ff}, background={bg}")

        if not dry_run:
            _assert_trainable(dst_final)

        model_path = model_root / f"{route}_combo_model.pt"
        metrics_path = model_root / f"{route}_training_metrics.json"
        metrics = _train_model(
            route=route,
            dataset_final_dir=dst_final,
            model_path=model_path,
            metrics_path=metrics_path,
            dry_run=dry_run,
        )
        models_by_route[route] = model_path
        training_metrics[route] = metrics

    known_species = sorted(set(list(ROUTE_BY_SPECIES.keys()) + [s.species_name for v in sources_by_route.values() for s in v]))
    videos = _discover_inference_videos(raw_videos_root, known_species)

    max_videos = int(args.max_videos or 0)
    if max_videos > 0:
        videos = videos[:max_videos]

    print(f"[tmp-run] inference videos discovered: {len(videos)}")
    by_route_counts = {"day": 0, "night": 0}
    for v in videos:
        by_route_counts[v.route] = by_route_counts.get(v.route, 0) + 1
    print(f"[tmp-run] inference split: day={by_route_counts.get('day', 0)} night={by_route_counts.get('night', 0)}")

    inference_results: List[Dict[str, Any]] = []
    for i, v in enumerate(videos, start=1):
        model_path = models_by_route.get(v.route)
        if model_path is None:
            inference_results.append(
                {
                    "video_path": str(v.video_path),
                    "route": v.route,
                    "species_name": v.species_name,
                    "status": "skipped",
                    "error": f"no trained model for route={v.route}",
                }
            )
            continue

        print(f"[tmp-run][infer] {i}/{len(videos)} route={v.route} species={v.species_name} video={v.video_path.name}")
        out_root = inference_root / v.route / v.video_path.stem
        res = _run_inference_for_video(
            route=v.route,
            model_path=model_path,
            video_path=v.video_path,
            output_root=out_root,
            dry_run=dry_run,
        )
        res["species_name"] = v.species_name
        inference_results.append(res)

    record: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "paths": {
            "run_root": str(run_root),
            "raw_videos_root": str(raw_videos_root),
            "combined_dataset_root": str(combined_dataset_root),
            "model_root": str(model_root),
            "inference_root": str(inference_root),
        },
        "routing": {
            "route_by_species": dict(ROUTE_BY_SPECIES),
            "route_by_video_stem": dict(ROUTE_BY_VIDEO_STEM),
            "route_day_hint_tokens": list(ROUTE_NAME_HINT_DAY_TOKENS),
            "route_night_hint_tokens": list(ROUTE_NAME_HINT_NIGHT_TOKENS),
            "route_default": str(ROUTE_DEFAULT),
            "require_explicit_route": bool(REQUIRE_EXPLICIT_ROUTE),
        },
        "training_sources_by_route": {
            r: [{"species": s.species_name, "path": str(s.path)} for s in srcs]
            for r, srcs in sources_by_route.items()
        },
        "dataset_summaries": dataset_summaries,
        "models_by_route": {r: str(p) for r, p in models_by_route.items()},
        "training_metrics": training_metrics,
        "inference_results": inference_results,
    }

    if not dry_run:
        run_root.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(record, indent=2))
        print(f"[tmp-run] metadata: {metadata_path}")

    failed = [r for r in inference_results if str(r.get("status")) not in {"ok", "skipped"}]
    skipped = [r for r in inference_results if str(r.get("status")) == "skipped"]
    if failed:
        print(f"[tmp-run] completed with failures: {len(failed)} video(s) failed.")
        for f in failed[:20]:
            print(f"  - {f.get('video_path')}: {f.get('error')}")
        return 1

    if skipped:
        print(f"[tmp-run] completed with skips: {len(skipped)} video(s) skipped.")
        for s in skipped[:20]:
            print(f"  - {s.get('video_path')}: {s.get('error')}")

    print("[tmp-run] done.")
    print("[tmp-run] trained models:")
    for route in ("day", "night"):
        mp = models_by_route.get(route)
        if mp is not None:
            print(f"  - {route}: {mp}")
    print(f"[tmp-run] inference root: {inference_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
