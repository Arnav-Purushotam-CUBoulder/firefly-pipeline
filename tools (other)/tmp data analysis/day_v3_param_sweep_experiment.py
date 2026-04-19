#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT_DEFAULT = Path(
    "/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/"
    "tmp_day_night_combo_train_and_infer__20260328__005102"
)
RAW_ROOT_DEFAULT = Path("/mnt/Samsung_SSD_2TB/integrated prototype raw videos")
WORKTREE_DEFAULT = Path("/mnt/Samsung_SSD_2TB/temp to delete/day_v3_rca_sweep_worktree_20260328")
EXPERIMENT_ROOT_DEFAULT = Path("/mnt/Samsung_SSD_2TB/temp to delete/day_v3_rca_experiments_20260328_sweep")
REPORT_DIR_DEFAULT = SCRIPT_DIR / "reports" / "day_v3_param_sweep__20260328"
PATCH_MODEL_PATH = Path(
    "/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/"
    "tmp_day_night_combo_train_and_infer__20260328__004250/models/day/global_all_species.pt"
)
WINDOW_FRAMES_DEFAULT = 200


VARIANTS_BY_SPECIES: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "bicellonycha-wickershamorum": [
        ("baseline", {}),
        ("link25_gap10", {"STAGE3_1_LINK_RADIUS_PX": 25.0, "STAGE3_1_MAX_FRAME_GAP": 10}),
        ("minpts2", {"STAGE3_1_MIN_TRACK_POINTS": 2}),
        ("minfrac0p5", {"STAGE3_1_HILL_MIN_MONOTONIC_FRAC": 0.50}),
        ("range2500", {"STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE": 2500}),
        ("range2000", {"STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE": 2000}),
        (
            "global_fn_v1",
            {
                "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE": 2500,
                "STAGE3_1_HILL_MIN_UP_STEPS": 1,
                "STAGE3_1_HILL_MIN_DOWN_STEPS": 1,
            },
        ),
        (
            "global_fn_v2",
            {
                "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE": 2000,
                "STAGE3_1_HILL_MIN_UP_STEPS": 1,
                "STAGE3_1_HILL_MIN_DOWN_STEPS": 1,
            },
        ),
        (
            "range2000_hill11",
            {
                "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE": 2000,
                "STAGE3_1_HILL_MIN_UP_STEPS": 1,
                "STAGE3_1_HILL_MIN_DOWN_STEPS": 1,
            },
        ),
    ],
    "photinus-acuminatus": [
        ("baseline", {}),
        ("link25_gap10", {"STAGE3_1_LINK_RADIUS_PX": 25.0, "STAGE3_1_MAX_FRAME_GAP": 10}),
        ("minpts2", {"STAGE3_1_MIN_TRACK_POINTS": 2}),
        (
            "global_fn_v1",
            {
                "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE": 2500,
                "STAGE3_1_HILL_MIN_UP_STEPS": 1,
                "STAGE3_1_HILL_MIN_DOWN_STEPS": 1,
            },
        ),
        (
            "global_fn_v2",
            {
                "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE": 2000,
                "STAGE3_1_HILL_MIN_UP_STEPS": 1,
                "STAGE3_1_HILL_MIN_DOWN_STEPS": 1,
            },
        ),
        ("hill1_2", {"STAGE3_1_HILL_MIN_UP_STEPS": 1, "STAGE3_1_HILL_MIN_DOWN_STEPS": 2}),
        ("hill1_1", {"STAGE3_1_HILL_MIN_UP_STEPS": 1, "STAGE3_1_HILL_MIN_DOWN_STEPS": 1}),
        ("hill_off", {"STAGE3_1_HIGHVAR_REQUIRE_HILL_SHAPE": False}),
    ],
    "photuris-bethaniensis": [
        ("baseline", {}),
        ("link25_gap10", {"STAGE3_1_LINK_RADIUS_PX": 25.0, "STAGE3_1_MAX_FRAME_GAP": 10}),
        ("minpts2", {"STAGE3_1_MIN_TRACK_POINTS": 2}),
        (
            "global_fn_v1",
            {
                "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE": 2500,
                "STAGE3_1_HILL_MIN_UP_STEPS": 1,
                "STAGE3_1_HILL_MIN_DOWN_STEPS": 1,
            },
        ),
        (
            "global_fn_v2",
            {
                "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE": 2000,
                "STAGE3_1_HILL_MIN_UP_STEPS": 1,
                "STAGE3_1_HILL_MIN_DOWN_STEPS": 1,
            },
        ),
        ("hill1_2", {"STAGE3_1_HILL_MIN_UP_STEPS": 1, "STAGE3_1_HILL_MIN_DOWN_STEPS": 2}),
        ("hill1_1", {"STAGE3_1_HILL_MIN_UP_STEPS": 1, "STAGE3_1_HILL_MIN_DOWN_STEPS": 1}),
        ("hill_off", {"STAGE3_1_HIGHVAR_REQUIRE_HILL_SHAPE": False}),
    ],
}


@dataclass
class ClipSpec:
    species_name: str
    video_stem: str
    video_path: Path
    gt_csv_path: Path
    start_frame: int
    num_frames: int
    fn_count_in_window: int

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.num_frames - 1

    @property
    def clip_stem(self) -> str:
        return f"{self.video_stem}__clip_t{self.start_frame:06d}_{self.end_frame:06d}"


def _load_fn_details(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _find_raw_pair(raw_root: Path, video_stem: str) -> tuple[Path, Path]:
    videos = sorted(raw_root.rglob(f"{video_stem}.mp4"))
    csvs = sorted(raw_root.rglob(f"{video_stem}.csv"))
    if not videos:
        raise FileNotFoundError(f"Raw video not found for {video_stem}")
    if not csvs:
        raise FileNotFoundError(f"GT CSV not found for {video_stem}")
    return videos[0], csvs[0]


def _best_window(frames: list[int], window: int) -> tuple[int, int]:
    frames = sorted(int(t) for t in frames)
    best_start = frames[0]
    best_count = -1
    j = 0
    for i, t0 in enumerate(frames):
        while j < len(frames) and frames[j] < t0 + window:
            j += 1
        count = j - i
        if count > best_count:
            best_start = t0
            best_count = count
    return best_start, best_count


def choose_representative_clips(
    *,
    fn_details_csv: Path,
    raw_root: Path,
    window_frames: int,
) -> list[ClipSpec]:
    rows = _load_fn_details(fn_details_csv)
    by_species_video: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        by_species_video[(row["species_name"], row["video_stem"])].append(int(float(row["t"])))

    clips: list[ClipSpec] = []
    species_names = sorted({species for species, _ in by_species_video.keys()})
    for species_name in species_names:
        candidates = [
            (video_stem, ts)
            for (species, video_stem), ts in by_species_video.items()
            if species == species_name
        ]
        if not candidates:
            continue
        video_stem, ts = max(candidates, key=lambda item: len(item[1]))
        start_frame, fn_count = _best_window(ts, window_frames)
        video_path, gt_csv_path = _find_raw_pair(raw_root, video_stem)
        clips.append(
            ClipSpec(
                species_name=species_name,
                video_stem=video_stem,
                video_path=video_path,
                gt_csv_path=gt_csv_path,
                start_frame=int(start_frame),
                num_frames=int(window_frames),
                fn_count_in_window=int(fn_count),
            )
        )
    return clips


def _read_gt_rows(gt_csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gt_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _frame_col(fieldnames: list[str]) -> str:
    for name in ("frame", "t"):
        if name in fieldnames:
            return name
    raise ValueError(f"No frame column found in {fieldnames}")


def build_clip_inputs(out_root: Path, clip: ClipSpec) -> tuple[Path, Path, Path]:
    original_dir = out_root / "original videos"
    gt_dir = out_root / "ground truth"
    original_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    out_video = original_dir / f"{clip.clip_stem}.mp4"
    out_gt = gt_dir / f"{clip.clip_stem}.csv"

    cap = cv2.VideoCapture(str(clip.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {clip.video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create {out_video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(clip.start_frame))
    written = 0
    for _ in range(int(clip.num_frames)):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    if written <= 0:
        raise RuntimeError(f"No frames written for clip {clip.video_stem}")

    fieldnames, rows = _read_gt_rows(clip.gt_csv_path)
    frame_key = _frame_col(fieldnames)
    kept_rows: list[dict[str, str]] = []
    start_t = int(clip.start_frame)
    end_t = int(clip.start_frame + clip.num_frames)
    for row in rows:
        try:
            frame_idx = int(float(row[frame_key]))
        except Exception:
            continue
        if start_t <= frame_idx < end_t:
            new_row = dict(row)
            new_row[frame_key] = str(frame_idx - start_t)
            kept_rows.append(new_row)
    with out_gt.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept_rows)
    return out_video, out_gt, out_root


def _run_one_experiment(
    *,
    worktree: Path,
    out_root: Path,
    patch_model_path: Path,
    yolo_model_path: Path,
    overrides: dict[str, Any],
    run_stage6_overlay: bool,
) -> None:
    dayv3_dir = worktree / "day time pipeline v3 (yolo + patch classifier ensemble)"
    wrapper = out_root / "_run_wrapper.py"
    wrapper.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                f"sys.path.insert(0, {str(dayv3_dir)!r})",
                "import params",
                f"root = Path({str(out_root)!r})",
                "params.ROOT = root",
                "params.ORIGINAL_VIDEOS_DIR = root / 'original videos'",
                "params.STAGE1_DIR = root / 'stage1_long_exposure'",
                "params.STAGE2_DIR = root / 'stage2_yolo_detections'",
                "params.STAGE3_DIR = root / 'stage3_patch_classifier'",
                "params.STAGE4_DIR = root / 'stage4_rendering'",
                "params.STAGE5_DIR = root / 'stage5 validation'",
                "params.STAGE6_DIR = root / 'stage6 overlay videos'",
                "params.STAGE7_DIR = root / 'stage7 fn analysis'",
                "params.STAGE8_DIR = root / 'stage8 fp analysis'",
                "params.STAGE9_DIR = root / 'stage9 detection summary'",
                "params.GT_CSV_DIR = root / 'ground truth'",
                "params.GT_CSV_PATH = None",
                "params.RUN_PRE_RUN_CLEANUP = True",
                "params.MAX_FRAMES = None",
                f"params.YOLO_MODEL_WEIGHTS = Path({str(yolo_model_path)!r})",
                f"params.PATCH_MODEL_PATH = Path({str(patch_model_path)!r})",
                "params.STAGE5_MODEL_PATH = params.PATCH_MODEL_PATH",
                f"params.RUN_STAGE6_OVERLAY = {bool(run_stage6_overlay)!r}",
                "params.RUN_STAGE7_FN_ANALYSIS = False",
                "params.RUN_STAGE8_FP_ANALYSIS = False",
                "params.RUN_STAGE9_DETECTION_SUMMARY = False",
                "params.STAGE3_1_PLOT_TRAJECTORY_INTENSITY = False",
                "params.STAGE3_1_PLOT_TRAJECTORY_INTENSITY_HIGHVAR = False",
                "params.STAGE3_2_SAVE_ANNOTATED_CROPS = False",
                "params.STAGE3_2_XYT_EXPORT_DIR = root / 'stage3_2 xyt for 3d reconstruction'",
                f"overrides = {json.dumps(overrides, sort_keys=True)!r}",
                "for key, value in __import__('json').loads(overrides).items():",
                "    setattr(params, key, value)",
                "import orchestrator",
                "raise SystemExit(orchestrator.main([]))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(wrapper)], cwd=str(dayv3_dir), check=True)


def _count_csv_rows(path: Path) -> int:
    with path.open("r", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _first_match(path: Path, pattern: str, *, recursive: bool = False) -> Path:
    matches = sorted(path.rglob(pattern) if recursive else path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {path}")
    return matches[0]


def summarize_experiment(out_root: Path) -> dict[str, Any]:
    original_dir = out_root / "original videos"
    video_path = _first_match(original_dir, "*.mp4")
    clip_stem = video_path.stem
    stage2_csv = _first_match(out_root / "stage2_yolo_detections", "*.csv", recursive=True)
    stage3_video_dir = out_root / "stage3_patch_classifier" / clip_stem
    stage3_all = _first_match(stage3_video_dir, "*_patches_motion_all.csv")
    stage3_sel = _first_match(stage3_video_dir, "*_patches_motion.csv")
    stage32_csv = _first_match(stage3_video_dir / "stage3_2", "*_stage3_2_firefly_background_logits.csv")
    stage5_video_dir = out_root / "stage5 validation" / clip_stem
    thr_dir = _first_match(stage5_video_dir, "thr_10.0px")
    fps_csv = thr_dir / "fps.csv"
    tps_csv = thr_dir / "tps.csv"
    fns_csv = thr_dir / "fns.csv"
    return {
        "clip_stem": clip_stem,
        "stage2_boxes": _count_csv_rows(stage2_csv),
        "stage3_positive_all": _count_csv_rows(stage3_all),
        "stage3_1_selected": _count_csv_rows(stage3_sel),
        "stage3_2_rows": _count_csv_rows(stage32_csv),
        "tp": _count_csv_rows(tps_csv),
        "fp": _count_csv_rows(fps_csv),
        "fn": _count_csv_rows(fns_csv),
        "root": str(out_root),
    }


def _load_params_yolo_model(worktree: Path) -> Path:
    params_py = worktree / "day time pipeline v3 (yolo + patch classifier ensemble)" / "params.py"
    text = params_py.read_text(encoding="utf-8")
    needle = 'YOLO_MODEL_WEIGHTS: Path = Path("'
    idx = text.find(needle)
    if idx < 0:
        raise RuntimeError("Could not locate YOLO_MODEL_WEIGHTS in params.py")
    start = idx + len(needle)
    end = text.find('")', start)
    return Path(text[start:end])


def run_sweep(
    *,
    run_root: Path,
    raw_root: Path,
    worktree: Path,
    experiment_root: Path,
    report_dir: Path,
    patch_model_path: Path,
    window_frames: int,
    cleanup: bool,
    run_stage6_overlay: bool,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    fn_details_csv = SCRIPT_DIR / "reports" / "day_v3_root_cause_analysis__20260328" / "fn_root_cause_details.csv"
    clips = choose_representative_clips(
        fn_details_csv=fn_details_csv,
        raw_root=raw_root,
        window_frames=window_frames,
    )
    yolo_model_path = _load_params_yolo_model(worktree)

    results: list[dict[str, Any]] = []
    for clip in clips:
        variants = VARIANTS_BY_SPECIES.get(clip.species_name, [("baseline", {})])
        for variant_name, overrides in variants:
            variant_root = experiment_root / clip.species_name / variant_name
            if variant_root.exists():
                shutil.rmtree(variant_root)
            build_clip_inputs(variant_root, clip)
            _run_one_experiment(
                worktree=worktree,
                out_root=variant_root,
                patch_model_path=patch_model_path,
                yolo_model_path=yolo_model_path,
                overrides=overrides,
                run_stage6_overlay=run_stage6_overlay,
            )
            summary = summarize_experiment(variant_root)
            summary.update(
                {
                    "species_name": clip.species_name,
                    "source_video_stem": clip.video_stem,
                    "source_video_path": str(clip.video_path),
                    "source_gt_csv_path": str(clip.gt_csv_path),
                    "clip_start_frame": clip.start_frame,
                    "clip_end_frame": clip.end_frame,
                    "clip_num_frames": clip.num_frames,
                    "fn_count_in_selected_window": clip.fn_count_in_window,
                    "variant_name": variant_name,
                    "overrides": overrides,
                }
            )
            results.append(summary)

    summary_json = report_dir / "summary.json"
    summary_csv = report_dir / "metrics.csv"
    payload = {
        "run_root": str(run_root),
        "raw_root": str(raw_root),
        "worktree": str(worktree),
        "experiment_root": str(experiment_root),
        "patch_model_path": str(patch_model_path),
        "yolo_model_path": str(yolo_model_path),
        "window_frames": window_frames,
        "run_stage6_overlay": bool(run_stage6_overlay),
        "results": results,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with summary_csv.open("w", newline="") as f:
        fieldnames = [
            "species_name",
            "variant_name",
            "clip_start_frame",
            "clip_end_frame",
            "stage2_boxes",
            "stage3_positive_all",
            "stage3_1_selected",
            "stage3_2_rows",
            "tp",
            "fp",
            "fn",
            "fn_count_in_selected_window",
            "root",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k) for k in fieldnames})

    if cleanup and experiment_root.exists():
        shutil.rmtree(experiment_root)
        payload["experiment_root_cleaned"] = True
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT_DEFAULT)
    parser.add_argument("--worktree", type=Path, default=WORKTREE_DEFAULT)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT_DEFAULT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR_DEFAULT)
    parser.add_argument("--patch-model-path", type=Path, default=PATCH_MODEL_PATH)
    parser.add_argument("--window-frames", type=int, default=WINDOW_FRAMES_DEFAULT)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--enable-overlay", action="store_true")
    parser.add_argument("--only-species", nargs="*", default=None)
    parser.add_argument("--only-variants", nargs="*", default=None)
    args = parser.parse_args()

    if args.only_species or args.only_variants:
        global VARIANTS_BY_SPECIES
        allowed_species = set(args.only_species or [])
        allowed_variants = set(args.only_variants or [])
        filtered: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for species_name, variants in VARIANTS_BY_SPECIES.items():
            if allowed_species and species_name not in allowed_species:
                continue
            new_variants = [
                (name, overrides)
                for name, overrides in variants
                if (not allowed_variants) or (name in allowed_variants)
            ]
            if new_variants:
                filtered[species_name] = new_variants
        VARIANTS_BY_SPECIES = filtered

    payload = run_sweep(
        run_root=args.run_root,
        raw_root=args.raw_root,
        worktree=args.worktree,
        experiment_root=args.experiment_root,
        report_dir=args.report_dir,
        patch_model_path=args.patch_model_path,
        window_frames=args.window_frames,
        cleanup=args.cleanup,
        run_stage6_overlay=args.enable_overlay,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
