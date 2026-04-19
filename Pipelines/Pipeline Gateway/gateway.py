#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------------------------------------
# User-configurable globals (defaults; CLI args can override)
# -----------------------------------------------------------------------------

# Path to a single video file OR a folder of videos.
# Leave empty by default so direct gateway runs require an explicit current input
# path instead of silently using a stale historical location.
INPUT_PATH: str | Path = ""

# Base output folder for the integrated gateway.
# If provided, pipeline outputs are written under:
#   <OUTPUT_ROOT>/<DAY_OUTPUT_SUBDIR>/...
#   <OUTPUT_ROOT>/<NIGHT_OUTPUT_SUBDIR>/...
# This overrides output paths in the pipeline param files (ROOT + derived dirs).
OUTPUT_ROOT: str | Path = '/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data'
DAY_OUTPUT_SUBDIR: str = "day_pipeline_v3"
NIGHT_OUTPUT_SUBDIR: str = "night_time_pipeline"

# Gateway-level overrides applied to both pipelines (when videos are routed through this gateway)
FORCE_ALL_FRAMES: bool = True           # overrides MAX_FRAMES -> None
FORCE_START_FROM_FRAME_0: bool = True  # overrides start/offset params -> 0 when present

# Route selection (brightness routing is disabled).
# Priority:
#   1) --route-override (CLI)
#   2) day/night tokens in the video path
#   3) ROUTE_DEFAULT unless REQUIRE_EXPLICIT_ROUTE=True
ROUTE_NAME_HINT_DAY_TOKENS: tuple[str, ...] = ("day", "daytime", "day_time")
ROUTE_NAME_HINT_NIGHT_TOKENS: tuple[str, ...] = ("night", "nighttime", "night_time")
ROUTE_DEFAULT: str = "night"
REQUIRE_EXPLICIT_ROUTE: bool = False

# Folder ingestion behavior
RECURSIVE: bool = False

# If True, only print routing decisions (do not run pipelines).
DRY_RUN: bool = False

# Max videos to process concurrently (each video runs in its own subprocess).
MAX_CONCURRENT_VIDEOS: int = 2

# Optional model overrides (passed into the pipelines; only applied when set).
# - For day pipeline: overrides params.PATCH_MODEL_PATH (+ STAGE5_MODEL_PATH for FN scoring)
#   and params.YOLO_MODEL_WEIGHTS.
# - For night pipeline: overrides pipeline_params.CNN_MODEL_PATH (+ STAGE9_MODEL_PATH).
DAY_PATCH_MODEL_PATH: str | Path = ""
DAY_YOLO_MODEL_PATH: str | Path = ""
NIGHT_CNN_MODEL_PATH: str | Path = ""

# If True, force pipeline validation/test stages on (Stage5+ for day, Stage9+ for night).
FORCE_TESTS: bool = False


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class PipelinePaths:
    day_dir: Path
    night_dir: Path


@dataclass(frozen=True)
class RunningJob:
    video_path: Path
    route: str
    proc: subprocess.Popen


def _get_pipeline_paths() -> PipelinePaths:
    here = Path(__file__).resolve()
    test1_dir = here.parents[1]
    day_dir = test1_dir / "day time pipeline v3 (yolo + patch classifier ensemble)"
    night_dir = test1_dir / "night_time_pipeline"

    if not (day_dir / "orchestrator.py").exists():
        raise FileNotFoundError(f"Day pipeline orchestrator not found at: {day_dir / 'orchestrator.py'}")
    if not (night_dir / "orchestrator.py").exists():
        raise FileNotFoundError(
            f"Night pipeline orchestrator not found at: {night_dir / 'orchestrator.py'}"
        )

    return PipelinePaths(day_dir=day_dir, night_dir=night_dir)


def _iter_videos(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    it = input_path.rglob("*") if recursive else input_path.iterdir()
    vids = [p for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(vids)


_ROUTE_WARNED_DEFAULT_DIRS: set[str] = set()


def _route_hint_from_video_path(video_path: Path) -> str | None:
    token_src = str(video_path).lower()
    has_day = any(tok in token_src for tok in ROUTE_NAME_HINT_DAY_TOKENS)
    has_night = any(tok in token_src for tok in ROUTE_NAME_HINT_NIGHT_TOKENS)
    if has_day and has_night:
        raise RuntimeError(f"Conflicting day/night tokens in video path: {video_path}")
    if has_day:
        return "day"
    if has_night:
        return "night"
    return None


def _infer_route_for_video(video_path: Path) -> str:
    hint = _route_hint_from_video_path(video_path)
    if hint is not None:
        return hint
    if REQUIRE_EXPLICIT_ROUTE:
        raise RuntimeError(
            f"No route tokens found in video path: {video_path}. "
            "Pass --route-override or rename path to include day/night token."
        )
    key = str(video_path.parent)
    if key not in _ROUTE_WARNED_DEFAULT_DIRS:
        print(f"[gateway] WARNING: route unresolved for {video_path}; defaulting to '{ROUTE_DEFAULT}'.")
        _ROUTE_WARNED_DEFAULT_DIRS.add(key)
    return str(ROUTE_DEFAULT).strip().lower()


def _parse_frame_like(value: object) -> int | None:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return int(round(float(s)))
    except Exception:
        m = re.search(r"\d+", s)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
    return None


def _max_annotated_t_from_gt_csv(gt_csv: Path) -> int | None:
    try:
        with gt_csv.open(newline="") as f:
            r = csv.DictReader(f)
            fieldnames = list(r.fieldnames or [])
            if not fieldnames:
                return None
            cols = {str(c).strip().lower(): c for c in fieldnames}
            t_col = cols.get("t") or cols.get("frame") or cols.get("frame_idx")
            if not t_col:
                return None
            max_t: int | None = None
            for row in r:
                t = _parse_frame_like(row.get(t_col))
                if t is None:
                    continue
                if max_t is None or t > max_t:
                    max_t = t
            return max_t
    except Exception as exc:
        print(f"[gateway] WARNING: failed reading GT CSV {gt_csv}: {exc}")
        return None


def _max_frames_override_from_gt(video_path: Path, *, route: str, out_base: Path | None) -> int | None:
    if out_base is None:
        return None
    route_root = out_base / (DAY_OUTPUT_SUBDIR if str(route) == "day" else NIGHT_OUTPUT_SUBDIR)
    candidates = [
        route_root / "ground truth" / f"gt_{video_path.stem}.csv",
        route_root / "ground truth" / "gt.csv",
        route_root / "gt.csv",
    ]
    for gt_csv in candidates:
        if not gt_csv.exists():
            continue
        max_t = _max_annotated_t_from_gt_csv(gt_csv)
        if max_t is None:
            continue
        max_frames = int(max_t) + 1
        print(
            f"[gateway] {video_path.name}  gt_max_t={max_t}  ->  max_frames={max_frames} "
            f"(from {gt_csv})"
        )
        return max_frames
    return None


def _run_subprocess_python(code: str, *, cwd: Path) -> int:
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(cwd))
    return int(proc.returncode)


def _start_subprocess_python(code: str, *, cwd: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code], cwd=str(cwd))


def _day_pipeline_code(
    video_path: Path,
    *,
    output_root: Path | None,
    patch_model_path: Path | None,
    day_yolo_model_path: Path | None,
    force_tests: bool,
    force_no_cleanup: bool,
    max_frames_override: int | None,
) -> str:
    return "\n".join(
        [
            "from pathlib import Path",
            "import params",
            f"output_root = Path({repr(str(output_root))}).expanduser().resolve() if {output_root is not None} else None",
            f"patch_model_path = Path({repr(str(patch_model_path))}).expanduser().resolve() if {patch_model_path is not None} else None",
            f"day_yolo_model_path = Path({repr(str(day_yolo_model_path))}).expanduser().resolve() if {day_yolo_model_path is not None} else None",
            f"force_tests = {bool(force_tests)}",
            f"force_no_cleanup = {bool(force_no_cleanup)}",
            f"force_all_frames = {bool(FORCE_ALL_FRAMES)}",
            f"force_start_from_0 = {bool(FORCE_START_FROM_FRAME_0)}",
            f"max_frames_override = {int(max_frames_override)} if {max_frames_override is not None} else None",
            "if output_root is not None:",
            "    root = output_root",
            "    root.mkdir(parents=True, exist_ok=True)",
            "    params.ROOT = root",
            "    params.ORIGINAL_VIDEOS_DIR = root / 'original videos'",
            "    params.STAGE1_DIR = root / 'stage1_long_exposure'",
            "    params.STAGE2_DIR = root / 'stage2_yolo_detections'",
            "    params.STAGE3_DIR = root / 'stage3_patch_classifier'",
            "    params.STAGE4_DIR = root / 'stage4_rendering'",
            "    params.STAGE5_DIR = root / 'stage5 validation'",
            "    params.STAGE6_DIR = root / 'stage6 overlay videos'",
            "    params.STAGE7_DIR = root / 'stage7 fn analysis'",
            "    params.STAGE8_DIR = root / 'stage8 fp analysis'",
            "    params.STAGE9_DIR = root / 'stage9 detection summary'",
            "    params.GT_CSV_DIR = root / 'ground truth'",
            "if patch_model_path is not None:",
            "    params.PATCH_MODEL_PATH = patch_model_path",
            "    if hasattr(params, 'STAGE5_MODEL_PATH'):",
            "        params.STAGE5_MODEL_PATH = patch_model_path",
            "if day_yolo_model_path is not None:",
            "    params.YOLO_MODEL_WEIGHTS = day_yolo_model_path",
            "if max_frames_override is not None:",
            "    params.MAX_FRAMES = int(max_frames_override)",
            "elif force_all_frames:",
            "    params.MAX_FRAMES = None",
            "if force_start_from_0:",
            "    if hasattr(params, 'GT_T_OFFSET'):",
            "        params.GT_T_OFFSET = 0",
            "    for _name in (",
            "        'START_FRAME', 'START_FRAME_IDX', 'START_FRAME_INDEX',",
            "        'FRAME_START', 'START_AT_FRAME', 'FIRST_FRAME', 'FIRST_FRAME_IDX',",
            "    ):",
            "        if hasattr(params, _name):",
            "            setattr(params, _name, 0)",
            "if force_tests:",
            "    for _name in (",
            "        'RUN_STAGE5_VALIDATE','RUN_STAGE6_OVERLAY','RUN_STAGE7_FN_ANALYSIS',",
            "        'RUN_STAGE8_FP_ANALYSIS','RUN_STAGE9_DETECTION_SUMMARY'",
            "    ):",
            "        if hasattr(params, _name):",
            "            setattr(params, _name, True)",
            "if force_no_cleanup:",
            "    params.RUN_PRE_RUN_CLEANUP = False",
            "import orchestrator",
            f"video_path = Path({repr(str(video_path))}).expanduser().resolve()",
            "orig_list_videos = params.list_videos",
            "def _one_video():",
            "    return [video_path]",
            "params.list_videos = _one_video",
            "try:",
            "    rc = orchestrator.main([])",
            "finally:",
            "    params.list_videos = orig_list_videos",
            "raise SystemExit(int(rc))",
        ]
    )


def _night_pipeline_code(
    video_path: Path,
    *,
    output_root: Path | None,
    cnn_model_path: Path | None,
    force_tests: bool,
    force_no_cleanup: bool,
    max_frames_override: int | None,
) -> str:
    return "\n".join(
        [
            "from pathlib import Path",
            "import pipeline_params as pp",
            f"output_root = Path({repr(str(output_root))}).expanduser().resolve() if {output_root is not None} else None",
            f"cnn_model_path = Path({repr(str(cnn_model_path))}).expanduser().resolve() if {cnn_model_path is not None} else None",
            f"force_tests = {bool(force_tests)}",
            f"force_no_cleanup = {bool(force_no_cleanup)}",
            f"force_all_frames = {bool(FORCE_ALL_FRAMES)}",
            f"force_start_from_0 = {bool(FORCE_START_FROM_FRAME_0)}",
            f"max_frames_override = {int(max_frames_override)} if {max_frames_override is not None} else None",
            "if output_root is not None:",
            "    root = output_root",
            "    root.mkdir(parents=True, exist_ok=True)",
            "    pp.ROOT = root",
            "    pp.DIR_ORIG_VIDEOS = root / 'original videos'",
            "    pp.DIR_BGS_VIDEOS = root / 'BS videos'",
            "    pp.DIR_CSV = root / 'csv files'",
            "    pp.DIR_OUT_BGS = root / 'BS initial output annotated videos'",
            "    pp.DIR_OUT_ORIG = root / 'original initial output annotated videos'",
            "    pp.DIR_OUT_ORIG_10 = root / 'original 10px overlay annotated videos'",
            "    pp.DIR_STAGE8_CROPS = root / 'stage8 crops'",
            "    pp.DIR_STAGE9_OUT = root / 'stage9 validation'",
            "    pp.DIR_STAGE10_OUT = root / 'stage10 overlay videos'",
            "    pp.DIR_STAGE8_9_OUT = root / 'stage8.9 gt centroid crops'",
            "    pp.GT_CSV_PATH = root / 'ground truth' / 'gt.csv'",
            "if cnn_model_path is not None:",
            "    pp.CNN_MODEL_PATH = cnn_model_path",
            "    if hasattr(pp, 'STAGE9_MODEL_PATH'):",
            "        pp.STAGE9_MODEL_PATH = cnn_model_path",
            "if max_frames_override is not None:",
            "    pp.MAX_FRAMES = int(max_frames_override)",
            "elif force_all_frames:",
            "    pp.MAX_FRAMES = None",
            "if force_start_from_0:",
            "    if hasattr(pp, 'GT_T_OFFSET'):",
            "        pp.GT_T_OFFSET = 0",
            "    for _name in (",
            "        'START_FRAME', 'START_FRAME_IDX', 'START_FRAME_INDEX',",
            "        'FRAME_START', 'START_AT_FRAME', 'FIRST_FRAME', 'FIRST_FRAME_IDX',",
            "    ):",
            "        if hasattr(pp, _name):",
            "            setattr(pp, _name, 0)",
            "if force_tests:",
            "    for _name in (",
            "        'RUN_STAGE9','RUN_STAGE10','RUN_STAGE11','RUN_STAGE12','RUN_STAGE14'",
            "    ):",
            "        if hasattr(pp, _name):",
            "            setattr(pp, _name, True)",
            "if force_no_cleanup:",
            "    pp.RUN_PRE_RUN_CLEANUP = False",
            "import orchestrator",
            f"video_path = Path({repr(str(video_path))}).expanduser().resolve()",
            "orig_iter = orchestrator._iter_videos",
            "def _one_video(_dir_path):",
            "    return [video_path]",
            "orchestrator._iter_videos = _one_video",
            "try:",
            "    orchestrator.main()",
            "finally:",
            "    orchestrator._iter_videos = orig_iter",
        ]
    )


def _run_day_pipeline(
    video_path: Path,
    *,
    pipeline: PipelinePaths,
    output_root: Path | None,
    patch_model_path: Path | None,
    day_yolo_model_path: Path | None,
    force_tests: bool,
    force_no_cleanup: bool,
    max_frames_override: int | None,
) -> int:
    code = _day_pipeline_code(
        video_path,
        output_root=output_root,
        patch_model_path=patch_model_path,
        day_yolo_model_path=day_yolo_model_path,
        force_tests=force_tests,
        force_no_cleanup=force_no_cleanup,
        max_frames_override=max_frames_override,
    )
    return _run_subprocess_python(code, cwd=pipeline.day_dir)


def _run_night_pipeline(
    video_path: Path,
    *,
    pipeline: PipelinePaths,
    output_root: Path | None,
    cnn_model_path: Path | None,
    force_tests: bool,
    force_no_cleanup: bool,
    max_frames_override: int | None,
) -> int:
    code = _night_pipeline_code(
        video_path,
        output_root=output_root,
        cnn_model_path=cnn_model_path,
        force_tests=force_tests,
        force_no_cleanup=force_no_cleanup,
        max_frames_override=max_frames_override,
    )
    return _run_subprocess_python(code, cwd=pipeline.night_dir)


def _start_day_pipeline(
    video_path: Path,
    *,
    pipeline: PipelinePaths,
    output_root: Path | None,
    patch_model_path: Path | None,
    day_yolo_model_path: Path | None,
    force_tests: bool,
    force_no_cleanup: bool,
    max_frames_override: int | None,
) -> subprocess.Popen:
    code = _day_pipeline_code(
        video_path,
        output_root=output_root,
        patch_model_path=patch_model_path,
        day_yolo_model_path=day_yolo_model_path,
        force_tests=force_tests,
        force_no_cleanup=force_no_cleanup,
        max_frames_override=max_frames_override,
    )
    return _start_subprocess_python(code, cwd=pipeline.day_dir)


def _start_night_pipeline(
    video_path: Path,
    *,
    pipeline: PipelinePaths,
    output_root: Path | None,
    cnn_model_path: Path | None,
    force_tests: bool,
    force_no_cleanup: bool,
    max_frames_override: int | None,
) -> subprocess.Popen:
    code = _night_pipeline_code(
        video_path,
        output_root=output_root,
        cnn_model_path=cnn_model_path,
        force_tests=force_tests,
        force_no_cleanup=force_no_cleanup,
        max_frames_override=max_frames_override,
    )
    return _start_subprocess_python(code, cwd=pipeline.night_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Routes videos to day/night pipelines (name-based + optional override).")
    p.add_argument("--input", type=str, default=str(INPUT_PATH) if INPUT_PATH else "", help="Video file or folder.")
    p.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUT_ROOT) if OUTPUT_ROOT else "",
        help="Base output folder for pipeline outputs (written into day/night subfolders).",
    )
    p.add_argument(
        "--route-override",
        type=str,
        choices=["day", "night"],
        default="",
        help="Force all input videos to this route; disables auto route inference.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Deprecated and ignored (brightness routing disabled).",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Deprecated and ignored (brightness routing disabled).",
    )
    p.add_argument(
        "--day-patch-model",
        type=str,
        default=str(DAY_PATCH_MODEL_PATH) if DAY_PATCH_MODEL_PATH else "",
        help="Override day pipeline patch model (.pt) path.",
    )
    p.add_argument(
        "--day-yolo-model",
        type=str,
        default=str(DAY_YOLO_MODEL_PATH) if DAY_YOLO_MODEL_PATH else "",
        help="Override day pipeline YOLO model (.pt) path.",
    )
    p.add_argument(
        "--night-cnn-model",
        type=str,
        default=str(NIGHT_CNN_MODEL_PATH) if NIGHT_CNN_MODEL_PATH else "",
        help="Override night pipeline CNN model (.pt) path.",
    )
    p.add_argument(
        "--force-tests",
        action="store_true",
        default=bool(FORCE_TESTS),
        help="Force pipeline validation/test stages on.",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=int(MAX_CONCURRENT_VIDEOS),
        help="Max number of videos to process concurrently.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Optional explicit frame bound. When set, pipelines run up to this many frames "
            "(typically last_annotated_t + 1). If omitted, gateway falls back to GT CSV discovery."
        ),
    )
    p.add_argument("--recursive", action="store_true", default=bool(RECURSIVE), help="Recurse into subfolders.")
    p.add_argument("--dry-run", action="store_true", default=bool(DRY_RUN), help="Only print routing decisions.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.input:
        raise SystemExit(
            "Set INPUT_PATH at the top of this file or pass `--input /path/to/video_or_folder`."
        )

    max_concurrent = int(getattr(args, "max_concurrent", 1))
    if max_concurrent < 1:
        raise SystemExit("--max-concurrent must be >= 1")

    pipeline = _get_pipeline_paths()
    input_path = Path(args.input).expanduser().resolve()
    videos = _iter_videos(input_path, recursive=bool(args.recursive))
    if not videos:
        print(f"[gateway] No videos found under: {input_path}")
        return 1

    out_base: Path | None = None
    if args.output_root:
        out_base = Path(args.output_root).expanduser().resolve()
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"[gateway] Output base: {out_base}")
        print(f"[gateway] Day outputs : {out_base / DAY_OUTPUT_SUBDIR}")
        print(f"[gateway] Night outputs: {out_base / NIGHT_OUTPUT_SUBDIR}")

    failures: list[tuple[Path, int]] = []
    day_cleanup_done = False
    night_cleanup_done = False

    running: list[RunningJob] = []

    def _reap_one(*, block: bool) -> None:
        while True:
            for i, job in enumerate(running):
                rc = job.proc.poll()
                if rc is None:
                    continue
                running.pop(i)
                print(f"[gateway] finished {job.video_path.name}  route={job.route}  exit={int(rc)}")
                if int(rc) != 0:
                    failures.append((job.video_path, int(rc)))
                return
            if not block:
                return
            time.sleep(0.2)

    if not bool(args.dry_run) and max_concurrent > 1:
        print(f"[gateway] Max concurrent videos: {max_concurrent}")

    day_patch_model: Path | None = None
    if getattr(args, "day_patch_model", ""):
        day_patch_model = Path(str(args.day_patch_model)).expanduser().resolve()
        if not day_patch_model.exists():
            raise SystemExit(f"--day-patch-model not found: {day_patch_model}")

    day_yolo_model: Path | None = None
    if getattr(args, "day_yolo_model", ""):
        day_yolo_model = Path(str(args.day_yolo_model)).expanduser().resolve()
        if not day_yolo_model.exists():
            raise SystemExit(f"--day-yolo-model not found: {day_yolo_model}")

    night_cnn_model: Path | None = None
    if getattr(args, "night_cnn_model", ""):
        night_cnn_model = Path(str(args.night_cnn_model)).expanduser().resolve()
        if not night_cnn_model.exists():
            raise SystemExit(f"--night-cnn-model not found: {night_cnn_model}")

    force_tests = bool(getattr(args, "force_tests", False))
    route_override = str(getattr(args, "route_override", "") or "").strip().lower() or None
    if route_override is not None and route_override not in {"day", "night"}:
        raise SystemExit("--route-override must be one of: day, night")

    try:
        for video in videos:
            try:
                if route_override is not None:
                    route = route_override
                    print(f"[gateway] {video.name}  route_override={route}")
                else:
                    route = _infer_route_for_video(video)
                    print(f"[gateway] {video.name}  route_from_name={route}")
            except Exception as exc:
                print(f"[gateway] ERROR routing {video}: {exc}")
                failures.append((video, 2))
                continue

            explicit_max_frames = getattr(args, "max_frames", None)
            if explicit_max_frames is not None:
                max_frames_override = int(explicit_max_frames)
                print(
                    f"[gateway] {video.name}  explicit_max_frames={max_frames_override} "
                    "(from --max-frames)"
                )
            else:
                max_frames_override = _max_frames_override_from_gt(video, route=route, out_base=out_base)

            if bool(args.dry_run):
                continue

            while len(running) >= max_concurrent:
                _reap_one(block=True)

            try:
                if route == "night":
                    night_root = (out_base / NIGHT_OUTPUT_SUBDIR) if out_base is not None else None
                    force_no_cleanup = (max_concurrent > 1) or night_cleanup_done
                    proc = _start_night_pipeline(
                        video,
                        pipeline=pipeline,
                        output_root=night_root,
                        cnn_model_path=night_cnn_model,
                        force_tests=force_tests,
                        force_no_cleanup=force_no_cleanup,
                        max_frames_override=max_frames_override,
                    )
                    night_cleanup_done = True
                else:
                    day_root = (out_base / DAY_OUTPUT_SUBDIR) if out_base is not None else None
                    force_no_cleanup = (max_concurrent > 1) or day_cleanup_done
                    proc = _start_day_pipeline(
                        video,
                        pipeline=pipeline,
                        output_root=day_root,
                        patch_model_path=day_patch_model,
                        day_yolo_model_path=day_yolo_model,
                        force_tests=force_tests,
                        force_no_cleanup=force_no_cleanup,
                        max_frames_override=max_frames_override,
                    )
                    day_cleanup_done = True
            except Exception as exc:
                print(f"[gateway] ERROR launching {route} pipeline for {video}: {exc}")
                failures.append((video, 3))
                continue

            running.append(RunningJob(video_path=video, route=route, proc=proc))
            print(f"[gateway] started  {video.name}  route={route}  pid={proc.pid}")
            _reap_one(block=False)

        while running:
            _reap_one(block=True)
    except KeyboardInterrupt:
        print("\n[gateway] Interrupted; terminating running jobs…")
        for job in running:
            try:
                job.proc.terminate()
            except Exception:
                pass
        for job in running:
            try:
                job.proc.wait(timeout=5)
            except Exception:
                pass
        raise

    if failures:
        print("\n[gateway] Done with failures:")
        for p, rc in failures:
            print(f"  - {p} (exit={rc})")
        return 1

    print("\n[gateway] All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
