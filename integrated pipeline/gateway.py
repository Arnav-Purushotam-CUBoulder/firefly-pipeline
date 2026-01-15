#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------------------------------------
# User-configurable globals (defaults; CLI args can override)
# -----------------------------------------------------------------------------

# Path to a single video file OR a folder of videos.
INPUT_PATH: str | Path = '/Volumes/DL Project SSD/integrated prototype data/testing gateway/input videos'

# Base output folder for the integrated gateway.
# If provided, pipeline outputs are written under:
#   <OUTPUT_ROOT>/<DAY_OUTPUT_SUBDIR>/...
#   <OUTPUT_ROOT>/<NIGHT_OUTPUT_SUBDIR>/...
# This overrides output paths in the pipeline param files (ROOT + derived dirs).
OUTPUT_ROOT: str | Path = '/Volumes/DL Project SSD/integrated prototype data/testing gateway/output inference data'
DAY_OUTPUT_SUBDIR: str = "day_pipeline_v3"
NIGHT_OUTPUT_SUBDIR: str = "night_time_pipeline"

# Gateway-level overrides applied to both pipelines (when videos are routed through this gateway)
FORCE_ALL_FRAMES: bool = True           # overrides MAX_FRAMES -> None
FORCE_START_FROM_FRAME_0: bool = True  # overrides start/offset params -> 0 when present

# Average brightness (0–255) computed from the first BRIGHTNESS_NUM_FRAMES frames.
# brightness < threshold  -> night pipeline
# brightness >= threshold -> day pipeline
BRIGHTNESS_THRESHOLD: float = 10.0
BRIGHTNESS_NUM_FRAMES: int = 10

# Folder ingestion behavior
RECURSIVE: bool = False

# If True, only print routing decisions (do not run pipelines).
DRY_RUN: bool = False

# Max videos to process concurrently (each video runs in its own subprocess).
MAX_CONCURRENT_VIDEOS: int = 2


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


def _avg_brightness_first_frames(video_path: Path, *, num_frames: int) -> float:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenCV is required for brightness routing. Install `opencv-python` (or ensure cv2 is importable)."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    brightness_vals: list[float] = []
    try:
        for _ in range(max(1, int(num_frames))):
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_vals.append(float(gray.mean()))
    finally:
        cap.release()

    if not brightness_vals:
        raise RuntimeError(f"Could not read any frames from: {video_path}")

    return sum(brightness_vals) / len(brightness_vals)


def _run_subprocess_python(code: str, *, cwd: Path) -> int:
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(cwd))
    return int(proc.returncode)


def _start_subprocess_python(code: str, *, cwd: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code], cwd=str(cwd))


def _day_pipeline_code(video_path: Path, *, output_root: Path | None, force_no_cleanup: bool) -> str:
    return "\n".join(
        [
            "from pathlib import Path",
            "import params",
            f"output_root = Path({repr(str(output_root))}).expanduser().resolve() if {output_root is not None} else None",
            f"force_no_cleanup = {bool(force_no_cleanup)}",
            f"force_all_frames = {bool(FORCE_ALL_FRAMES)}",
            f"force_start_from_0 = {bool(FORCE_START_FROM_FRAME_0)}",
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
            "if force_all_frames:",
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


def _night_pipeline_code(video_path: Path, *, output_root: Path | None, force_no_cleanup: bool) -> str:
    return "\n".join(
        [
            "from pathlib import Path",
            "import pipeline_params as pp",
            f"output_root = Path({repr(str(output_root))}).expanduser().resolve() if {output_root is not None} else None",
            f"force_no_cleanup = {bool(force_no_cleanup)}",
            f"force_all_frames = {bool(FORCE_ALL_FRAMES)}",
            f"force_start_from_0 = {bool(FORCE_START_FROM_FRAME_0)}",
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
            "if force_all_frames:",
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
    force_no_cleanup: bool,
) -> int:
    code = _day_pipeline_code(video_path, output_root=output_root, force_no_cleanup=force_no_cleanup)
    return _run_subprocess_python(code, cwd=pipeline.day_dir)


def _run_night_pipeline(
    video_path: Path,
    *,
    pipeline: PipelinePaths,
    output_root: Path | None,
    force_no_cleanup: bool,
) -> int:
    code = _night_pipeline_code(video_path, output_root=output_root, force_no_cleanup=force_no_cleanup)
    return _run_subprocess_python(code, cwd=pipeline.night_dir)


def _start_day_pipeline(
    video_path: Path,
    *,
    pipeline: PipelinePaths,
    output_root: Path | None,
    force_no_cleanup: bool,
) -> subprocess.Popen:
    code = _day_pipeline_code(video_path, output_root=output_root, force_no_cleanup=force_no_cleanup)
    return _start_subprocess_python(code, cwd=pipeline.day_dir)


def _start_night_pipeline(
    video_path: Path,
    *,
    pipeline: PipelinePaths,
    output_root: Path | None,
    force_no_cleanup: bool,
) -> subprocess.Popen:
    code = _night_pipeline_code(video_path, output_root=output_root, force_no_cleanup=force_no_cleanup)
    return _start_subprocess_python(code, cwd=pipeline.night_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Routes videos to day/night pipelines based on brightness.")
    p.add_argument("--input", type=str, default=str(INPUT_PATH) if INPUT_PATH else "", help="Video file or folder.")
    p.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUT_ROOT) if OUTPUT_ROOT else "",
        help="Base output folder for pipeline outputs (written into day/night subfolders).",
    )
    p.add_argument("--threshold", type=float, default=float(BRIGHTNESS_THRESHOLD), help="Brightness threshold (0–255).")
    p.add_argument("--frames", type=int, default=int(BRIGHTNESS_NUM_FRAMES), help="Frames to sample for brightness.")
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=int(MAX_CONCURRENT_VIDEOS),
        help="Max number of videos to process concurrently.",
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

    try:
        for video in videos:
            try:
                b = _avg_brightness_first_frames(video, num_frames=int(args.frames))
            except Exception as exc:
                print(f"[gateway] ERROR reading {video}: {exc}")
                failures.append((video, 2))
                continue

            route = "night" if b < float(args.threshold) else "day"
            print(f"[gateway] {video.name}  brightness={b:.2f}  ->  {route}")

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
                        force_no_cleanup=force_no_cleanup,
                    )
                    night_cleanup_done = True
                else:
                    day_root = (out_base / DAY_OUTPUT_SUBDIR) if out_base is not None else None
                    force_no_cleanup = (max_concurrent > 1) or day_cleanup_done
                    proc = _start_day_pipeline(
                        video,
                        pipeline=pipeline,
                        output_root=day_root,
                        force_no_cleanup=force_no_cleanup,
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
