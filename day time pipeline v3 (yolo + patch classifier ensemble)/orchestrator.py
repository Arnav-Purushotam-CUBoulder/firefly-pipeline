#!/usr/bin/env python3
from __future__ import annotations

import sys
import time

import params
from stage0_cleanup import cleanup_root
from stage1_long_exposure import run_for_video as stage1_run
from stage2_yolo_detect import run_for_video as stage2_run


def _print_stage_times(stage_times: dict[str, float]) -> None:
    keys = ["stage1", "stage2"]
    print("\nTiming summary:")
    for k in keys:
        print(f"  {k}: {stage_times.get(k, 0.0):.2f}s")
    print(f"  total: {sum(stage_times.get(k, 0.0) for k in keys):.2f}s")


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    _ = argv  # unused for now

    params.ROOT.mkdir(parents=True, exist_ok=True)

    if params.RUN_PRE_RUN_CLEANUP:
        print("[orchestrator] Running Stage 0 cleanup…")
        cleanup_root(verbose=True)

    videos = params.list_videos()
    if not videos:
        print(f"No videos found. Place input files in: {params.ORIGINAL_VIDEOS_DIR}")
        return 1

    print(f"Found {len(videos)} video(s) in {params.ORIGINAL_VIDEOS_DIR}")

    for vid in videos:
        print(f"\n=== Processing: {vid.name} ===")
        stage_times: dict[str, float] = {}

        t0 = time.perf_counter()
        out_images = stage1_run(vid)
        stage_times["stage1"] = time.perf_counter() - t0
        print(f"Stage1  Time: {stage_times['stage1']:.2f}s (images: {len(out_images)})")

        t0 = time.perf_counter()
        s2_csv = stage2_run(vid)
        stage_times["stage2"] = time.perf_counter() - t0
        print(f"Stage2  Time: {stage_times['stage2']:.2f}s (csv: {s2_csv.name})")

        _print_stage_times(stage_times)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

