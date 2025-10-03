#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import params
from stage0_cleanup import cleanup_root
from stage1_detect_trajectories import run_for_video as stage1_run
from stage2_patch_classifier import run_for_video as stage2_run
from stage3_gaussian_centroid import run_for_video as stage3_run
from stage4_render import run_for_video as stage4_run


def _print_stage_times(stage_times: dict[str, float]) -> None:
    keys_detect = ["stage1", "stage2", "stage3"]
    s_detect = sum(stage_times.get(k, 0.0) for k in keys_detect)
    print("\nTiming summary:")
    for k in keys_detect:
        print(f"  {k}: {stage_times.get(k, 0.0):.2f}s")
    print(f"  total(1-3): {s_detect:.2f}s")
    print(f"  stage4(render): {stage_times.get('stage4', 0.0):.2f}s")


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
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
        s1_csv = stage1_run(vid)
        stage_times["stage1"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        s2_csv = stage2_run(vid)
        stage_times["stage2"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        s3_csv = stage3_run(vid)
        stage_times["stage3"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        out_vid = stage4_run(vid)
        stage_times["stage4"] = time.perf_counter() - t0

        _print_stage_times(stage_times)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
