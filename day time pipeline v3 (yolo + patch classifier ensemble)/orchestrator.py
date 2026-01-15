#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import params
from stage0_cleanup import cleanup_root
from stage1_long_exposure import run_for_video as stage1_run
from stage2_yolo_detect import run_for_video as stage2_run
from stage3_patch_classifier import run_for_video as stage3_run
from stage3_1_trajectory_intensity_selector import run_for_video as stage3_1_run
from stage3_2_gaussian_centroids_and_logits import run_for_video as stage3_2_run
from stage4_render import run_for_video as stage4_run


def _print_stage_times(stage_times: dict[str, float]) -> None:
    keys = ["stage1", "stage2", "stage3", "stage3_1", "stage3_2", "stage4"]
    print("\nTiming summary:")
    for k in keys:
        print(f"  {k}: {stage_times.get(k, 0.0):.2f}s")
    print(f"  total: {sum(stage_times.get(k, 0.0) for k in keys):.2f}s")


def _find_gt_csv_for_video(stem: str) -> Path | None:
    """Locate a GT CSV for a given video stem."""
    gt_dir = getattr(params, "GT_CSV_DIR", None)
    if gt_dir is not None:
        try:
            gt_dir = Path(gt_dir)
            if gt_dir.exists():
                candidates = [
                    gt_dir / f"{stem}.csv",
                    gt_dir / f"{stem}_gt.csv",
                    gt_dir / f"gt_{stem}.csv",
                    gt_dir / "gt.csv",  # common single-video default
                ]
                for c in candidates:
                    if c.exists():
                        return c
                for c in sorted(gt_dir.glob(f"{stem}*.csv")):
                    return c
                for c in sorted(gt_dir.glob(f"gt_{stem}*.csv")):
                    return c
        except Exception:
            pass
    gt_single = getattr(params, "GT_CSV_PATH", None)
    if gt_single is not None:
        try:
            gp = Path(gt_single)
            return gp if gp.exists() else None
        except Exception:
            return None
    return None


def _log_prob(p: float, eps: float = 1e-8) -> float:
    p = float(p)
    p = min(1.0 - eps, max(eps, p))
    return float(math.log(p))


def _build_test_pred_csv(
    *,
    stem: str,
    stage3_csv: Path,
    stage3_2_csv: Path | None,
    out_dir: Path,
    box_w: int,
    box_h: int,
) -> Path:
    """Build a test-suite predictions CSV named `<stem>.csv` under `out_dir`.

    Output schema matches the Stage 5 validator expectations and also
    supports Stage 6 overlays:
      x,y,w,h,t,class,xy_semantics,firefly_logit,background_logit

    Prefers Stage 3.2 logits CSV if available; otherwise synthesizes logits from
    Stage 3 patch confidences.
    """
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{stem}.csv"

    fieldnames = [
        "x",
        "y",
        "w",
        "h",
        "t",
        "class",
        "xy_semantics",
        "firefly_logit",
        "background_logit",
    ]

    if stage3_2_csv is not None and stage3_2_csv.exists():
        with stage3_2_csv.open("r", newline="") as f_in, out_csv.open("w", newline="") as f_out:
            r = csv.DictReader(f_in)
            w = csv.DictWriter(f_out, fieldnames=fieldnames)
            w.writeheader()
            for row in r:
                try:
                    x = float(row["x"])
                    y = float(row["y"])
                    t = int(float(row.get("t", 0)))
                    ff = float(row["firefly_logit"])
                    bg = float(row["background_logit"])
                except Exception:
                    continue
                w.writerow(
                    {
                        "x": x,
                        "y": y,
                        "w": int(box_w),
                        "h": int(box_h),
                        "t": int(t),
                        "class": "firefly",
                        "xy_semantics": "center",
                        "firefly_logit": ff,
                        "background_logit": bg,
                    }
                )
        return out_csv

    # Fallback: build logits from Stage 3 patch CSV (confidence -> log-prob logits)
    with stage3_csv.open("r", newline="") as f_in, out_csv.open("w", newline="") as f_out:
        r = csv.DictReader(f_in)
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        for row in r:
            try:
                t = int(float(row.get("frame_idx", 0)))
                x0 = float(row["x"])
                y0 = float(row["y"])
                ww = float(row.get("w", box_w))
                hh = float(row.get("h", box_h))
                conf = float(row.get("conf", 0.0) or 0.0)
            except Exception:
                continue
            cx = x0 + ww / 2.0
            cy = y0 + hh / 2.0
            w.writerow(
                {
                    "x": float(cx),
                    "y": float(cy),
                    "w": int(box_w),
                    "h": int(box_h),
                    "t": int(t),
                    "class": "firefly",
                    "xy_semantics": "center",
                    "firefly_logit": _log_prob(conf),
                    "background_logit": _log_prob(1.0 - conf),
                }
            )
    return out_csv


def _maybe_run_test_suite(
    *,
    video_path: Path,
    stage3_csv: Path,
    stage3_2_csv: Path | None,
) -> None:
    wants_any = any(
        bool(getattr(params, name, False))
        for name in (
            "RUN_STAGE5_VALIDATE",
            "RUN_STAGE6_OVERLAY",
            "RUN_STAGE7_FN_ANALYSIS",
            "RUN_STAGE8_FP_ANALYSIS",
            "RUN_STAGE9_DETECTION_SUMMARY",
        )
    )
    if not wants_any:
        return

    gt_csv = _find_gt_csv_for_video(video_path.stem)
    if gt_csv is None:
        print(f"[tests] No GT CSV found for '{video_path.stem}'. Skipping test suite.")
        return

    # Local (copied) test-suite modules
    from stage5_validate import stage5_validate_against_gt
    from stage6_overlay_gt_vs_model import overlay_gt_vs_model
    from stage7_fn_analysis import stage7_fn_nearest_tp_analysis
    from stage8_fp_analysis import stage8_fp_nearest_tp_analysis
    from stage9_detection_summary import stage9_generate_detection_summary

    stage5_video_dir = params.STAGE5_DIR / video_path.stem
    pred_csv = _build_test_pred_csv(
        stem=video_path.stem,
        stage3_csv=stage3_csv,
        stage3_2_csv=stage3_2_csv,
        out_dir=stage5_video_dir,
        box_w=int(getattr(params, "STAGE6_GT_BOX_W", getattr(params, "PATCH_SIZE_PX", 10))),
        box_h=int(getattr(params, "STAGE6_GT_BOX_H", getattr(params, "PATCH_SIZE_PX", 10))),
    )

    ran_stage5 = False
    if getattr(params, "RUN_STAGE5_VALIDATE", False):
        stage5_validate_against_gt(
            orig_video_path=video_path,
            pred_csv_path=pred_csv,
            gt_csv_path=gt_csv,
            out_dir=stage5_video_dir,
            dist_thresholds=list(getattr(params, "DIST_THRESHOLDS_PX", [1.0, 2.0, 3.0, 4.0, 5.0])),
            crop_w=int(getattr(params, "STAGE5_CROP_W", getattr(params, "PATCH_SIZE_PX", 10))),
            crop_h=int(getattr(params, "STAGE5_CROP_H", getattr(params, "PATCH_SIZE_PX", 10))),
            gt_t_offset=int(getattr(params, "GT_T_OFFSET", 0)),
            max_frames=getattr(params, "MAX_FRAMES", None),
            only_firefly_rows=bool(getattr(params, "STAGE5_ONLY_FIREFLY_ROWS", True)),
            show_per_frame=bool(getattr(params, "STAGE5_SHOW_PER_FRAME", False)),
            model_path=getattr(params, "STAGE5_MODEL_PATH", None),
            backbone=str(getattr(params, "STAGE5_BACKBONE", "resnet18")),
            imagenet_normalize=bool(getattr(params, "STAGE5_IMAGENET_NORM", False)),
            print_load_status=bool(getattr(params, "STAGE5_PRINT_LOAD_STATUS", True)),
            gt_area_threshold_px=int(getattr(params, "STAGE5_GT_AREA_THRESHOLD_PX", 4)),
            gt_bright_max_threshold=int(getattr(params, "STAGE5_GT_BRIGHT_MAX_THRESHOLD", 50)),
            min_pixel_brightness_to_be_considered_in_area_calculation=int(
                getattr(params, "STAGE5_MIN_PIXEL_BRIGHTNESS_FOR_AREA_CALC", 50)
            ),
            gt_dedupe_dist_threshold_px=float(getattr(params, "STAGE5_GT_DEDUPE_DIST_PX", 2.0)),
        )
        ran_stage5 = True

    if not ran_stage5:
        return

    if getattr(params, "RUN_STAGE7_FN_ANALYSIS", False):
        stage7_video_dir = params.STAGE7_DIR / video_path.stem
        stage7_video_dir.mkdir(parents=True, exist_ok=True)
        stage7_fn_nearest_tp_analysis(
            stage5_video_dir=stage5_video_dir,
            output_dir=stage7_video_dir,
            orig_video_path=video_path,
            box_w=int(getattr(params, "STAGE6_GT_BOX_W", getattr(params, "STAGE5_CROP_W", 10))),
            box_h=int(getattr(params, "STAGE6_GT_BOX_H", getattr(params, "STAGE5_CROP_H", 10))),
            thickness=int(getattr(params, "OVERLAY_BOX_THICKNESS", 1)),
            verbose=True,
        )

    if getattr(params, "RUN_STAGE8_FP_ANALYSIS", False):
        stage8_video_dir = params.STAGE8_DIR / video_path.stem
        stage8_video_dir.mkdir(parents=True, exist_ok=True)
        stage8_fp_nearest_tp_analysis(
            stage5_video_dir=stage5_video_dir,
            output_dir=stage8_video_dir,
            orig_video_path=video_path,
            box_w=int(getattr(params, "STAGE6_GT_BOX_W", getattr(params, "STAGE5_CROP_W", 10))),
            box_h=int(getattr(params, "STAGE6_GT_BOX_H", getattr(params, "STAGE5_CROP_H", 10))),
            thickness=int(getattr(params, "OVERLAY_BOX_THICKNESS", 1)),
            verbose=True,
        )

    if getattr(params, "RUN_STAGE9_DETECTION_SUMMARY", False):
        summary_dir = params.STAGE9_DIR / video_path.stem
        stage9_generate_detection_summary(
            stage5_video_dir=stage5_video_dir,
            output_dir=summary_dir,
            stage7_video_dir=params.STAGE7_DIR / video_path.stem,
            stage8_video_dir=params.STAGE8_DIR / video_path.stem,
            include_nearest_tp=True,
            verbose=True,
        )

    if getattr(params, "RUN_STAGE6_OVERLAY", False):
        legend = "LEGEND_GT=GREEN_MODEL=RED_OVERLAP=YELLOW"
        out_overlay = params.STAGE6_DIR / f"{video_path.stem}_gt_vs_model__{legend}.mp4"
        overlay_gt_vs_model(
            orig_video_path=video_path,
            pred_csv_path=pred_csv,
            out_video_path=out_overlay,
            gt_norm_csv_path=None,  # Stage 5 wrote *_norm_offset*.csv into stage5_video_dir
            thickness=int(getattr(params, "OVERLAY_BOX_THICKNESS", 1)),
            gt_box_w=int(getattr(params, "STAGE6_GT_BOX_W", getattr(params, "STAGE5_CROP_W", 10))),
            gt_box_h=int(getattr(params, "STAGE6_GT_BOX_H", getattr(params, "STAGE5_CROP_H", 10))),
            only_firefly_rows=bool(getattr(params, "STAGE5_ONLY_FIREFLY_ROWS", True)),
            max_frames=getattr(params, "MAX_FRAMES", None),
            stage5_dir_hint=params.STAGE5_DIR,
            render_threshold_overlays=True,
            thr_box_w=int(getattr(params, "STAGE6_GT_BOX_W", getattr(params, "STAGE5_CROP_W", 10))),
            thr_box_h=int(getattr(params, "STAGE6_GT_BOX_H", getattr(params, "STAGE5_CROP_H", 10))),
        )


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    _ = argv  # unused for now

    params.ROOT.mkdir(parents=True, exist_ok=True)
    # Test suite outputs
    params.STAGE5_DIR.mkdir(parents=True, exist_ok=True)
    params.STAGE6_DIR.mkdir(parents=True, exist_ok=True)
    params.STAGE7_DIR.mkdir(parents=True, exist_ok=True)
    params.STAGE8_DIR.mkdir(parents=True, exist_ok=True)
    params.STAGE9_DIR.mkdir(parents=True, exist_ok=True)

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
        s3_stage32_csv: Path | None = None

        t0 = time.perf_counter()
        out_images = stage1_run(vid)
        stage_times["stage1"] = time.perf_counter() - t0
        print(f"Stage1  Time: {stage_times['stage1']:.2f}s (images: {len(out_images)})")

        t0 = time.perf_counter()
        s2_csv = stage2_run(vid)
        stage_times["stage2"] = time.perf_counter() - t0
        print(f"Stage2  Time: {stage_times['stage2']:.2f}s (csv: {s2_csv.name})")

        t0 = time.perf_counter()
        s3_csv = stage3_run(vid)
        stage_times["stage3"] = time.perf_counter() - t0
        print(f"Stage3  Time: {stage_times['stage3']:.2f}s (csv: {s3_csv.name})")

        if getattr(params, "RUN_STAGE3_1_TRAJECTORY_INTENSITY_SELECTOR", True):
            t0 = time.perf_counter()
            s3_stage31_csv = stage3_1_run(vid)
            stage_times["stage3_1"] = time.perf_counter() - t0
            print(
                f"Stage3.1 Time: {stage_times['stage3_1']:.2f}s (csv: {s3_stage31_csv.name})"
            )

        if getattr(params, "RUN_STAGE3_2", True):
            t0 = time.perf_counter()
            s3_stage32_csv = stage3_2_run(vid)
            stage_times["stage3_2"] = time.perf_counter() - t0
            print(
                f"Stage3.2 Time: {stage_times['stage3_2']:.2f}s (csv: {s3_stage32_csv.name})"
            )

        t0 = time.perf_counter()
        out_vid = stage4_run(vid)
        stage_times["stage4"] = time.perf_counter() - t0
        print(f"Stage4  Time: {stage_times['stage4']:.2f}s (video: {out_vid.name})")

        _print_stage_times(stage_times)

        # --- Post-pipeline test suite (Stage 5–9) ---
        try:
            _maybe_run_test_suite(
                video_path=vid,
                stage3_csv=s3_csv,
                stage3_2_csv=s3_stage32_csv,
            )
        except Exception as exc:
            print(f"[tests] Warning: test suite for {vid.name} encountered an issue: {exc}")

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
