#!/usr/bin/env python3
"""
Match baseline predictions against an already-processed pipeline GT CSV.

This script intentionally does not normalize/filter/dedupe raw GT. It expects a
final x,y,t GT CSV produced by the route-specific day/night pipeline validator,
then runs only the remaining prediction-matching and TP/FP/FN writing steps.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validator_path_for_route(route: str) -> Path:
    route_norm = str(route or "").strip().lower()
    if route_norm == "day":
        return _repo_root() / "Pipelines" / "day time pipeline v3 (yolo + patch classifier ensemble)" / "stage5_validate.py"
    if route_norm == "night":
        return _repo_root() / "Pipelines" / "night_time_pipeline" / "stage9_validate.py"
    raise ValueError(f"Unsupported route for processed-GT matcher: {route}")


def _load_validator_module(route: str):
    validator_path = _validator_path_for_route(route)
    if not validator_path.exists():
        raise FileNotFoundError(f"Validator not found for route={route}: {validator_path}")
    spec = importlib.util.spec_from_file_location(f"_processed_gt_matcher_{route}", validator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load validator module: {validator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_naming_bin_threshold(route: str) -> int:
    route_norm = str(route or "").strip().lower()
    if route_norm == "night":
        return 30
    return 50


def _read_processed_gt(csv_path: Path, max_frames: int | None) -> Dict[int, List[Tuple[int, int]]]:
    gt_by_t: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"x", "y", "t"}
        cols = set(reader.fieldnames or [])
        if not required.issubset(cols):
            raise ValueError(f"Processed GT CSV must have columns x,y,t; found: {reader.fieldnames}")
        for row in reader:
            try:
                x = int(round(float(row["x"])))
                y = int(round(float(row["y"])))
                t = int(round(float(row["t"])))
                if t < 0:
                    continue
                if max_frames is not None and t >= max_frames:
                    continue
            except Exception:
                continue
            gt_by_t[t].append((x, y))
    return gt_by_t


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match predictions to an already-processed pipeline GT CSV.")
    p.add_argument("--route", type=str, required=True, choices=["day", "night"], help="Pipeline route that produced the processed GT.")
    p.add_argument("--video", type=str, required=True, help="Original source video.")
    p.add_argument("--pred-csv", type=str, required=True, help="Predictions CSV to evaluate.")
    p.add_argument("--processed-gt-csv", type=str, required=True, help="Final processed GT CSV with columns x,y,t.")
    p.add_argument("--out-dir", type=str, required=True, help="Output dir where thr_*/fps.csv,tps.csv,fns.csv are written.")
    p.add_argument("--dist-thresholds", type=float, nargs="+", required=True, help="Distance thresholds in pixels.")
    p.add_argument("--crop-w", type=int, default=10, help="Crop width for TP/FP/FN crop writing.")
    p.add_argument("--crop-h", type=int, default=10, help="Crop height for TP/FP/FN crop writing.")
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames evaluated.")
    p.add_argument("--naming-bin-threshold", type=int, default=None, help="Brightness threshold used only for crop filename brightness/area stats.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    route = str(args.route).strip().lower()
    video_path = Path(args.video).expanduser().resolve()
    pred_csv_path = Path(args.pred_csv).expanduser().resolve()
    processed_gt_csv = Path(args.processed_gt_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    max_frames = int(args.max_frames) if args.max_frames is not None else None

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not pred_csv_path.exists():
        raise SystemExit(f"Predictions CSV not found: {pred_csv_path}")
    if not processed_gt_csv.exists():
        raise SystemExit(f"Processed GT CSV not found: {processed_gt_csv}")

    module = _load_validator_module(route)
    module._ensure_dir(out_dir)
    module._NAMING_BIN_THR = int(
        args.naming_bin_threshold
        if args.naming_bin_threshold is not None
        else _default_naming_bin_threshold(route)
    )

    gt_by_t = _read_processed_gt(processed_gt_csv, max_frames=max_frames)
    preds_by_t = module._read_predictions(pred_csv_path, only_firefly_rows=True, max_frames=max_frames)

    n_gt = sum(len(v) for v in gt_by_t.values())
    n_preds = sum(len(v) for v in preds_by_t.values())
    print(f"[processed-gt-match] route={route}")
    print(f"[processed-gt-match] video={video_path}")
    print(f"[processed-gt-match] pred_csv={pred_csv_path}")
    print(f"[processed-gt-match] processed_gt_csv={processed_gt_csv}")
    print(f"[processed-gt-match] gt_points={n_gt} pred_points={n_preds}")
    if max_frames is not None:
        print(f"[processed-gt-match] max_frames={max_frames}")

    for thr in [float(t) for t in args.dist_thresholds]:
        fps_by_t, tps_by_t, fns_by_t, (tp, fp, fn, mean_err) = module._evaluate_frames(gt_by_t, preds_by_t, thr)
        module._write_crops_and_csvs_for_threshold(
            video_path,
            out_dir,
            float(thr),
            fps_by_t,
            tps_by_t,
            fns_by_t,
            int(args.crop_w),
            int(args.crop_h),
            max_frames,
            fn_conf_getter=None,
        )
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        print(f"Threshold: {thr:.1f}px")
        print(f"  TP: {tp}   FP: {fp}   FN: {fn}")
        print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}")
        print(f"  Mean error (px): {mean_err:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
