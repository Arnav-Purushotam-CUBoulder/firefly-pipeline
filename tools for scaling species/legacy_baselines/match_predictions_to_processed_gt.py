#!/usr/bin/env python3
"""
Match baseline predictions against a normalized GT CSV.

This matcher is intentionally route-agnostic. It expects:
  - a predictions CSV with x,y,t (or x,y,frame/frame_idx), and
  - a normalized GT CSV with x,y,t

It writes only the thresholded TP/FP/FN CSVs needed by the temporary runner's
metric parser. No day/night pipeline validator import is required.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _pairwise_dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return float(dx * dx + dy * dy)


def _greedy_match_full(
    frame_gts: List[Tuple[int, int]],
    frame_preds_xy: List[Tuple[float, float]],
    max_dist_px: float,
):
    n_g = len(frame_gts)
    n_p = len(frame_preds_xy)
    if n_g == 0 and n_p == 0:
        return [], [], []

    max_d2 = float(max_dist_px) * float(max_dist_px)
    used_g = [False] * n_g
    used_p = [False] * n_p
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(frame_gts):
        for pi, p in enumerate(frame_preds_xy):
            d2 = _pairwise_dist2((float(g[0]), float(g[1])), p)
            if d2 <= max_d2:
                pairs.append((d2, gi, pi))
    pairs.sort(key=lambda x: x[0])

    matches: List[Tuple[int, int, float]] = []
    for d2, gi, pi in pairs:
        if not used_g[gi] and not used_p[pi]:
            used_g[gi] = True
            used_p[pi] = True
            matches.append((gi, pi, math.sqrt(d2)))

    unmatched_pred = [i for i, used in enumerate(used_p) if not used]
    unmatched_gt = [i for i, used in enumerate(used_g) if not used]
    return matches, unmatched_pred, unmatched_gt


def _read_gt(csv_path: Path, max_frames: int | None) -> Dict[int, List[Tuple[int, int]]]:
    gt_by_t: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"x", "y", "t"}
        cols = set(reader.fieldnames or [])
        if not required.issubset(cols):
            raise ValueError(f"GT CSV must have columns x,y,t; found: {reader.fieldnames}")
        for row in reader:
            try:
                x = int(round(float(row["x"])))
                y = int(round(float(row["y"])))
                t = int(round(float(row["t"])))
            except Exception:
                continue
            if t < 0:
                continue
            if max_frames is not None and t >= int(max_frames):
                continue
            gt_by_t[t].append((x, y))
    return gt_by_t


def _read_predictions(pred_csv: Path, max_frames: int | None) -> Dict[int, List[Dict[str, float]]]:
    preds_by_t: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    with pred_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        cols = {str(c).strip().lower(): c for c in fieldnames}
        x_col = cols.get("x")
        y_col = cols.get("y")
        t_col = cols.get("t") or cols.get("frame") or cols.get("frame_idx")
        class_col = cols.get("class")
        conf_col = cols.get("firefly_confidence")
        if not (x_col and y_col and t_col):
            raise ValueError(
                f"Predictions CSV must have x,y,t (or frame/frame_idx); found: {reader.fieldnames}"
            )

        for row in reader:
            if class_col:
                cls = str(row.get(class_col) or "").strip().lower()
                if cls and cls != "firefly":
                    continue
            try:
                x = float(row.get(x_col) or 0.0)
                y = float(row.get(y_col) or 0.0)
                t = int(round(float(row.get(t_col) or 0.0)))
            except Exception:
                continue
            if t < 0:
                continue
            if max_frames is not None and t >= int(max_frames):
                continue
            conf = None
            if conf_col:
                try:
                    conf = float(row.get(conf_col) or 0.0)
                except Exception:
                    conf = None
            preds_by_t[t].append({"x": x, "y": y, "conf": conf})
    return preds_by_t


def _write_rows_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _evaluate_frames(
    gt_by_t: Dict[int, List[Tuple[int, int]]],
    preds_by_t: Dict[int, List[Dict[str, float]]],
    max_dist_px: float,
):
    fps_rows: List[Dict[str, object]] = []
    tps_rows: List[Dict[str, object]] = []
    fns_rows: List[Dict[str, object]] = []
    err_sum = 0.0
    err_n = 0

    all_t = sorted(set(gt_by_t.keys()) | set(preds_by_t.keys()))
    for t in all_t:
        frame_gts = list(gt_by_t.get(t) or [])
        frame_preds = list(preds_by_t.get(t) or [])
        frame_preds_xy = [(float(p["x"]), float(p["y"])) for p in frame_preds]
        matches, unmatched_pred, unmatched_gt = _greedy_match_full(frame_gts, frame_preds_xy, max_dist_px)

        for gi, pi, dist_px in matches:
            gt_x, gt_y = frame_gts[gi]
            pred = frame_preds[pi]
            tps_rows.append(
                {
                    "t": int(t),
                    "gt_x": int(gt_x),
                    "gt_y": int(gt_y),
                    "pred_x": float(pred["x"]),
                    "pred_y": float(pred["y"]),
                    "dist_px": float(dist_px),
                    "firefly_confidence": pred.get("conf", ""),
                }
            )
            err_sum += float(dist_px)
            err_n += 1

        for pi in unmatched_pred:
            pred = frame_preds[pi]
            fps_rows.append(
                {
                    "t": int(t),
                    "x": float(pred["x"]),
                    "y": float(pred["y"]),
                    "firefly_confidence": pred.get("conf", ""),
                }
            )

        for gi in unmatched_gt:
            gt_x, gt_y = frame_gts[gi]
            fns_rows.append({"t": int(t), "x": int(gt_x), "y": int(gt_y)})

    tp = int(len(tps_rows))
    fp = int(len(fps_rows))
    fn = int(len(fns_rows))
    mean_err = (float(err_sum) / float(err_n)) if err_n > 0 else float("nan")
    return fps_rows, tps_rows, fns_rows, (tp, fp, fn, mean_err)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match baseline predictions against a normalized GT CSV.")
    p.add_argument(
        "--route",
        type=str,
        default="",
        help="Ignored; retained only for backward compatibility with older calls.",
    )
    p.add_argument("--video", type=str, required=True, help="Original source video.")
    p.add_argument("--pred-csv", type=str, required=True, help="Predictions CSV to evaluate.")
    p.add_argument("--gt-csv", type=str, default="", help="Normalized GT CSV with columns x,y,t.")
    p.add_argument(
        "--processed-gt-csv",
        type=str,
        default="",
        help="Deprecated alias for --gt-csv.",
    )
    p.add_argument("--out-dir", type=str, required=True, help="Output dir where thr_*/fps.csv,tps.csv,fns.csv are written.")
    p.add_argument("--dist-thresholds", type=float, nargs="+", required=True, help="Distance thresholds in pixels.")
    p.add_argument("--crop-w", type=int, default=10, help="Ignored; retained for backward compatibility.")
    p.add_argument("--crop-h", type=int, default=10, help="Ignored; retained for backward compatibility.")
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames evaluated.")
    p.add_argument(
        "--naming-bin-threshold",
        type=int,
        default=None,
        help="Ignored; retained for backward compatibility.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    video_path = Path(args.video).expanduser().resolve()
    pred_csv_path = Path(args.pred_csv).expanduser().resolve()
    gt_csv_arg = str(args.gt_csv or "").strip() or str(args.processed_gt_csv or "").strip()
    gt_csv_path = Path(gt_csv_arg).expanduser().resolve() if gt_csv_arg else None
    out_dir = Path(args.out_dir).expanduser().resolve()
    max_frames = int(args.max_frames) if args.max_frames is not None else None

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not pred_csv_path.exists():
        raise SystemExit(f"Predictions CSV not found: {pred_csv_path}")
    if gt_csv_path is None or not gt_csv_path.exists():
        raise SystemExit(f"GT CSV not found: {gt_csv_path}")

    gt_by_t = _read_gt(gt_csv_path, max_frames=max_frames)
    preds_by_t = _read_predictions(pred_csv_path, max_frames=max_frames)

    n_gt = sum(len(v) for v in gt_by_t.values())
    n_preds = sum(len(v) for v in preds_by_t.values())
    print(f"[baseline-gt-match] video={video_path}")
    print(f"[baseline-gt-match] pred_csv={pred_csv_path}")
    print(f"[baseline-gt-match] gt_csv={gt_csv_path}")
    print(f"[baseline-gt-match] gt_points={n_gt} pred_points={n_preds}")
    if max_frames is not None:
        print(f"[baseline-gt-match] max_frames={max_frames}")

    for thr in [float(t) for t in args.dist_thresholds]:
        fps_rows, tps_rows, fns_rows, (tp, fp, fn, mean_err) = _evaluate_frames(gt_by_t, preds_by_t, thr)
        thr_dir = out_dir / f"thr_{thr:.1f}px"
        _write_rows_csv(thr_dir / "fps.csv", ["t", "x", "y", "firefly_confidence"], fps_rows)
        _write_rows_csv(
            thr_dir / "tps.csv",
            ["t", "gt_x", "gt_y", "pred_x", "pred_y", "dist_px", "firefly_confidence"],
            tps_rows,
        )
        _write_rows_csv(thr_dir / "fns.csv", ["t", "x", "y"], fns_rows)

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
