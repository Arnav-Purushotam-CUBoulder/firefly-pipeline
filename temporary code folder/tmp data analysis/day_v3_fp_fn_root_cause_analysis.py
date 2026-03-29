#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent


FP_RE = re.compile(
    r"^(?P<label>TP|FP|FN)_t(?P<t>\d+)_x(?P<x>-?\d+)_y(?P<y>-?\d+)_conf(?P<conf>\d+(?:\.\d+)?)_max(?P<max>\d+)_area(?P<area>\d+)\.png$",
    re.IGNORECASE,
)


@dataclass
class Stage2Box:
    x: float
    y: float
    w: float
    h: float
    start_t: int
    end_t: int

    @property
    def area(self) -> float:
        return float(self.w * self.h)

    def contains(self, x: float, y: float, t: int) -> bool:
        return (
            self.start_t <= int(t) <= self.end_t
            and self.x <= x <= (self.x + self.w)
            and self.y <= y <= (self.y + self.h)
        )

    def center_dist(self, x: float, y: float) -> float:
        cx = self.x + self.w / 2.0
        cy = self.y + self.h / 2.0
        return math.hypot(float(x) - cx, float(y) - cy)


@dataclass
class Stage3Det:
    t: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    det_id: int
    traj_id: Optional[int] = None
    traj_size: Optional[int] = None
    traj_motion_xy: Optional[float] = None
    traj_intensity_range: Optional[int] = None
    traj_is_selected: Optional[int] = None

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class Stage32Det:
    t: int
    x: float
    y: float
    firefly_logit: float
    background_logit: float


@dataclass
class FinalDet:
    species_name: str
    video_stem: str
    label: str
    x: float
    y: float
    t: int
    confidence: float
    filepath: str
    crop_max: Optional[int]
    crop_area: Optional[int]


def _safe_float(v: object) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: object) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except Exception:
        return None


def _summary(vals: Iterable[float]) -> dict[str, Optional[float]]:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": float(mean(xs)),
        "median": float(median(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def _infer_stage31_rejection_reason(
    *,
    traj_size: Optional[int],
    traj_intensity_range: Optional[int],
    traj_is_selected: Optional[int],
    min_track_points: int,
    min_intensity_range: int,
) -> Optional[str]:
    if traj_size is None or traj_intensity_range is None or traj_is_selected is None:
        return None
    if int(traj_size) < int(min_track_points):
        return "traj_too_short"
    if int(traj_intensity_range) < int(min_intensity_range):
        return "intensity_range_too_low"
    if int(traj_is_selected) == 0:
        return "hill_shape_failure_inferred"
    return "selected"


def _find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {path}")
    if len(matches) > 1:
        # Stable enough to take the first sorted match.
        return matches[0]
    return matches[0]


def _parse_crop_stats(filepath: str) -> tuple[Optional[int], Optional[int]]:
    m = FP_RE.match(Path(filepath).name)
    if not m:
        return None, None
    return int(m.group("max")), int(m.group("area"))


def _load_stage2_boxes(path: Path) -> list[Stage2Box]:
    out: list[Stage2Box] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                x = float(row["x"])
                y = float(row["y"])
                w = float(row["w"])
                h = float(row["h"])
                start_s, end_s = str(row["frame_range"]).split("-", 1)
                start_t = int(start_s)
                end_t = int(end_s)
            except Exception:
                continue
            out.append(Stage2Box(x=x, y=y, w=w, h=h, start_t=start_t, end_t=end_t))
    return out


def _load_stage3_motion_all(path: Path) -> list[Stage3Det]:
    out: list[Stage3Det] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append(
                    Stage3Det(
                        t=int(row["frame_idx"]),
                        x=float(row["x"]),
                        y=float(row["y"]),
                        w=float(row["w"]),
                        h=float(row["h"]),
                        conf=float(row["conf"]),
                        det_id=int(float(row["det_id"])),
                        traj_id=_safe_int(row.get("traj_id")),
                        traj_size=_safe_int(row.get("traj_size")),
                        traj_motion_xy=_safe_float(row.get("traj_motion_xy")),
                        traj_intensity_range=_safe_int(row.get("traj_intensity_range")),
                        traj_is_selected=_safe_int(row.get("traj_is_selected")),
                    )
                )
            except Exception:
                continue
    return out


def _load_stage32(path: Path) -> list[Stage32Det]:
    out: list[Stage32Det] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append(
                    Stage32Det(
                        t=int(float(row["t"])),
                        x=float(row["x"]),
                        y=float(row["y"]),
                        firefly_logit=float(row["firefly_logit"]),
                        background_logit=float(row["background_logit"]),
                    )
                )
            except Exception:
                continue
    return out


def _load_final_csv(path: Path, species_name: str, video_stem: str, label: str) -> list[FinalDet]:
    out: list[FinalDet] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            crop_max, crop_area = _parse_crop_stats(row.get("filepath", ""))
            try:
                out.append(
                    FinalDet(
                        species_name=species_name,
                        video_stem=video_stem,
                        label=label,
                        x=float(row["x"]),
                        y=float(row["y"]),
                        t=int(float(row["t"])),
                        confidence=float(row.get("confidence") or 0.0),
                        filepath=row.get("filepath", ""),
                        crop_max=crop_max,
                        crop_area=crop_area,
                    )
                )
            except Exception:
                continue
    return out


def _nearest_stage3(dets: list[Stage3Det], x: float, y: float, t: int, max_dist: float) -> tuple[Optional[Stage3Det], Optional[float]]:
    best: tuple[Optional[Stage3Det], Optional[float]] = (None, None)
    thr2 = float(max_dist) * float(max_dist)
    for d in dets:
        if d.t != int(t):
            continue
        dx = float(x) - d.cx
        dy = float(y) - d.cy
        dsq = dx * dx + dy * dy
        if dsq > thr2:
            continue
        dist = math.sqrt(dsq)
        if best[0] is None or dist < float(best[1]):
            best = (d, dist)
    return best


def _nearest_stage32(dets: list[Stage32Det], x: float, y: float, t: int, max_dist: float) -> tuple[Optional[Stage32Det], Optional[float]]:
    best: tuple[Optional[Stage32Det], Optional[float]] = (None, None)
    thr2 = float(max_dist) * float(max_dist)
    for d in dets:
        if d.t != int(t):
            continue
        dx = float(x) - d.x
        dy = float(y) - d.y
        dsq = dx * dx + dy * dy
        if dsq > thr2:
            continue
        dist = math.sqrt(dsq)
        if best[0] is None or dist < float(best[1]):
            best = (d, dist)
    return best


def _containing_stage2(boxes: list[Stage2Box], x: float, y: float, t: int) -> list[Stage2Box]:
    return [b for b in boxes if b.contains(x, y, t)]


def _stage2_best_box(boxes: list[Stage2Box], x: float, y: float, t: int) -> Optional[Stage2Box]:
    contain = _containing_stage2(boxes, x, y, t)
    if not contain:
        return None
    return min(contain, key=lambda b: (b.area, b.center_dist(x, y)))


def _video_case_dirs(run_root: Path) -> list[Path]:
    root = run_root / "inference_outputs" / "pipelines" / "day_videos" / "global_all_species"
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _species_map(final_results_csv: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    with final_results_csv.open("r", newline="") as f:
        for row in csv.DictReader(f):
            out[row["video_name"]] = (row["species_name"], row["video_name"])
    return out


def analyze_run(run_root: Path, output_dir: Path, dist_px: float = 10.0, selected_near_px: float = 20.0) -> dict:
    final_results_csv = run_root / "final_results.csv"
    video_meta = _species_map(final_results_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    fn_detail_csv = output_dir / "fn_root_cause_details.csv"
    det_detail_csv = output_dir / "tp_fp_comparison_details.csv"

    per_species = defaultdict(lambda: {
        "tp": [],
        "fp": [],
        "fn": [],
        "fn_cause_counts": Counter(),
        "fn_stage31_rejection_subcauses": Counter(),
        "fn_confident_stage3_positive": 0,
        "fn_stage2_hit": 0,
        "fn_stage3_hit": 0,
        "fn_stage3_selected_near": 0,
        "n_videos": 0,
    })

    min_track_points = 3
    min_intensity_range = 3000

    fn_rows_for_csv: list[dict] = []
    det_rows_for_csv: list[dict] = []

    for case_dir in _video_case_dirs(run_root):
        # video stem from ground truth/gt.csv or directory naming
        gt_csv = case_dir / "day_pipeline_v3" / "ground truth" / "gt.csv"
        if not gt_csv.exists():
            continue
        video_stem = case_dir.name.split("__", 1)[1] if "__" in case_dir.name else None
        # actual stem is easier from stage5 directory
        stage5_root = _find_one(case_dir / "day_pipeline_v3" / "stage5 validation", "*")
        video_stem = stage5_root.name
        species_name = None
        with final_results_csv.open("r", newline="") as f:
            for row in csv.DictReader(f):
                if row["video_name"] == video_stem:
                    species_name = row["species_name"]
                    break
        if species_name is None:
            continue
        per_species[species_name]["n_videos"] += 1

        stage2_csv = _find_one(case_dir / "day_pipeline_v3" / "stage2_yolo_detections" / video_stem, "*.csv")
        stage3_dir = case_dir / "day_pipeline_v3" / "stage3_patch_classifier" / video_stem
        stage3_motion_all_csv = stage3_dir / f"{video_stem}_patches_motion_all.csv"
        stage32_csv = _find_one(stage3_dir / "stage3_2", "*_stage3_2_firefly_background_logits.csv")
        thr_dir = case_dir / "day_pipeline_v3" / "stage5 validation" / video_stem / f"thr_{dist_px:.1f}px"

        stage2_boxes = _load_stage2_boxes(stage2_csv)
        stage3_all = _load_stage3_motion_all(stage3_motion_all_csv)
        stage32 = _load_stage32(stage32_csv)

        tps = _load_final_csv(thr_dir / "tps.csv", species_name, video_stem, "TP")
        fps = _load_final_csv(thr_dir / "fps.csv", species_name, video_stem, "FP")
        fns = _load_final_csv(thr_dir / "fns.csv", species_name, video_stem, "FN")

        for det in tps + fps:
            stage3_match, stage3_dist = _nearest_stage3(stage3_all, det.x, det.y, det.t, max_dist=dist_px)
            det_rows_for_csv.append(
                {
                    "species_name": species_name,
                    "video_stem": video_stem,
                    "label": det.label,
                    "x": det.x,
                    "y": det.y,
                    "t": det.t,
                    "confidence": det.confidence,
                    "crop_max": det.crop_max,
                    "crop_area": det.crop_area,
                    "stage3_dist_px": stage3_dist,
                    "traj_id": None if stage3_match is None else stage3_match.traj_id,
                    "traj_size": None if stage3_match is None else stage3_match.traj_size,
                    "traj_motion_xy": None if stage3_match is None else stage3_match.traj_motion_xy,
                    "traj_intensity_range": None if stage3_match is None else stage3_match.traj_intensity_range,
                    "traj_is_selected": None if stage3_match is None else stage3_match.traj_is_selected,
                }
            )
            per_species[species_name][det.label.lower()].append(
                {
                    "confidence": det.confidence,
                    "crop_max": det.crop_max,
                    "crop_area": det.crop_area,
                    "traj_size": None if stage3_match is None else stage3_match.traj_size,
                    "traj_motion_xy": None if stage3_match is None else stage3_match.traj_motion_xy,
                    "traj_intensity_range": None if stage3_match is None else stage3_match.traj_intensity_range,
                }
            )

        for det in fns:
            stage2_box = _stage2_best_box(stage2_boxes, det.x, det.y, det.t)
            stage3_10, stage3_10_dist = _nearest_stage3(stage3_all, det.x, det.y, det.t, max_dist=dist_px)
            stage3_20, stage3_20_dist = _nearest_stage3(stage3_all, det.x, det.y, det.t, max_dist=selected_near_px)
            stage32_10, stage32_10_dist = _nearest_stage32(stage32, det.x, det.y, det.t, max_dist=dist_px)
            stage32_20, stage32_20_dist = _nearest_stage32(stage32, det.x, det.y, det.t, max_dist=selected_near_px)

            if stage2_box is not None:
                per_species[species_name]["fn_stage2_hit"] += 1
            if stage3_10 is not None:
                per_species[species_name]["fn_stage3_hit"] += 1
            if stage32_20 is not None:
                per_species[species_name]["fn_stage3_selected_near"] += 1
            if stage3_10 is not None and det.confidence >= 0.95:
                per_species[species_name]["fn_confident_stage3_positive"] += 1

            if stage2_box is None:
                cause = "stage2_yolo_long_exposure_miss"
            elif stage3_10 is None:
                if stage3_20 is not None:
                    cause = "stage3_patch_recentering_offset"
                else:
                    cause = "stage3_patch_classifier_or_recentering_miss"
            elif stage32_10 is None:
                if stage3_10.traj_is_selected == 0:
                    cause = "stage3_1_trajectory_rejection"
                elif stage32_20 is not None:
                    cause = "stage3_2_centroid_or_validation_threshold_miss"
                else:
                    cause = "selected_prediction_missing_after_stage3_1"
            else:
                cause = "unexpected_matched_prediction"

            per_species[species_name]["fn_cause_counts"][cause] += 1
            stage31_subcause = None
            if cause == "stage3_1_trajectory_rejection":
                stage31_subcause = _infer_stage31_rejection_reason(
                    traj_size=None if stage3_10 is None else stage3_10.traj_size,
                    traj_intensity_range=None if stage3_10 is None else stage3_10.traj_intensity_range,
                    traj_is_selected=None if stage3_10 is None else stage3_10.traj_is_selected,
                    min_track_points=min_track_points,
                    min_intensity_range=min_intensity_range,
                )
                if stage31_subcause:
                    per_species[species_name]["fn_stage31_rejection_subcauses"][stage31_subcause] += 1
            per_species[species_name]["fn"].append(
                {
                    "confidence": det.confidence,
                    "crop_max": det.crop_max,
                    "crop_area": det.crop_area,
                    "traj_size": None if stage3_10 is None else stage3_10.traj_size,
                    "traj_motion_xy": None if stage3_10 is None else stage3_10.traj_motion_xy,
                    "traj_intensity_range": None if stage3_10 is None else stage3_10.traj_intensity_range,
                }
            )
            fn_rows_for_csv.append(
                {
                    "species_name": species_name,
                    "video_stem": video_stem,
                    "x": det.x,
                    "y": det.y,
                    "t": det.t,
                    "fn_confidence": det.confidence,
                    "crop_max": det.crop_max,
                    "crop_area": det.crop_area,
                    "cause": cause,
                    "stage2_hit": int(stage2_box is not None),
                    "stage2_box_area": None if stage2_box is None else stage2_box.area,
                    "stage2_box_w": None if stage2_box is None else stage2_box.w,
                    "stage2_box_h": None if stage2_box is None else stage2_box.h,
                    "nearest_stage3_dist_px": stage3_10_dist,
                    "nearest_stage3_conf": None if stage3_10 is None else stage3_10.conf,
                    "nearest_stage3_traj_size": None if stage3_10 is None else stage3_10.traj_size,
                    "nearest_stage3_traj_motion_xy": None if stage3_10 is None else stage3_10.traj_motion_xy,
                    "nearest_stage3_traj_intensity_range": None if stage3_10 is None else stage3_10.traj_intensity_range,
                    "nearest_stage3_traj_is_selected": None if stage3_10 is None else stage3_10.traj_is_selected,
                    "nearest_stage3_selected_dist_px": stage32_20_dist,
                    "stage3_1_rejection_subcause": stage31_subcause,
                }
            )

    with fn_detail_csv.open("w", newline="") as f:
        fields = list(fn_rows_for_csv[0].keys()) if fn_rows_for_csv else ["species_name"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in fn_rows_for_csv:
            w.writerow(row)

    with det_detail_csv.open("w", newline="") as f:
        fields = list(det_rows_for_csv[0].keys()) if det_rows_for_csv else ["species_name"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in det_rows_for_csv:
            w.writerow(row)

    summary: dict[str, object] = {
        "run_root": str(run_root),
        "threshold_px": dist_px,
        "selected_near_px": selected_near_px,
        "species": {},
        "detail_csvs": {
            "fn_root_cause_details": str(fn_detail_csv),
            "tp_fp_comparison_details": str(det_detail_csv),
        },
    }

    for species_name, data in sorted(per_species.items()):
        tp = data["tp"]
        fp = data["fp"]
        fn = data["fn"]
        summary["species"][species_name] = {
            "n_videos": data["n_videos"],
            "counts": {
                "tp": len(tp),
                "fp": len(fp),
                "fn": len(fn),
            },
            "fp_vs_tp_stats": {
                "tp_confidence": _summary(x["confidence"] for x in tp),
                "fp_confidence": _summary(x["confidence"] for x in fp),
                "tp_crop_max": _summary(x["crop_max"] for x in tp),
                "fp_crop_max": _summary(x["crop_max"] for x in fp),
                "tp_crop_area": _summary(x["crop_area"] for x in tp),
                "fp_crop_area": _summary(x["crop_area"] for x in fp),
                "tp_traj_size": _summary(x["traj_size"] for x in tp),
                "fp_traj_size": _summary(x["traj_size"] for x in fp),
                "tp_traj_motion_xy": _summary(x["traj_motion_xy"] for x in tp),
                "fp_traj_motion_xy": _summary(x["traj_motion_xy"] for x in fp),
                "tp_traj_intensity_range": _summary(x["traj_intensity_range"] for x in tp),
                "fp_traj_intensity_range": _summary(x["traj_intensity_range"] for x in fp),
            },
            "fn_stage_breakdown": {
                "cause_counts": dict(data["fn_cause_counts"]),
                "stage3_1_rejection_subcauses": dict(data["fn_stage31_rejection_subcauses"]),
                "stage2_hit_frac": (data["fn_stage2_hit"] / len(fn)) if fn else None,
                "stage3_positive_hit_frac": (data["fn_stage3_hit"] / len(fn)) if fn else None,
                "stage3_selected_near_frac": (data["fn_stage3_selected_near"] / len(fn)) if fn else None,
                "high_confidence_fn_with_stage3_positive_frac": (data["fn_confident_stage3_positive"] / len(fn)) if fn else None,
                "fn_confidence": _summary(x["confidence"] for x in fn),
                "fn_crop_max": _summary(x["crop_max"] for x in fn),
                "fn_crop_area": _summary(x["crop_area"] for x in fn),
                "fn_traj_size": _summary(x["traj_size"] for x in fn),
                "fn_traj_motion_xy": _summary(x["traj_motion_xy"] for x in fn),
                "fn_traj_intensity_range": _summary(x["traj_intensity_range"] for x in fn),
            },
        }

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Root-cause analysis for day pipeline v3 TP/FP/FN artifacts.")
    ap.add_argument(
        "--run-root",
        type=Path,
        default=Path("/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/tmp_day_night_combo_train_and_infer__20260328__005102"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "reports" / "day_v3_root_cause_analysis__20260328",
    )
    ap.add_argument("--threshold-px", type=float, default=10.0)
    ap.add_argument("--selected-near-px", type=float, default=20.0)
    args = ap.parse_args()

    summary = analyze_run(
        run_root=args.run_root,
        output_dir=args.output_dir,
        dist_px=float(args.threshold_px),
        selected_near_px=float(args.selected_near_px),
    )
    out_json = args.output_dir / "summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nsummary_json={out_json}")


if __name__ == "__main__":
    main()
