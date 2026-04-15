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
RUN_ROOT_DEFAULT = Path(
    "/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/"
    "tmp_day_night_combo_train_and_infer__20260328__005102"
)
OUT_DIR_DEFAULT = SCRIPT_DIR / "reports" / "day_v3_hill_shape_deep_dive__20260328"

TRAJ_RE = re.compile(
    r"^(?P<traj_id>\d+)_(?P<status>ACCEPT|REJECT)_n(?P<n>\d+)_t(?P<t0>\d+)-(?P<t1>\d+)_m(?P<motion>[\d.]+)_range(?P<intensity>\d+)$"
)


@dataclass
class Trajectory:
    species_name: str
    video_stem: str
    traj_id: int
    status: str
    traj_size: int
    motion_xy: float
    intensity_range: int
    frame_start: int
    frame_end: int
    crop_sums: list[int]


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


def _moving_average(vals: list[float], win: int) -> list[float]:
    if win <= 1 or len(vals) <= 2:
        return [float(v) for v in vals]
    half = win // 2
    out: list[float] = []
    for i in range(len(vals)):
        a = max(0, i - half)
        b = min(len(vals), i + half + 1)
        seg = vals[a:b]
        out.append(float(sum(seg)) / float(len(seg)))
    return out


def _hill_metrics(
    ys: list[int],
    *,
    min_range: int = 3000,
    smooth_window: int = 1,
    min_up_steps: int = 2,
    min_down_steps: int = 2,
    min_monotonic_frac: float = 0.60,
    peak_pos_min_frac: float = 0.0,
    peak_pos_max_frac: float = 1.0,
) -> dict[str, object]:
    out: dict[str, object] = {
        "n_points": len(ys),
        "intensity_range": int(max(ys) - min(ys)) if ys else 0,
        "peak_idx": None,
        "peak_pos_frac": None,
        "up_steps": None,
        "down_steps": None,
        "up_frac": None,
        "down_frac": None,
        "hill_ok": False,
        "fail_reason": None,
    }
    if len(ys) < 3:
        out["fail_reason"] = "traj_too_short"
        return out
    if int(out["intensity_range"]) < int(min_range):
        out["fail_reason"] = "intensity_range_too_low"
        return out

    y = _moving_average(ys, int(smooth_window))
    peak_i = int(max(range(len(y)), key=lambda i: y[i]))
    n = len(y)
    peak_frac = peak_i / float(max(1, n - 1))
    out["peak_idx"] = peak_i
    out["peak_pos_frac"] = float(peak_frac)

    if peak_i <= int(math.floor(float(peak_pos_min_frac) * (n - 1))) or peak_i >= int(math.ceil(float(peak_pos_max_frac) * (n - 1))):
        out["fail_reason"] = "peak_edge_fail"
        return out

    diffs = [y[i + 1] - y[i] for i in range(n - 1)]
    signs = []
    for d in diffs:
        if abs(d) <= 1e-6:
            signs.append(0)
        elif d > 0:
            signs.append(1)
        else:
            signs.append(-1)

    before = [s for s in signs[:peak_i] if s != 0]
    after = [s for s in signs[peak_i:] if s != 0]
    if not before or not after:
        out["fail_reason"] = "no_before_after_slope"
        return out

    up_steps = sum(1 for s in before if s > 0)
    down_steps = sum(1 for s in after if s < 0)
    up_frac = up_steps / float(max(1, len(before)))
    down_frac = down_steps / float(max(1, len(after)))
    out["up_steps"] = int(up_steps)
    out["down_steps"] = int(down_steps)
    out["up_frac"] = float(up_frac)
    out["down_frac"] = float(down_frac)

    if up_steps < int(min_up_steps):
        out["fail_reason"] = "up_steps_too_low"
        return out
    if down_steps < int(min_down_steps):
        out["fail_reason"] = "down_steps_too_low"
        return out
    if up_frac < float(min_monotonic_frac):
        out["fail_reason"] = "up_monotonic_frac_too_low"
        return out
    if down_frac < float(min_monotonic_frac):
        out["fail_reason"] = "down_monotonic_frac_too_low"
        return out

    out["hill_ok"] = True
    out["fail_reason"] = "selected"
    return out


def _load_final_results(run_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with (run_root / "final_results.csv").open("r", newline="") as f:
        for row in csv.DictReader(f):
            out[row["video_name"]] = row["species_name"]
    return out


def _case_dirs(run_root: Path) -> list[Path]:
    root = run_root / "inference_outputs" / "pipelines" / "day_videos" / "global_all_species"
    return sorted(p for p in root.iterdir() if p.is_dir())


def _find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {path}")
    return matches[0]


def _load_trajectories(run_root: Path) -> dict[tuple[str, str, int], Trajectory]:
    species_by_video = _load_final_results(run_root)
    out: dict[tuple[str, str, int], Trajectory] = {}
    for case_dir in _case_dirs(run_root):
        stage5_root = _find_one(case_dir / "day_pipeline_v3" / "stage5 validation", "*")
        video_stem = stage5_root.name
        species_name = species_by_video[video_stem]
        traj_root = case_dir / "day_pipeline_v3" / "stage3_patch_classifier" / video_stem / "stage3_1_trajectory_crops"
        for traj_dir in sorted(p for p in traj_root.iterdir() if p.is_dir()):
            m = TRAJ_RE.match(traj_dir.name)
            if not m:
                continue
            traj_csv = traj_dir / "trajectory.csv"
            crop_sums: list[int] = []
            with traj_csv.open("r", newline="") as f:
                rows = sorted(csv.DictReader(f), key=lambda r: int(r["frame_idx"]))
            for row in rows:
                try:
                    crop_sum = int(float(row["crop_sum"]))
                except Exception:
                    continue
                crop_sums.append(crop_sum)
            traj = Trajectory(
                species_name=species_name,
                video_stem=video_stem,
                traj_id=int(m.group("traj_id")),
                status=m.group("status"),
                traj_size=int(m.group("n")),
                motion_xy=float(m.group("motion")),
                intensity_range=int(m.group("intensity")),
                frame_start=int(m.group("t0")),
                frame_end=int(m.group("t1")),
                crop_sums=crop_sums,
            )
            out[(species_name, video_stem, traj.traj_id)] = traj
    return out


def _load_group_members(report_dir: Path) -> tuple[dict[str, Counter], dict[str, Counter], dict[str, Counter]]:
    tp_fp = report_dir / "tp_fp_comparison_details.csv"
    fn_csv = report_dir / "fn_root_cause_details.csv"
    tp_members: dict[str, Counter] = defaultdict(Counter)
    fp_members: dict[str, Counter] = defaultdict(Counter)
    fn_members: dict[str, Counter] = defaultdict(Counter)

    with tp_fp.open("r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = f"{row['species_name']}|{row['video_stem']}|{int(float(row['traj_id']))}"
            except Exception:
                continue
            if row["label"] == "TP":
                tp_members[row["species_name"]][key] += 1
            elif row["label"] == "FP":
                fp_members[row["species_name"]][key] += 1

    with fn_csv.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("cause") != "stage3_1_trajectory_rejection":
                continue
            try:
                key = f"{row['species_name']}|{row['video_stem']}|{int(float(row['nearest_stage3_traj_id']))}"
            except Exception:
                continue
            fn_members[row["species_name"]][key] += 1

    return tp_members, fp_members, fn_members


def analyze(run_root: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = SCRIPT_DIR / "reports" / "day_v3_root_cause_analysis__20260328"
    trajectories = _load_trajectories(run_root)
    tp_members, fp_members, fn_members = _load_group_members(report_dir)

    traj_rows: list[dict[str, object]] = []
    detection_weighted_fn_fail: dict[str, Counter] = defaultdict(Counter)
    trajectory_group_metrics: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))

    all_keys_by_species: dict[str, set[str]] = defaultdict(set)
    for species_name, video_stem, traj_id in trajectories.keys():
        all_keys_by_species[species_name].add(f"{species_name}|{video_stem}|{traj_id}")

    for key, traj in trajectories.items():
        species_name, video_stem, traj_id = key
        hm = _hill_metrics(traj.crop_sums)
        row = {
            "species_name": species_name,
            "video_stem": video_stem,
            "traj_id": traj_id,
            "status": traj.status,
            "traj_size": traj.traj_size,
            "motion_xy": traj.motion_xy,
            "intensity_range": traj.intensity_range,
            "frame_start": traj.frame_start,
            "frame_end": traj.frame_end,
            "n_points": hm["n_points"],
            "peak_pos_frac": hm["peak_pos_frac"],
            "up_steps": hm["up_steps"],
            "down_steps": hm["down_steps"],
            "up_frac": hm["up_frac"],
            "down_frac": hm["down_frac"],
            "hill_ok": hm["hill_ok"],
            "fail_reason": hm["fail_reason"],
        }
        traj_key = f"{species_name}|{video_stem}|{traj_id}"
        groups: list[str] = []
        if tp_members[species_name][traj_key]:
            groups.append("tp")
        if fp_members[species_name][traj_key]:
            groups.append("fp")
        if fn_members[species_name][traj_key]:
            groups.append("fn_stage31")
            detection_weighted_fn_fail[species_name][str(hm["fail_reason"])] += fn_members[species_name][traj_key]
        if not groups and traj.status == "REJECT":
            groups.append("rejected_noise")
        row["groups"] = ",".join(groups)
        traj_rows.append(row)
        for group in groups:
            trajectory_group_metrics[species_name][group].append(row)

    out_csv = output_dir / "trajectory_hill_metrics.csv"
    with out_csv.open("w", newline="") as f:
        fields = list(traj_rows[0].keys()) if traj_rows else ["species_name"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in traj_rows:
            w.writerow(row)

    summary: dict[str, object] = {
        "run_root": str(run_root),
        "trajectory_metrics_csv": str(out_csv),
        "species": {},
    }
    for species_name, groups in sorted(trajectory_group_metrics.items()):
        species_summary: dict[str, object] = {
            "detection_weighted_fn_fail_reasons": dict(detection_weighted_fn_fail[species_name]),
            "groups": {},
        }
        for group_name, rows in sorted(groups.items()):
            species_summary["groups"][group_name] = {
                "n_trajectories": len(rows),
                "traj_size": _summary(r["traj_size"] for r in rows),
                "motion_xy": _summary(r["motion_xy"] for r in rows),
                "intensity_range": _summary(r["intensity_range"] for r in rows),
                "peak_pos_frac": _summary(r["peak_pos_frac"] for r in rows),
                "up_steps": _summary(r["up_steps"] for r in rows),
                "down_steps": _summary(r["down_steps"] for r in rows),
                "up_frac": _summary(r["up_frac"] for r in rows),
                "down_frac": _summary(r["down_frac"] for r in rows),
                "fail_reason_counts": dict(Counter(r["fail_reason"] for r in rows)),
            }
        summary["species"][species_name] = species_summary

    out_json = output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep dive into day-v3 Stage3.1 hill-shape behavior.")
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = ap.parse_args()
    summary = analyze(args.run_root, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f"\nsummary_json={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
