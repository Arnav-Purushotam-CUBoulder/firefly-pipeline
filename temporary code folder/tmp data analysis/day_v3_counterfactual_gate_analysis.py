#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT_DEFAULT = Path(
    "/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/"
    "tmp_day_night_combo_train_and_infer__20260328__005102"
)
OUT_DIR_DEFAULT = SCRIPT_DIR / "reports" / "day_v3_counterfactual_gate_analysis__20260328"
ROOT_CAUSE_REPORT_DIR = SCRIPT_DIR / "reports" / "day_v3_root_cause_analysis__20260328"
HILL_SCRIPT = SCRIPT_DIR / "day_v3_hill_shape_deep_dive.py"


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "bicell_range2500": {"min_range": 2500},
    "bicell_range2000": {"min_range": 2000},
    "hill1_2": {"min_up_steps": 1, "min_down_steps": 2},
    "hill1_1": {"min_up_steps": 1, "min_down_steps": 1},
    "hill_off": {"require_hill": False},
}


def _load_hill_module():
    spec = importlib.util.spec_from_file_location("day_v3_hill_shape_deep_dive", HILL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {HILL_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _traj_key(species_name: str, video_stem: str, traj_id: int) -> str:
    return f"{species_name}|{video_stem}|{int(traj_id)}"


def _variant_eval(mod, crop_sums: list[int], variant_name: str) -> dict[str, Any]:
    if variant_name == "baseline":
        return mod._hill_metrics(crop_sums)
    if variant_name == "bicell_range2500":
        return mod._hill_metrics(crop_sums, min_range=2500)
    if variant_name == "bicell_range2000":
        return mod._hill_metrics(crop_sums, min_range=2000)
    if variant_name == "hill1_2":
        return mod._hill_metrics(crop_sums, min_up_steps=1, min_down_steps=2)
    if variant_name == "hill1_1":
        return mod._hill_metrics(crop_sums, min_up_steps=1, min_down_steps=1)
    if variant_name == "hill_off":
        hm = mod._hill_metrics(crop_sums, min_up_steps=0, min_down_steps=0, min_monotonic_frac=0.0)
        hm["hill_ok"] = hm["fail_reason"] not in {"traj_too_short", "intensity_range_too_low"}
        return hm
    raise KeyError(variant_name)


def analyze(run_root: Path, out_dir: Path) -> dict[str, Any]:
    mod = _load_hill_module()
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories = mod._load_trajectories(run_root)
    tp_members, fp_members, fn_members = mod._load_group_members(ROOT_CAUSE_REPORT_DIR)

    species_names = sorted({species for species, _, _ in trajectories.keys()})
    payload: dict[str, Any] = {
        "run_root": str(run_root),
        "variants": VARIANTS,
        "species": {},
    }

    for species_name in species_names:
        species_trajs = [t for t in trajectories.values() if t.species_name == species_name]
        species_summary: dict[str, Any] = {"variants": {}}
        for variant_name in VARIANTS:
            if species_name == "bicellonycha-wickershamorum":
                if variant_name.startswith("hill"):
                    continue
            else:
                if variant_name.startswith("bicell_"):
                    continue

            rows = {}
            for group_name in ("fn_stage31", "rejected_noise"):
                if group_name == "fn_stage31":
                    group_trajs = [
                        t for t in species_trajs
                        if fn_members[species_name][_traj_key(t.species_name, t.video_stem, t.traj_id)] > 0
                    ]
                    weight_fn = lambda t: fn_members[species_name][_traj_key(t.species_name, t.video_stem, t.traj_id)]
                else:
                    group_trajs = [
                        t for t in species_trajs
                        if t.status == "REJECT"
                        and fn_members[species_name][_traj_key(t.species_name, t.video_stem, t.traj_id)] == 0
                    ]
                    weight_fn = lambda t: int(t.traj_size)

                passed = 0
                total = 0
                for traj in group_trajs:
                    weight = int(weight_fn(traj))
                    total += weight
                    if _variant_eval(mod, traj.crop_sums, variant_name)["hill_ok"]:
                        passed += weight
                rows[group_name] = {
                    "passed_weight": passed,
                    "total_weight": total,
                    "pass_rate": (float(passed) / float(total)) if total else 0.0,
                }
            species_summary["variants"][variant_name] = rows
        payload["species"][species_name] = species_summary

    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    payload = analyze(RUN_ROOT_DEFAULT, OUT_DIR_DEFAULT)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
