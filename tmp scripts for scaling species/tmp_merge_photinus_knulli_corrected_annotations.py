#!/bin/bin/env python3
"""
Merge corrected annotation CSVs into root raw-video annotation CSVs.

Configure the species/folder-specific values in the global variables near the
top of this file. The script can either:
1. Auto-discover targets from a corrected folder, or
2. Use an explicit target list if you need manual pairing.

Behavior:
1. Discover root CSV / corrected CSV / video pairs.
2. Create timestamped backups of the original root CSVs and corrected CSVs.
3. Append corrected annotations to the existing root annotations.
4. Sort by frame.
5. Remove exact duplicates.
6. Run Stage-9-style same-frame distance dedupe using crop BGR-sum weight from
   the original video, keeping the heaviest point in each <=4 px component.
7. Overwrite the root CSVs with the merged result.
8. Save a JSON report with merge analytics.
9. Optionally delete the corrected folder after success.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


@dataclass(frozen=True)
class MergeTarget:
    video_path: Path
    root_csv_path: Path
    corrected_csv_path: Path


# ---------------------------------------------------------------------------
# Edit only this block for future runs.
# ---------------------------------------------------------------------------

RAW_DIR = Path("/mnt/Samsung_SSD_2TB/integrated prototype raw videos/night_Photinus Knulli")
CORRECTED_SUBDIR_NAME = "corrected"
CORRECTED_DIR = RAW_DIR / CORRECTED_SUBDIR_NAME

ROOT_CSV_SUFFIX = ".csv"
CORRECTED_FILE_SUFFIX = ".firefly.csv"
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

# Optional filter. Use basenames without any suffixes. Example:
# TARGET_BASENAMES = ["video_a", "video_b"]
TARGET_BASENAMES: Optional[List[str]] = None

# Leave as None for normal auto-discovery. If a species folder has irregular
# filenames, you can manually specify exact file pairs here.
EXPLICIT_TARGETS: Optional[List[MergeTarget]] = None

BACKUP_DIR_NAME = "_tmp_annotation_merge_backups"
REPORT_PREFIX = "tmp_annotation_merge_report"
DELETE_CORRECTED_DIR_AFTER_SUCCESS = True

STAGE9_CROP_W = 10
STAGE9_CROP_H = 10
STAGE9_GT_DEDUPE_DIST_PX = 4.0

# ---------------------------------------------------------------------------


class ProgressBar:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(1, int(total))
        self.desc = desc
        self.current = 0
        self.last_emit = 0.0
        self.width = 36
        self._tqdm = tqdm(total=self.total, desc=desc, unit="frame") if tqdm else None

    def update(self, n: int = 1) -> None:
        self.current += n
        if self._tqdm is not None:
            self._tqdm.update(n)
            return
        now = time.time()
        if self.current < self.total and (now - self.last_emit) < 0.05:
            return
        self.last_emit = now
        frac = min(1.0, self.current / float(self.total))
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stdout.write(f"\r{self.desc:28s} [{bar}] {frac*100:6.2f}%")
        sys.stdout.flush()
        if self.current >= self.total:
            sys.stdout.write("\n")

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
        elif self.current < self.total:
            self.current = self.total
            self.update(0)


def _require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _normalized_basename(name: str) -> str:
    base = Path(name).name
    if CORRECTED_FILE_SUFFIX and base.endswith(CORRECTED_FILE_SUFFIX):
        base = base[: -len(CORRECTED_FILE_SUFFIX)]
    elif ROOT_CSV_SUFFIX and base.endswith(ROOT_CSV_SUFFIX):
        base = base[: -len(ROOT_CSV_SUFFIX)]
    elif base.endswith(".csv"):
        base = base[:-4]
    return base


def _discover_corrected_csvs() -> List[Path]:
    _require_exists(CORRECTED_DIR)
    candidates = sorted(path for path in CORRECTED_DIR.glob("*.csv") if path.is_file())
    if CORRECTED_FILE_SUFFIX:
        candidates = [path for path in candidates if path.name.endswith(CORRECTED_FILE_SUFFIX)]
    if TARGET_BASENAMES is None:
        return candidates
    wanted = {_normalized_basename(name) for name in TARGET_BASENAMES}
    selected = [path for path in candidates if _normalized_basename(path.name) in wanted]
    missing = sorted(wanted - {_normalized_basename(path.name) for path in selected})
    if missing:
        raise FileNotFoundError(f"Missing corrected CSVs for basenames: {missing}")
    return selected


def _find_video_for_basename(base: str) -> Path:
    matches = [RAW_DIR / f"{base}{ext}" for ext in VIDEO_EXTENSIONS if (RAW_DIR / f"{base}{ext}").exists()]
    if not matches:
        raise FileNotFoundError(f"No video found for basename: {base}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple videos found for basename {base}: {matches}")
    return matches[0]


def _resolve_targets() -> List[MergeTarget]:
    if EXPLICIT_TARGETS is not None:
        return list(EXPLICIT_TARGETS)

    corrected_csvs = _discover_corrected_csvs()
    if not corrected_csvs:
        raise RuntimeError(f"No corrected CSVs found in {CORRECTED_DIR}")

    targets: List[MergeTarget] = []
    for corrected_csv_path in corrected_csvs:
        base = _normalized_basename(corrected_csv_path.name)
        root_csv_path = RAW_DIR / f"{base}{ROOT_CSV_SUFFIX}"
        video_path = _find_video_for_basename(base)
        targets.append(
            MergeTarget(
                video_path=video_path,
                root_csv_path=root_csv_path,
                corrected_csv_path=corrected_csv_path,
            )
        )
    return targets


def _center_crop_clamped(frame: np.ndarray, cx: float, cy: float, w: int, h: int) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x0 = max(0, min(x0, frame_w - w))
    y0 = max(0, min(y0, frame_h - h))
    return frame[y0 : y0 + h, x0 : x0 + w].copy()


def _crop_weight(frame: np.ndarray, x: int, y: int) -> float:
    crop = _center_crop_clamped(frame, float(x), float(y), STAGE9_CROP_W, STAGE9_CROP_H)
    if crop.size == 0:
        return 0.0
    return float(crop.sum())


def _load_csv(csv_path: Path, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["x", "y", "w", "h", "frame"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    df = df[required].copy()
    for col in required:
        df[col] = df[col].round().astype(int)
    df["_source"] = source_label
    return df


def _make_backup_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d__%H%M%S")
    backup_root = RAW_DIR / BACKUP_DIR_NAME
    backup_dir = backup_root / ts
    backup_dir.mkdir(parents=True, exist_ok=False)
    return backup_dir


def _backup_inputs(backup_dir: Path, targets: Iterable[MergeTarget]) -> None:
    for target in targets:
        shutil.copy2(target.root_csv_path, backup_dir / target.root_csv_path.name)
        shutil.copy2(target.corrected_csv_path, backup_dir / target.corrected_csv_path.name)


def _build_components(rows: List[dict], weights: List[float], dist_threshold_px: float) -> Tuple[List[int], int, Dict[str, int]]:
    n = len(rows)
    if n <= 1:
        return list(range(n)), 0, {"root": 0, "corrected": 0}

    parent = list(range(n))
    rank = [0] * n
    thr2 = float(dist_threshold_px) * float(dist_threshold_px)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i in range(n):
        xi = float(rows[i]["x"])
        yi = float(rows[i]["y"])
        for j in range(i + 1, n):
            dx = xi - float(rows[j]["x"])
            dy = yi - float(rows[j]["y"])
            if dx * dx + dy * dy <= thr2:
                union(i, j)

    comps: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        comps[find(idx)].append(idx)

    kept_indices: List[int] = []
    removed_by_source = {"root": 0, "corrected": 0}
    for comp in comps.values():
        best_idx = max(comp, key=lambda idx: (weights[idx], -idx))
        kept_indices.append(best_idx)
        for idx in comp:
            if idx != best_idx:
                removed_by_source[str(rows[idx]["_source"])] += 1

    kept_indices.sort()
    removed_total = n - len(kept_indices)
    return kept_indices, removed_total, removed_by_source


def _stage9_style_dedupe(video_path: Path, merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if merged_df.empty:
        return merged_df.copy(), {
            "proximity_duplicates_removed": 0,
            "proximity_duplicates_removed_root": 0,
            "proximity_duplicates_removed_corrected": 0,
            "final_rows_from_root": 0,
            "final_rows_from_corrected": 0,
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for dedupe weighting: {video_path}")

    frame_groups: Dict[int, List[dict]] = defaultdict(list)
    for row in merged_df.to_dict("records"):
        frame_groups[int(row["frame"])].append(row)

    max_frame = int(max(frame_groups))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (max_frame + 1)
    limit = min(total_frames, max_frame + 1)
    progress = ProgressBar(total=limit, desc=f"dedupe {video_path.stem[-14:]}")

    deduped_rows: List[dict] = []
    prox_removed_total = 0
    prox_removed_root = 0
    prox_removed_corrected = 0

    frame_idx = 0
    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        rows = frame_groups.get(frame_idx, [])
        if rows:
            weights = [_crop_weight(frame, int(r["x"]), int(r["y"])) for r in rows]
            kept_indices, removed_total, removed_by_source = _build_components(
                rows=rows,
                weights=weights,
                dist_threshold_px=STAGE9_GT_DEDUPE_DIST_PX,
            )
            for idx in kept_indices:
                deduped_rows.append(rows[idx])
            prox_removed_total += removed_total
            prox_removed_root += removed_by_source["root"]
            prox_removed_corrected += removed_by_source["corrected"]
        frame_idx += 1
        progress.update(1)

    progress.close()
    cap.release()

    deduped_df = pd.DataFrame(deduped_rows)
    if deduped_df.empty:
        deduped_df = pd.DataFrame(columns=list(merged_df.columns))
    deduped_df = deduped_df.sort_values(["frame", "x", "y", "_source"]).reset_index(drop=True)

    final_rows_from_root = int((deduped_df["_source"] == "root").sum()) if "_source" in deduped_df else 0
    final_rows_from_corrected = int((deduped_df["_source"] == "corrected").sum()) if "_source" in deduped_df else 0

    return deduped_df, {
        "proximity_duplicates_removed": int(prox_removed_total),
        "proximity_duplicates_removed_root": int(prox_removed_root),
        "proximity_duplicates_removed_corrected": int(prox_removed_corrected),
        "final_rows_from_root": final_rows_from_root,
        "final_rows_from_corrected": final_rows_from_corrected,
    }


def _frame_stats(df: pd.DataFrame) -> Dict[str, int | None]:
    if df.empty:
        return {
            "rows": 0,
            "unique_frames": 0,
            "first_frame": None,
            "last_frame": None,
        }
    return {
        "rows": int(len(df)),
        "unique_frames": int(df["frame"].nunique()),
        "first_frame": int(df["frame"].min()),
        "last_frame": int(df["frame"].max()),
    }


def _merge_target(target: MergeTarget) -> Dict[str, object]:
    print(f"\n[merge] {target.root_csv_path.name}")
    old_df = _load_csv(target.root_csv_path, "root")
    corrected_df = _load_csv(target.corrected_csv_path, "corrected")

    old_keys = set(map(tuple, old_df[["frame", "x", "y", "w", "h"]].itertuples(index=False, name=None)))
    corrected_keys = set(map(tuple, corrected_df[["frame", "x", "y", "w", "h"]].itertuples(index=False, name=None)))
    exact_cross_source_overlap = len(old_keys & corrected_keys)

    combined_df = pd.concat([old_df, corrected_df], ignore_index=True)
    combined_df["_source_rank"] = combined_df["_source"].map({"root": 0, "corrected": 1}).fillna(9).astype(int)
    combined_df = combined_df.sort_values(["frame", "_source_rank", "x", "y", "w", "h"]).reset_index(drop=True)

    exact_duplicate_mask = combined_df.duplicated(subset=["frame", "x", "y", "w", "h"], keep="first")
    exact_duplicates_removed = int(exact_duplicate_mask.sum())
    combined_exact_df = combined_df.loc[~exact_duplicate_mask].copy().reset_index(drop=True)

    deduped_df, dedupe_stats = _stage9_style_dedupe(target.video_path, combined_exact_df)
    final_df = deduped_df[["x", "y", "w", "h", "frame"]].copy()
    final_df = final_df.sort_values(["frame", "x", "y"]).reset_index(drop=True)
    final_df.to_csv(target.root_csv_path, index=False)

    report = {
        "video_path": str(target.video_path),
        "root_csv_path": str(target.root_csv_path),
        "corrected_csv_path": str(target.corrected_csv_path),
        "old": _frame_stats(old_df),
        "corrected": _frame_stats(corrected_df),
        "combined_before_dedupe": _frame_stats(combined_df),
        "combined_after_exact_dedupe": _frame_stats(combined_exact_df),
        "final": _frame_stats(final_df),
        "exact_duplicates_removed": exact_duplicates_removed,
        "exact_cross_source_overlap_unique_rows": exact_cross_source_overlap,
        "final_net_new_rows_vs_old": int(len(final_df) - len(old_df)),
    }
    report.update(dedupe_stats)
    return report


def _print_report(report_path: Path, per_file_reports: List[Dict[str, object]], corrected_deleted: bool) -> None:
    print("\n=== Merge Summary ===")
    print(f"report: {report_path}")
    print(f"corrected_deleted: {corrected_deleted}")
    print()
    header = (
        "file",
        "old",
        "new",
        "exact_rm",
        "prox_rm",
        "final",
        "net_new",
        "root_kept",
        "corr_kept",
    )
    print(
        f"{header[0]:24s} {header[1]:>7s} {header[2]:>7s} {header[3]:>9s} "
        f"{header[4]:>8s} {header[5]:>7s} {header[6]:>8s} {header[7]:>10s} {header[8]:>10s}"
    )
    print("-" * 100)
    for item in per_file_reports:
        file_name = Path(str(item["root_csv_path"])).name
        print(
            f"{file_name[-24:]:24s} "
            f"{int(item['old']['rows']):7d} "
            f"{int(item['corrected']['rows']):7d} "
            f"{int(item['exact_duplicates_removed']):9d} "
            f"{int(item['proximity_duplicates_removed']):8d} "
            f"{int(item['final']['rows']):7d} "
            f"{int(item['final_net_new_rows_vs_old']):8d} "
            f"{int(item['final_rows_from_root']):10d} "
            f"{int(item['final_rows_from_corrected']):10d}"
        )
    print("-" * 100)


def main() -> int:
    targets = _resolve_targets()

    print(f"[config] raw_dir={RAW_DIR}")
    print(f"[config] corrected_dir={CORRECTED_DIR}")
    print(f"[config] discovered_targets={len(targets)}")
    for target in targets:
        print(f"[target] {target.root_csv_path.name}")

    for target in targets:
        _require_exists(target.video_path)
        _require_exists(target.root_csv_path)
        _require_exists(target.corrected_csv_path)

    backup_dir = _make_backup_dir()
    print(f"[backup] {backup_dir}")
    _backup_inputs(backup_dir, targets)

    per_file_reports: List[Dict[str, object]] = []
    for idx, target in enumerate(targets, start=1):
        print(f"[run] {idx}/{len(targets)}")
        per_file_reports.append(_merge_target(target))

    corrected_deleted = False
    if DELETE_CORRECTED_DIR_AFTER_SUCCESS and CORRECTED_DIR.exists():
        shutil.rmtree(CORRECTED_DIR)
        corrected_deleted = True

    report_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw_dir": str(RAW_DIR),
        "corrected_dir": str(CORRECTED_DIR),
        "backup_dir": str(backup_dir),
        "corrected_file_suffix": CORRECTED_FILE_SUFFIX,
        "root_csv_suffix": ROOT_CSV_SUFFIX,
        "target_basenames": TARGET_BASENAMES,
        "stage9_crop_w": STAGE9_CROP_W,
        "stage9_crop_h": STAGE9_CROP_H,
        "stage9_gt_dedupe_dist_px": STAGE9_GT_DEDUPE_DIST_PX,
        "corrected_dir_deleted": corrected_deleted,
        "files": per_file_reports,
    }
    report_path = RAW_DIR / f"{REPORT_PREFIX}__{datetime.now().strftime('%Y%m%d__%H%M%S')}.json"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    _print_report(report_path, per_file_reports, corrected_deleted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
