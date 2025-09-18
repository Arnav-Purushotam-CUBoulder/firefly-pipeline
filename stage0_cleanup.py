#!/usr/bin/env python3
"""Stage 0 — optional cleanup of a species inference folder prior to pipeline run."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Tuple


def _resolve_named_dir(root: Path, target: str) -> Path:
    target_lower = target.lower()
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == target_lower:
            return child
    # Fall back to the literal name if it does not exist yet.
    return root / target


def cleanup_inference_root(
    root: Path,
    *,
    keep_dirs: Iterable[str] = ("ground truth", "original videos"),
    gt_filename: str = "gt.csv",
    verbose: bool = True,
) -> None:
    """Remove prior run artifacts so the pipeline starts from a clean slate.

    Steps performed:
      1. Delete every directory in *root* except those whose names are listed in
         *keep_dirs* (case-insensitive).
      2. Remove the existing ``gt.csv`` (if any) inside the ground-truth folder.
      3. Find a CSV file located directly under *root* and copy it into the
         ground-truth folder as ``gt.csv``.

    The function prints its actions when *verbose* is True.
    """

    root = Path(root)
    if not root.exists():
        if verbose:
            print(f"[stage0_cleanup] Skipped: root not found → {root}")
        return

    keep_lower = {name.lower() for name in keep_dirs}

    if verbose:
        print(f"[stage0_cleanup] Cleaning inference root: {root}")

    # 1) Remove directories not in keep list.
    for child in list(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.lower() in keep_lower:
            continue
        try:
            shutil.rmtree(child)
            if verbose:
                print(f"[stage0_cleanup] Removed directory: {child}")
        except Exception as exc:
            if verbose:
                print(f"[stage0_cleanup] WARNING: could not remove {child}: {exc}")

    # 2) Remove old ground-truth CSV.
    gt_dir = _resolve_named_dir(root, "ground truth")
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_csv_path = gt_dir / gt_filename
    if gt_csv_path.exists():
        try:
            gt_csv_path.unlink()
            if verbose:
                print(f"[stage0_cleanup] Deleted existing GT CSV: {gt_csv_path}")
        except Exception as exc:
            if verbose:
                print(f"[stage0_cleanup] WARNING: could not delete {gt_csv_path}: {exc}")

    # 3) Copy the latest CSV from the root into the ground-truth folder.
    csv_candidates = [p for p in root.glob('*.csv') if p.is_file()]
    if not csv_candidates:
        if verbose:
            print("[stage0_cleanup] NOTE: no root-level CSV found to seed ground truth.")
        return

    # Prefer the most recently modified file when multiple CSVs exist.
    src_csv = max(csv_candidates, key=lambda p: p.stat().st_mtime)
    if len(csv_candidates) > 1 and verbose:
        others = [p.name for p in csv_candidates if p != src_csv]
        print(f"[stage0_cleanup] WARNING: multiple CSVs found; using latest: {src_csv.name}. Others: {others}")

    try:
        shutil.copy2(src_csv, gt_csv_path)
        if verbose:
            print(f"[stage0_cleanup] Copied {src_csv.name} → {gt_csv_path}")
    except Exception as exc:
        if verbose:
            print(f"[stage0_cleanup] ERROR: failed to copy {src_csv} to {gt_csv_path}: {exc}")


__all__ = ["cleanup_inference_root"]
