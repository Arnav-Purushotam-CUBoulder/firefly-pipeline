#!/usr/bin/env python3
from __future__ import annotations

"""
Convenience runner: Stage 2 (train) + Stage 3 (test) only.

This avoids re-running Stage 1 ingestion by delegating to orchestrator.py with:
  --skip-ingest

Outputs are written exactly where orchestrator.py would write them (model zoo,
inference outputs, history jsonl), since orchestrator.py is still the executor.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


# ── USER SETTINGS (VS Code friendly) ─────────────────────────────
# If you run this file directly from VS Code (Run Python File), you can set these
# defaults here and avoid any CLI args. CLI args still override these.
# Defaulted to your current Photinus Knulli run. Edit as needed.
DEFAULT_SPECIES_NAME = "photinus-knulli"  # e.g. "photinus-knulli"
DEFAULT_OBSERVED_DIR = "/mnt/Samsung_SSD_2TB/integrated prototype raw videos/Photinus Knulli"
DEFAULT_TYPE_OF_VIDEO = "night"  # "day" | "night"
DEFAULT_TRAIN_PAIR_FRACTION = 0.8
DEFAULT_ROOT = "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data"


def _orchestrator_path() -> Path:
    return (Path(__file__).resolve().parent / "orchestrator.py").resolve()


def main() -> int:
    p = argparse.ArgumentParser(description="Run Stage 2+3 only (skip ingestion).")
    p.add_argument(
        "--species-name",
        type=str,
        default=str(DEFAULT_SPECIES_NAME),
        help="Species name (e.g. photinus-knulli). Must match orchestrator naming.",
    )
    p.add_argument(
        "--observed-dir",
        type=str,
        default=str(DEFAULT_OBSERVED_DIR),
        help="Folder containing observed videos + annotator CSVs (for held-out validation videos).",
    )
    p.add_argument(
        "--type-of-video",
        default=str(DEFAULT_TYPE_OF_VIDEO),
        type=str,
        help="day|night (routes to day_time_dataset or night_time_dataset). Default: night",
    )
    p.add_argument(
        "--train-pair-fraction",
        default=float(DEFAULT_TRAIN_PAIR_FRACTION),
        type=float,
        help="Fraction of pairs assigned to training (rest held out for validation/testing). Default: 0.8",
    )
    p.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT),
        help="Integrated prototype root (same as orchestrator --root).",
    )
    p.add_argument("--dry-run", action="store_true", default=False)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true", default=False, help="Run Stage 2 only (skip Stage 3).")
    mode.add_argument("--eval-only", action="store_true", default=False, help="Run Stage 3 only (skip Stage 2).")

    args = p.parse_args()

    if not str(args.species_name or "").strip():
        raise SystemExit("Set DEFAULT_SPECIES_NAME at top of file or pass --species-name.")
    if not str(args.observed_dir or "").strip():
        raise SystemExit("Set DEFAULT_OBSERVED_DIR at top of file or pass --observed-dir.")
    if not str(args.root or "").strip():
        raise SystemExit("Set DEFAULT_ROOT at top of file or pass --root.")

    orch = _orchestrator_path()
    if not orch.exists():
        raise SystemExit(f"orchestrator.py not found at: {orch}")

    cmd: list[str] = [
        sys.executable,
        str(orch),
        "--species-name",
        str(args.species_name),
        "--observed-dir",
        str(args.observed_dir),
        "--type-of-video",
        str(args.type_of_video),
        "--train-pair-fraction",
        str(float(args.train_pair_fraction)),
        "--root",
        str(args.root),
        "--skip-ingest",
    ]

    if args.train_only:
        cmd.append("--skip-test")
    if args.eval_only:
        cmd.append("--skip-train")
    if args.dry_run:
        cmd.append("--dry-run")

    print("[run_stage2_stage3_only] exec:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
