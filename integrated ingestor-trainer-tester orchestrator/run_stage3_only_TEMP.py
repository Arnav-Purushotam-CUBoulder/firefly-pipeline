#!/usr/bin/env python3
from __future__ import annotations

"""
TEMP convenience runner: Stage 3 only (pipeline eval), without re-ingestion or re-training.

Run this directly from VS Code ("Run Python File") and it will:
  - reuse the latest trained models from the model zoo
  - run the gateway + validation scoring on the held-out validation videos
  - write outputs to the exact same folders as orchestrator.py normally would

Delete this file whenever you no longer need it.
"""

import sys
from pathlib import Path


# ── USER SETTINGS ─────────────────────────────────────────────────────────────
SPECIES_NAME = "photinus-knulli"
OBSERVED_DIR = "/mnt/Samsung_SSD_2TB/integrated prototype raw videos/Photinus Knulli"
TYPE_OF_VIDEO = "night"  # "day" | "night"
TRAIN_PAIR_FRACTION = 0.8
ROOT = "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data"
DRY_RUN = False

# Optional: reduce extra work during Stage 3 (leave as-is for full parity).
# If you're fighting memory pressure, set RUN_BASELINES = False.
RUN_BASELINES = True


def main() -> int:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    import orchestrator as orch  # type: ignore

    if not RUN_BASELINES:
        orch.RUN_BASELINE_EVAL = False
        orch.RUN_BASELINE_LAB_METHOD = False
        orch.RUN_BASELINE_RAPHAEL_METHOD = False

    argv = [
        "orchestrator.py",
        "--species-name",
        str(SPECIES_NAME),
        "--observed-dir",
        str(OBSERVED_DIR),
        "--type-of-video",
        str(TYPE_OF_VIDEO),
        "--train-pair-fraction",
        str(float(TRAIN_PAIR_FRACTION)),
        "--root",
        str(ROOT),
        "--skip-ingest",
        "--skip-train",
    ]
    if DRY_RUN:
        argv.append("--dry-run")

    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        return int(orch.main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())

