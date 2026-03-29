#!/usr/bin/env python3
from __future__ import annotations

"""
One-click Greeni workflow:
1) Ingest only the training-half Greeni videos from the split catalog.
2) Retrain only the day/global patch-classifier model used by the scaling runner.

Safe to delete later.
"""

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path("/home/guest/Desktop/arnav's files/firefly pipeline")
INGESTOR_ONLY_PATH = REPO_ROOT / "integrated ingestor-trainer-tester orchestrator" / "ingestor_only.py"
SCALING_RUNNER_PATH = REPO_ROOT / "tmp scripts for scaling species" / "tmp_day_night_combo_train_and_infer.py"

RAW_SPECIES_DIR_NAME = "day_Photinus greeni"
DRY_RUN = False


def _load_module(module_name: str, path: Path):
    for extra in (REPO_ROOT, REPO_ROOT / "integrated ingestor-trainer-tester orchestrator"):
        s = str(extra)
        if s not in sys.path:
            sys.path.insert(0, s)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    if not INGESTOR_ONLY_PATH.exists():
        raise FileNotFoundError(INGESTOR_ONLY_PATH)
    if not SCALING_RUNNER_PATH.exists():
        raise FileNotFoundError(SCALING_RUNNER_PATH)

    print(f"[greeni-run] ingest script: {INGESTOR_ONLY_PATH}")
    print(f"[greeni-run] train script: {SCALING_RUNNER_PATH}")

    ingestor = _load_module("ingestor_only_greeni_run", INGESTOR_ONLY_PATH)
    scaling = _load_module("tmp_day_night_combo_train_and_infer_greeni_run", SCALING_RUNNER_PATH)

    print("[greeni-run] Step 1/2: ingest Greeni training-only videos from split catalog")
    ingestor.DRY_RUN = bool(DRY_RUN)
    ingest_argv = ["--only-species", RAW_SPECIES_DIR_NAME]
    if bool(DRY_RUN):
        ingest_argv.insert(0, "--dry-run")
    ingest_rc = ingestor.main(ingest_argv)
    if int(ingest_rc) != 0:
        print(f"[greeni-run] ingestion failed with exit code {ingest_rc}")
        return int(ingest_rc)

    print("[greeni-run] Step 2/2: retrain day/global patch-classifier model only")
    scaling.RUN_INGESTION_FROM_SCRATCH = False
    scaling.RUN_MODEL_TRAINING = True
    scaling.RUN_DAY_MODEL_TRAINING = True
    scaling.RUN_NIGHT_MODEL_TRAINING = False
    scaling.TRAIN_GLOBAL_MODELS = True
    scaling.TRAIN_LEAVEOUT_MODELS = False

    scaling.RUN_BASELINE_METHODS_INFERENCE = False
    scaling.RUN_LAB_BASELINE = False
    scaling.RUN_RAPHAEL_BASELINE = False
    scaling.RUN_DAY_PIPELINE_INFERENCE = False
    scaling.RUN_NIGHT_PIPELINE_INFERENCE = False
    scaling.RUN_GLOBAL_MODEL_INFERENCE = False
    scaling.RUN_LEAVEOUT_MODEL_INFERENCE = False

    scaling.DAY_INFERENCE_SPECIES_SWITCHES = dict(scaling.DAY_INFERENCE_SPECIES_SWITCHES)
    scaling.DAY_INFERENCE_SPECIES_SWITCHES["photinus-greeni"] = False
    scaling.DRY_RUN = bool(DRY_RUN)

    train_rc = scaling.main(["--dry-run"] if bool(DRY_RUN) else [])
    if int(train_rc) != 0:
        print(f"[greeni-run] training failed with exit code {train_rc}")
        return int(train_rc)

    print("[greeni-run] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
