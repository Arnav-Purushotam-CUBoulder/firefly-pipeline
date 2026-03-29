#!/usr/bin/env python3
from __future__ import annotations

"""
Local convenience runner for `tools/train_multi_species_patch_classifiers.py`.

Edit the CONFIG section below and run this file (VSCode Run button).

Why this exists
---------------
- No CLI typing needed for local experiments.
- Lets you cap concurrency to e.g. 2 "at a time".
- Optionally oversubscribes a single GPU (2 trainings sharing 1 GPU) for quick
  throughput tests (may be slower / may OOM depending on settings).
"""

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


# =========================
# CONFIG (EDIT THESE)
# =========================

# Folder that contains species folders (e.g. <datasets_dir>/<species>/<version>/final dataset/...).
DATASETS_DIR = r"/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/patch training datasets and pipeline validation data/Integrated_prototype_datasets/single species datasets"

# Where to write outputs/ and logs/ locally (this should have lots of free space).
RUNROOT = r"/mnt/Samsung_SSD_2TB/firefly_patch_training_local_run"

# Train up to this many models concurrently.
CONCURRENT_JOBS = 2

# If you only have 1 GPU but want CONCURRENT_JOBS=2, set True to run 2 trainings on the same GPU.
ALLOW_GPU_OVERSUBSCRIBE = True

# DataLoader workers PER training process. Start with 4–8 locally; reduce if you see stalls.
NUM_WORKERS = 8

# Training hyperparams (match the main script defaults unless you want faster tests)
RESNET = "resnet18"
EPOCHS = 100
BATCH_SIZE = 128
LR = 3e-4

# Optional: set to e.g. "0" to force a specific GPU. Leave "" to use default.
CUDA_VISIBLE_DEVICES = ""

# If True, just prints discovery + planned jobs and exits.
DRY_RUN = False

# =========================


def _q(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def main() -> int:
    trainer = Path(__file__).with_name("train_multi_species_patch_classifiers.py")
    if not trainer.exists():
        raise SystemExit(f"Trainer not found: {trainer}")

    datasets_dir = Path(DATASETS_DIR).expanduser()
    if not datasets_dir.exists():
        raise SystemExit(f"DATASETS_DIR does not exist: {datasets_dir}\nEdit DATASETS_DIR in this file.")

    runroot = Path(RUNROOT).expanduser()
    out_dir = runroot / "outputs"
    logs_dir = runroot / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"local_train_{ts}.log"

    cmd = [
        sys.executable,
        "-u",
        str(trainer),
        "--root",
        str(runroot),
        "--datasets-dir",
        str(datasets_dir),
        "--output-dir",
        str(out_dir),
        "--gpus",
        str(int(CONCURRENT_JOBS)),
        "--num-workers",
        str(int(NUM_WORKERS)),
        "--resnet",
        str(RESNET),
        "--epochs",
        str(int(EPOCHS)),
        "--batch-size",
        str(int(BATCH_SIZE)),
        "--lr",
        str(float(LR)),
    ]
    if bool(ALLOW_GPU_OVERSUBSCRIBE):
        cmd.append("--allow-gpu-oversubscribe")
    if bool(DRY_RUN):
        cmd.append("--dry-run")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if str(CUDA_VISIBLE_DEVICES).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES).strip()

    print(f"[local-runner] datasets_dir={datasets_dir}")
    print(f"[local-runner] runroot={runroot}")
    print(f"[local-runner] log={log_path}")
    print(f"[local-runner] cmd:\n  {_q(cmd)}")

    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        return int(p.wait())


if __name__ == "__main__":
    raise SystemExit(main())

