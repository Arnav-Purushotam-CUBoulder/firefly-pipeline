#!/usr/bin/env python3
"""
Split firefly/background images into train/val/test folders
and print a per-class + overall summary when done.
"""

# ─── GLOBAL CONFIG (edit) ──────────────────────────────────────────────
SRC_DIR   = '/Volumes/DL Project SSD/back subtracted method data/Firefly flash data science project data/forresti data' # has firefly/ and background/
DEST_DIR  = '/Users/arnavps/Desktop/tremulans and forresti individual datasets/final forresti dataset'    # new root with train/ val/ test/
TRAIN_PCT = 0.50
VAL_PCT   = 0.25
TEST_PCT  = 0.25               # splits must sum to 1.0
SEED      = 1337

# NEW: maximum number of images to copy per class (None for no cap)
MAX_FIREFLY    = 10000   # e.g., 12000
MAX_BACKGROUND = 100000   # e.g., 12000
# ───────────────────────────────────────────────────────────────────────

import random, shutil, os
from pathlib import Path
from tqdm import tqdm

assert abs((TRAIN_PCT + VAL_PCT + TEST_PCT) - 1.0) < 1e-6, "splits ≠ 1.0"

random.seed(SEED)
src_root  = Path(SRC_DIR).expanduser().resolve()
dest_root = Path(DEST_DIR).expanduser().resolve()
splits    = {"train": TRAIN_PCT, "val": VAL_PCT, "test": TEST_PCT}

metrics = {}  # {class: {split: count, … , total: count}}

for cls_dir in [p for p in src_root.iterdir()
                if p.is_dir() and p.name not in splits]:
    files = sorted(f for f in cls_dir.iterdir() if f.is_file())
    random.shuffle(files)

    # ── apply per-class caps (optional) ──
    if cls_dir.name == 'firefly' and MAX_FIREFLY is not None:
        files = files[:int(MAX_FIREFLY)]
    if cls_dir.name == 'background' and MAX_BACKGROUND is not None:
        files = files[:int(MAX_BACKGROUND)]

    n_total = len(files)

    n_train = int(n_total * TRAIN_PCT)
    n_val   = int(n_total * VAL_PCT)

    split_files = {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
        "test":  files[n_train + n_val:]
    }

    metrics[cls_dir.name] = {"total": n_total}

    for split_name, file_list in split_files.items():
        dst_dir = dest_root / split_name / cls_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        metrics[cls_dir.name][split_name] = len(file_list)

        print(f"Copying {len(file_list)} {cls_dir.name!r} → {split_name}/")
        for src_path in tqdm(file_list, desc=f"{cls_dir.name}->{split_name}",
                             unit="img", ncols=80):
            shutil.copy2(src_path, dst_dir / src_path.name)

# ─── summary ───────────────────────────────────────────────────────────
print("\n=== split summary ===")
grand = {"train":0, "val":0, "test":0, "total":0}
for cls, m in metrics.items():
    grand = {k: grand.get(k,0)+m.get(k,0) for k in grand}
    pct = lambda x: f"{100*x/m['total']:.1f}%"
    print(f"{cls:11} | "
          f"train {m['train']:5d} ({pct(m['train'])}) | "
          f"val {m['val']:5d} ({pct(m['val'])}) | "
          f"test {m['test']:5d} ({pct(m['test'])}) | "
          f"total {m['total']:5d}")

print(f"-----------+---------------------------------------------")
print(f"all classes | "
      f"train {grand['train']:5d} | "
      f"val {grand['val']:5d} | "
      f"test {grand['test']:5d} | "
      f"total {grand['total']:5d}")

print(f"\n✅ dataset split written to {dest_root}")
