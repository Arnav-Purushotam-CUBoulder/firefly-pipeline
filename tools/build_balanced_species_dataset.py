#!/usr/bin/env python3
"""
Build a species-balanced dataset (≈50:50 Pyralis:Frontalis) from verified pools.

Given two roots (each containing 'firefly/' and 'background/' patch folders),
this script assembles four output folders under a destination directory:
  - pyralis_firefly
  - pyralis_background
  - frontalis_firefly
  - frontalis_background

By default, it now samples the same number of patches for each of the four
output folders, where that number equals the minimum available count among the
four corresponding source folders (pyralis/firefly, pyralis/background,
frontalis/firefly, frontalis/background). Files are chosen uniformly at random.

Usage:
  1) Edit the globals below (PYRALIS_DIR, FRONTALIS_DIR, DEST_DIR, etc.)
  2) Run: python build_balanced_species_dataset.py

Notes:
  - Uses copying by default; pass --link to create symlinks instead of copying.
  - If filenames collide across sources, a numeric suffix is appended.
  - Image extensions recognized: .png, .jpg, .jpeg, .bmp, .tif, .tiff
"""

from __future__ import annotations
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ─── USER GLOBALS ─────────────────────────────────────────────────────────────
# Set these three paths before running.
PYRALIS_DIR   = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset/initial dataset')      # contains firefly/ and background/
FRONTALIS_DIR = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/tremulans and forresti and frontalis individual datasets/final frontalis dataset/total')    # contains firefly/ and background/
DEST_DIR      = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/standardization function data/frontalis and pyrallis balanced data')

# Optional knobs
LINK       = False    # True → symlink files instead of copying
ZIP_OUTPUT = False    # True → create a .zip archive of DEST_DIR when done
SEED       = 1337
# ─────────────────────────────────────────────────────────────────────────────


def list_images(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def sample_paths(paths: List[Path], n: int, rng: random.Random) -> List[Path]:
    if n <= 0:
        return []
    if n >= len(paths):
        out = paths.copy()
        rng.shuffle(out)
        return out
    return rng.sample(paths, n)


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def unique_dest(dest_dir: Path, src: Path) -> Path:
    base = src.stem
    ext = src.suffix
    out = dest_dir / (base + ext)
    if not out.exists():
        return out
    k = 1
    while True:
        cand = dest_dir / f"{base}_{k}{ext}"
        if not cand.exists():
            return cand
        k += 1


def copy_or_link(src: Path, dst: Path, link: bool) -> None:
    if link:
        # Create relative symlink when possible
        try:
            rel = os.path.relpath(src, start=dst.parent)
            os.symlink(rel, dst)
        except OSError:
            # Fallback to absolute symlink
            os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def build_dataset(pyralis_dir: Path, frontalis_dir: Path, dest_dir: Path,
                  link: bool, seed: int) -> dict:
    rng = random.Random(seed)

    # Validate inputs
    for d, label in [(pyralis_dir, "pyralis"), (frontalis_dir, "frontalis")]:
        if not d.exists():
            raise SystemExit(f"{label} dir not found: {d}")
        if not (d / "firefly").is_dir() or not (d / "background").is_dir():
            raise SystemExit(f"{label} dir must contain 'firefly' and 'background' subfolders: {d}")

    # Collect files
    p_firefly = list_images(pyralis_dir / "firefly")
    p_bg      = list_images(pyralis_dir / "background")
    f_firefly = list_images(frontalis_dir / "firefly")
    f_bg      = list_images(frontalis_dir / "background")

    # Determine uniform per-class target = minimum across all four source folders
    counts = {
        'pyralis_firefly': len(p_firefly),
        'pyralis_background': len(p_bg),
        'frontalis_firefly': len(f_firefly),
        'frontalis_background': len(f_bg),
    }
    if any(c == 0 for c in counts.values()):
        raise SystemExit("One or more source folders are empty – cannot create balanced four-way dataset.")
    per_class_target = min(counts.values())

    # Sample
    chosen = {
        "pyralis_firefly":      sample_paths(p_firefly, per_class_target, rng),
        "pyralis_background":   sample_paths(p_bg,      per_class_target, rng),
        "frontalis_firefly":    sample_paths(f_firefly, per_class_target, rng),
        "frontalis_background": sample_paths(f_bg,      per_class_target, rng),
    }

    # Prepare output dirs
    out_dirs = {k: dest_dir / k for k in chosen}
    for d in out_dirs.values():
        ensure_dir(d)

    # Copy/link
    copied = {k: 0 for k in chosen}
    for key, paths in chosen.items():
        d = out_dirs[key]
        for src in tqdm(paths, desc=f"{key}", unit="img", ncols=80):
            dst = unique_dest(d, src)
            copy_or_link(src, dst, link)
            copied[key] += 1

    return {
        "per_class_target": per_class_target,
        "pyralis": {"firefly": per_class_target, "background": per_class_target, "total": per_class_target * 2},
        "frontalis": {"firefly": per_class_target, "background": per_class_target, "total": per_class_target * 2},
        "dest": str(dest_dir.resolve()),
        "link": link,
    }


def maybe_zip(dest: Path) -> Path:
    zip_base = dest.resolve()
    archive = shutil.make_archive(str(zip_base), 'zip', root_dir=str(dest), base_dir='.')
    return Path(archive)


def main() -> None:
    stats = build_dataset(PYRALIS_DIR, FRONTALIS_DIR, DEST_DIR,
                          LINK, SEED)

    print("\n=== build summary ===")
    print(f"dest: {stats['dest']}")
    print(f"per_class_target: {stats['per_class_target']}")
    print(f"pyralis    | firefly {stats['pyralis']['firefly']:6d} | background {stats['pyralis']['background']:6d} | total {stats['pyralis']['total']:6d}")
    print(f"frontalis  | firefly {stats['frontalis']['firefly']:6d} | background {stats['frontalis']['background']:6d} | total {stats['frontalis']['total']:6d}")

    if ZIP_OUTPUT:
        archive = maybe_zip(DEST_DIR)
        print(f"\n✅ Zipped dataset → {archive}")
    else:
        print("\n✅ Dataset folders created (set ZIP_OUTPUT=True to archive)")


if __name__ == '__main__':
    main()
