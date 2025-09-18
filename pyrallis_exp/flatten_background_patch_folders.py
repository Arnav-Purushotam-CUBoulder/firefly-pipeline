#!/usr/bin/env python3
"""
Flatten 1‑level background patch folders into a single output folder.

Given an input parent directory containing many one‑level subfolders (e.g.,
  parent/
    00001/t_000100.png
    00002/t_000182.png
    ...
this script copies (or moves) all image patches from each subfolder directly
into a single OUTPUT_DIR, renaming to avoid collisions. By default it prefixes
filenames with the subfolder name as:
  bg_<subfolder>_<original_name>

Configure the GLOBAL SETTINGS below and run the script.
"""

from __future__ import annotations
from pathlib import Path
import shutil
import sys

# ───────────── GLOBAL SETTINGS (EDIT THESE) ─────────────

# Folder that contains many one‑level subfolders with PNG patches
INPUT_PARENT_DIR = Path(
    '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/pinfield videos/component_gifs'  # e.g., BASE_DIR / "component_gifs"
)

# Where to write the flattened patches
OUTPUT_DIR = Path(
    '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/pinfield videos/background patches/initial'
)

# Copy vs move patches into OUTPUT_DIR
MODE = "copy"   # "copy" | "move"

# Start with a clean output folder every run
CLEAN_OUTPUT_FIRST = True

# Prefix for output names to avoid collisions. Examples: "bg", "comp", "" (empty for none)
FILENAME_PREFIX = "bg"

# Only include these file extensions (case‑insensitive)
INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# If a file with the same name exists, append __1, __2, ...
RENAME_ON_COLLISION = True

# Log every N copied files (0 to disable)
LOG_EVERY = 1000

# Only traverse one level of subfolders (the default). Set True to recurse.
RECURSIVE = False


# ───────────── IMPLEMENTATION ─────────────

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _purge_dir(p: Path) -> None:
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass
    p.mkdir(parents=True, exist_ok=True)

def _is_included(path: Path) -> bool:
    return path.is_file() and (path.suffix.lower() in INCLUDE_EXTS)

def _unique_target(dst_dir: Path, filename: str) -> Path:
    if not RENAME_ON_COLLISION:
        return dst_dir / filename
    stem = Path(filename).stem
    suf = Path(filename).suffix
    k = 1
    candidate = dst_dir / filename
    while candidate.exists():
        candidate = dst_dir / f"{stem}__{k}{suf}"
        k += 1
    return candidate

def _out_name(subdir: str, src_name: str) -> str:
    if FILENAME_PREFIX:
        return f"{FILENAME_PREFIX}_{subdir}_{src_name}"
    return f"{subdir}_{src_name}"

def _iter_sources(parent: Path):
    if RECURSIVE:
        yield from (p for p in parent.rglob("*") if p.is_dir())
    else:
        yield from (p for p in parent.iterdir() if p.is_dir())

def main() -> None:
    if not INPUT_PARENT_DIR.exists() or not INPUT_PARENT_DIR.is_dir():
        print(f"ERROR: INPUT_PARENT_DIR does not exist or is not a directory: {INPUT_PARENT_DIR}")
        sys.exit(2)

    if CLEAN_OUTPUT_FIRST:
        _purge_dir(OUTPUT_DIR)
    else:
        _ensure_dir(OUTPUT_DIR)

    copied = 0
    subdirs = 0

    for sub in _iter_sources(INPUT_PARENT_DIR):
        subdirs += 1
        for src in sorted(sub.iterdir()):
            if not _is_included(src):
                continue
            out_name = _out_name(sub.name, src.name)
            dst = _unique_target(OUTPUT_DIR, out_name)
            if MODE == "move":
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            copied += 1
            if LOG_EVERY and (copied % LOG_EVERY == 0):
                print(f"Processed {copied} files…")

    print(f"Done. Subfolders scanned: {subdirs}. Files written: {copied}. → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

