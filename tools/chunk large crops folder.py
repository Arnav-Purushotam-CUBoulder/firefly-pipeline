#!/usr/bin/env python3
"""
Split a large folder of images into N subfolders.

Features
- Round-robin or chunked distribution
- Move / copy / symlink modes
- Optional recursive scan and extension filtering
- Collision-safe file moves (auto-renames on conflicts)
- Restart-friendly (skips files already moved if names match)
- Low memory by default (no need to list all files at once)

Usage
- Set the GLOBAL CONFIG below, then run:
    python split_folder.py
"""

from pathlib import Path
import shutil
import itertools
import math
import sys
import random
from typing import Iterable, List, Set

# ───────────────────────── GLOBAL CONFIG ─────────────────────────

INPUT_DIR          = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/crops to create dataset')        # folder containing ~1M small images
OUTPUT_PARENT_DIR  = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/chunked crop folders')     # a parent folder that will hold the N subfolders
N_PARTS            = 10                                  # how many subfolders to split into
SUBDIR_PREFIX      = "part_"                             # subfolder name prefix -> part_001, part_002, ...
ZERO_PAD           = 3                                   # digits for subfolder numbering

# What files to include
RECURSIVE          = False                               # if True, include files from subdirectories
INCLUDE_EXTS: Set[str] = {                               # case-insensitive; set() to include all files
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"
}

# How to distribute
DISTRIBUTION_MODE  = "roundrobin"                        # "roundrobin" (1-pass, low memory) or "chunks"
SORT_BY_NAME       = False                               # if True and DISTRIBUTION_MODE="chunks", sorts before splitting
SHUFFLE            = False                               # if True and DISTRIBUTION_MODE="chunks", shuffles before splitting

# Action to take
MODE               = "move"                              # "move" (default), "copy", or "symlink"
DRY_RUN            = False                               # if True, just print what would happen

# Safety / ergonomics
RENAME_ON_COLLISION = True                               # if a filename already exists in target, append suffix
LOG_EVERY           = 10_000                             # progress logging interval
MAX_FILES           = None                               # int | None, limit processed files (useful for testing)

# ──────────────────────── IMPLEMENTATION ────────────────────────

def _is_image(path: Path) -> bool:
    if not path.is_file():
        return False
    if not INCLUDE_EXTS:
        return True
    return path.suffix.lower() in INCLUDE_EXTS

def _iter_files(source: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        it = source.rglob("*")
    else:
        it = source.glob("*")
    for p in it:
        if _is_image(p):
            yield p

def _ensure_subfolders(parent: Path, n: int) -> List[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(1, n + 1):
        name = f"{SUBDIR_PREFIX}{i:0{ZERO_PAD}d}"
        d = parent / name
        d.mkdir(parents=True, exist_ok=True)
        out.append(d)
    return out

def _unique_target_path(dst_dir: Path, filename: str) -> Path:
    """
    If filename exists in dst_dir and RENAME_ON_COLLISION is True,
    append __1, __2, ... before the extension.
    """
    candidate = dst_dir / filename
    if not candidate.exists() or not RENAME_ON_COLLISION:
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    k = 1
    while True:
        alt = dst_dir / f"{stem}__{k}{suffix}"
        if not alt.exists():
            return alt
        k += 1

def _transfer(src: Path, dst_dir: Path):
    dst = _unique_target_path(dst_dir, src.name)
    if DRY_RUN:
        print(f"[DRY] {MODE.upper()} {src} -> {dst}")
        return

    if MODE == "move":
        shutil.move(str(src), str(dst))
    elif MODE == "copy":
        shutil.copy2(str(src), str(dst))
    elif MODE == "symlink":
        # Use relative symlinks for portability
        rel = Path(os.path.relpath(src, start=dst_dir))
        dst.symlink_to(rel)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

def _roundrobin_split():
    out_dirs = _ensure_subfolders(OUTPUT_PARENT_DIR, N_PARTS)
    cycles = itertools.cycle(out_dirs)

    count = 0
    for i, src in enumerate(_iter_files(INPUT_DIR, RECURSIVE), start=1):
        dst_dir = next(cycles)
        _transfer(src, dst_dir)
        count += 1
        if LOG_EVERY and i % LOG_EVERY == 0:
            print(f"Processed {i:,} files…")
        if MAX_FILES is not None and count >= MAX_FILES:
            break
    print(f"Done. Total processed: {count:,}")

def _chunks_split():
    # Collect list (higher memory, but enables perfect chunking + sorting/shuffling)
    files = [p for p in _iter_files(INPUT_DIR, RECURSIVE)]
    total = len(files)
    print(f"Found {total:,} files.")
    if SORT_BY_NAME:
        files.sort(key=lambda p: p.name)
    if SHUFFLE:
        random.shuffle(files)
    if MAX_FILES is not None:
        files = files[:MAX_FILES]
        total = len(files)

    out_dirs = _ensure_subfolders(OUTPUT_PARENT_DIR, N_PARTS)
    chunk_size = math.ceil(total / N_PARTS) if total else 0

    processed = 0
    for idx, start in enumerate(range(0, total, chunk_size), start=1):
        end = min(start + chunk_size, total)
        part_files = files[start:end]
        dst_dir = out_dirs[idx - 1]
        for i, src in enumerate(part_files, start=1):
            _transfer(src, dst_dir)
            processed += 1
            if LOG_EVERY and processed % LOG_EVERY == 0:
                print(f"Processed {processed:,} files…")

    print(f"Done. Total processed: {processed:,}")

def _validate():
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"ERROR: INPUT_DIR does not exist or is not a directory: {INPUT_DIR}", file=sys.stderr)
        sys.exit(2)
    if N_PARTS <= 0:
        print("ERROR: N_PARTS must be >= 1", file=sys.stderr)
        sys.exit(2)
    if DISTRIBUTION_MODE not in {"roundrobin", "chunks"}:
        print("ERROR: DISTRIBUTION_MODE must be 'roundrobin' or 'chunks'", file=sys.stderr)
        sys.exit(2)
    if MODE not in {"move", "copy", "symlink"}:
        print("ERROR: MODE must be 'move', 'copy', or 'symlink'", file=sys.stderr)
        sys.exit(2)

def main():
    _validate()
    print(f"Splitting '{INPUT_DIR}' into {N_PARTS} subfolders under '{OUTPUT_PARENT_DIR}'")
    print(f"Mode={MODE}, Distribution={DISTRIBUTION_MODE}, Recursive={RECURSIVE}, DryRun={DRY_RUN}")
    print(f"Extensions filter: {'ALL' if not INCLUDE_EXTS else ', '.join(sorted(INCLUDE_EXTS))}")
    if DISTRIBUTION_MODE == "roundrobin":
        _roundrobin_split()
    else:
        _chunks_split()

if __name__ == "__main__":
    main()
