#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple

# Global variables: set these to your directories
INPUT_DIR = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/output data/20240606_cam1_GS010064/raw 10x10 crops from sbd'
OUTPUT_DIR = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset/v2/initial dataset/background'

# Behavior toggles
RECURSIVE = True                 # Move files in all subfolders
PRESERVE_SUBDIRS = False         # Recreate subfolders under INPUT_DIR
ON_CONFLICT = "rename"           # 'overwrite' | 'skip' | 'rename'
DRY_RUN = False                  # True to preview without moving


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def iter_files(src: Path) -> Iterable[Path]:
    if RECURSIVE:
        yield from (p for p in src.rglob("*") if p.is_file())
    else:
        yield from (p for p in src.iterdir() if p.is_file())


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def move_one(src_file: Path, dst_dir: Path) -> Optional[Path]:
    dst_file = dst_dir / src_file.name
    if dst_file.exists():
        if ON_CONFLICT == "skip":
            return None
        elif ON_CONFLICT == "rename":
            dst_file = unique_target(dst_file)
        elif ON_CONFLICT == "overwrite":
            dst_file.unlink()
        else:
            raise ValueError(f"Invalid ON_CONFLICT: {ON_CONFLICT}")

    if DRY_RUN:
        return dst_file

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_file), str(dst_file))
    return dst_file


def move_all(src_dir: Path, dst_dir: Path) -> Tuple[int, int]:
    if not src_dir.exists() or not src_dir.is_dir():
        raise ValueError(f"Source directory does not exist or is not a directory: {src_dir}")
    if is_within(dst_dir, src_dir):
        raise ValueError(f"Destination {dst_dir} is inside source {src_dir}; aborting.")

    moved = 0
    skipped = 0
    for f in iter_files(src_dir):
        rel_dir = f.parent.relative_to(src_dir) if PRESERVE_SUBDIRS else Path()
        target_dir = dst_dir / rel_dir
        res = move_one(f, target_dir)
        if res is None:
            skipped += 1
        else:
            moved += 1

    if RECURSIVE and not DRY_RUN:
        # Clean up empty dirs left in source
        for root, dirs, files in os.walk(src_dir, topdown=False):
            if not dirs and not files:
                try:
                    Path(root).rmdir()
                except OSError:
                    pass

    return moved, skipped


def main():
    # Move from INPUT_DIR to OUTPUT_DIR
    src = Path(INPUT_DIR).expanduser()
    dst = Path(OUTPUT_DIR).expanduser()
    
    def count_files(d: Path) -> int:
        if not d.exists() or not d.is_dir():
            return 0
        if RECURSIVE:
            return sum(1 for p in d.rglob('*') if p.is_file())
        return sum(1 for p in d.iterdir() if p.is_file())

    src_before = count_files(src)
    dst_before = count_files(dst)
    print(f"Initial counts — src: {src_before}, dst: {dst_before}")

    moved, skipped = move_all(src, dst)
    msg = f"Moved {moved} files from '{src.resolve()}' to '{dst.resolve()}'"
    if skipped:
        msg += f" (skipped {skipped} on conflict: {ON_CONFLICT})"
    if DRY_RUN:
        msg += " [DRY RUN]"
    print(msg)

    # Final counts (actual after move when not DRY_RUN)
    src_after = count_files(src)
    dst_after = count_files(dst)
    print(f"Final counts — src: {src_after}, dst: {dst_after}")


if __name__ == "__main__":
    main()
