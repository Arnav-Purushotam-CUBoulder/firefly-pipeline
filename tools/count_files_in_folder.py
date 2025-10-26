#!/usr/bin/env python3
from pathlib import Path


# Global configuration — set these
FOLDER_PATH = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/long exposure shots/temp testing dataset collection/10x10 patches' # Set this to the folder you want to count
RECURSIVE = False                 # True to include all subfolders
INCLUDE_HIDDEN = False            # True to include dotfiles
ONLY_NUMBER = False               # True to print just the number


def count_files(directory: Path, recursive: bool = False, include_hidden: bool = False) -> int:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    iterator = directory.rglob("*") if recursive else directory.iterdir()
    count = 0
    for p in iterator:
        try:
            if p.is_file():
                if not include_hidden and p.name.startswith("."):
                    continue
                count += 1
        except (PermissionError, OSError):
            # Skip unreadable paths
            continue
    return count


def main() -> None:
    directory = Path(FOLDER_PATH).expanduser()
    total = count_files(directory, recursive=RECURSIVE, include_hidden=INCLUDE_HIDDEN)

    if ONLY_NUMBER:
        print(total)
    else:
        mode = "recursive" if RECURSIVE else "top-level"
        hidden = ", including hidden" if INCLUDE_HIDDEN else ""
        print(f"Counted {total} files in '{directory.resolve()}' ({mode}{hidden}).")


if __name__ == "__main__":
    main()
