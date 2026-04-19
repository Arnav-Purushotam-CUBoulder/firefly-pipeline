#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import params


def cleanup_root(*, verbose: bool = True) -> None:
    """Delete every directory under ROOT except \"original videos\" and \"ground truth\".

    Only removes directories; leaves files alone.
    """
    root: Path = params.ROOT
    keep = {"original videos", "ground truth"}
    if verbose:
        print(f"[stage0_cleanup] Cleaning root: {root}")
    root.mkdir(parents=True, exist_ok=True)

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in keep:
            continue
        try:
            shutil.rmtree(child)
            if verbose:
                print(f"[stage0_cleanup] Removed: {child}")
        except Exception as exc:
            if verbose:
                print(f"[stage0_cleanup] WARNING: failed to remove {child}: {exc}")


__all__ = ["cleanup_root"]
