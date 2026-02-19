#!/usr/bin/env python3
"""
DEPRECATED LOCATION (shim)
-------------------------
The species scaler script was moved under:
  integrated ingestor-trainer-tester orchestrator/stage1_ingestor_core.py

The old filename `integrated ingestor-trainer-tester orchestrator/species_scaler.py`
is kept as a shim for compatibility.

This file remains as a compatibility shim so any old commands still work.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _moved_path() -> Path:
    # repo_root/<this_dir>/species_scaler.py -> repo_root/integrated .../species_scaler.py
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "integrated ingestor-trainer-tester orchestrator" / "species_scaler.py"


def main() -> None:
    moved = _moved_path()
    if not moved.exists():
        raise SystemExit(f"Moved species scaler not found: {moved}")

    # Preserve CLI args; run the moved script as __main__.
    sys.argv[0] = str(moved)
    runpy.run_path(str(moved), run_name="__main__")


if __name__ == "__main__":
    main()
