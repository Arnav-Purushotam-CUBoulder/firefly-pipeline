#!/usr/bin/env python3
"""
Export assets for a selected set of GLOBAL component IDs.

Inputs:
  • COMPONENT_IDS: list of global component ids to export
  • COMPONENT_GIFS_DIR: folder created by the main pipeline's Stage 5
      └── component_gifs/
            00001/t_000123.png
            00001/t_000130.png
            00002/...
  • CC_CROPS_DIR: folder created by the pipeline when it saved per-chunk
    connected-component crops (full streak shapes), files like:
      gid_067847_x2439_y287_w26_h14_t9001.png
      comp_000123_x0123_y0456_w0030_h0012_chunk_000000_000499.png

Outputs:
  • OUTPUT_FLASH_PATCHES_DIR: all 10×10 patches (flashes) for the selected ids,
    flattened into a single folder
  • OUTPUT_STREAK_CROPS_DIR: all streak/shape crops for the selected ids,
    flattened into a single folder
  • SELECTED_COMPONENTS_CSV_PATH: CSV listing the selected component ids
      (column: global_component_id)

Edit COMPONENT_IDS and the paths below as needed.
"""

from __future__ import annotations
from pathlib import Path
import re
import shutil
import sys

# ───────────── GLOBAL PARAMETERS (EDIT THESE) ─────────────

# List of GLOBAL component ids you want to export
COMPONENT_IDS = [
    # e.g., 12, 47, 131
    4925, 4935, 4893, 4894, 4956,4936,4916,4933,4914,4905,4981,4872,4919,
]

# Base directory where your previous pipeline wrote outputs
BASE_DIR = Path("/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/pinfield videos")

# Where Stage 5 saved per-component 10×10 patches (one subfolder per component id)
COMPONENT_GIFS_DIR = BASE_DIR / "component_gifs"

# Where Stage 4 saved per-component streak crops (your earlier script’s “cc_crops”)
# This folder contains per-chunk subfolders with many PNGs named like:
#   gid_067847_x2439_y287_w26_h14_t9001.png
CC_CROPS_DIR = BASE_DIR / "cc_crops"

# Output folders (flat)
OUTPUT_FLASH_PATCHES_DIR = BASE_DIR / "exported_flash_patches"
OUTPUT_STREAK_CROPS_DIR  = BASE_DIR / "exported_streak_crops"

# CSV of the selected component “names” (ids)
SELECTED_COMPONENTS_CSV_PATH = BASE_DIR / "selected_components_list.csv"

# If a destination file exists, choose whether to overwrite it
OVERWRITE = True

# ───────────── HELPERS ─────────────

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _copy_file(src: Path, dst_dir: Path, new_name: str | None = None) -> None:
    _ensure_dir(dst_dir)
    dst = dst_dir / (new_name if new_name else src.name)
    if dst.exists():
        if OVERWRITE:
            dst.unlink()
        else:
            return
    shutil.copy2(str(src), str(dst))

def _format_cid(cid: int) -> str:
    # Match your pipeline’s zero-padding style; adjust if you used 6 digits elsewhere
    return f"{int(cid):05d}"

# ───────────── EXPORT LOGIC ─────────────

def export_flash_patches_for_components(component_ids: list[int]) -> int:
    """
    Copy all 10×10 patches (flash frames) for the given component ids
    from COMPONENT_GIFS_DIR/<cid_padded>/*.png into OUTPUT_FLASH_PATCHES_DIR.
    """
    copied = 0
    for cid in component_ids:
        cid_pad = _format_cid(cid)
        src_dir = COMPONENT_GIFS_DIR / cid_pad
        if not src_dir.exists():
            print(f"⚠️  Missing flash/patch folder for component {cid} → {src_dir}")
            continue

        for src in sorted(src_dir.glob("*.png")):
            # Preserve time in the filename to keep ordering
            new_name = f"comp_{cid_pad}_{src.name}"
            _copy_file(src, OUTPUT_FLASH_PATCHES_DIR, new_name)
            copied += 1
    return copied

def export_streak_crops_for_components(component_ids: list[int]) -> int:
    """
    Copy all connected-component streak crops whose filename encodes the global id.
    Searches recursively under CC_CROPS_DIR and matches both of these forms:
      • gid_<id>_x..._y..._w..._h..._t....png
      • comp_<id>_x..._y..._w..._h..._chunk_....png
    """
    if not CC_CROPS_DIR.exists():
        print(f"⚠️  CC_CROPS_DIR does not exist: {CC_CROPS_DIR}")
        return 0

    # Accept either "gid_<digits>_" or "comp_<digits>_"
    re_comp = re.compile(r"(?:^|_)(?:gid|comp)_(\d+)(?:_|\. )", re.IGNORECASE)

    wanted = set(int(c) for c in component_ids)
    copied = 0

    for src in sorted(CC_CROPS_DIR.rglob("*.png")):
        m = re_comp.search(src.name)
        if not m:
            continue
        file_cid = int(m.group(1))
        if file_cid in wanted:
            _copy_file(src, OUTPUT_STREAK_CROPS_DIR)
            copied += 1

    if copied == 0:
        # Small hint if nothing matched
        print("⚠️  No streak crops matched the requested component ids.")
        print("    Example expected filenames: gid_000123_*.png  or  comp_000123_*.png")
    return copied

def write_selected_components_csv(component_ids: list[int], csv_path: Path) -> None:
    """
    Write a simple CSV with the selected component ids.
    Columns: global_component_id
    """
    _ensure_dir(csv_path.parent)
    with open(csv_path, "w") as f:
        f.write("global_component_id\n")
        for cid in component_ids:
            f.write(f"{int(cid)}\n")

# ───────────── MAIN ─────────────

def main():
    if not COMPONENT_IDS:
        print("No COMPONENT_IDS provided. Edit the list at the top of this file.")
        sys.exit(1)

    print(f"Components to export ({len(COMPONENT_IDS)}): {COMPONENT_IDS}")

    # 1) Copy 10×10 flash patches
    n_patches = export_flash_patches_for_components(COMPONENT_IDS)
    print(f"✅ Copied flash patches: {n_patches} → {OUTPUT_FLASH_PATCHES_DIR}")

    # 2) Copy streak/shape crops
    n_crops = export_streak_crops_for_components(COMPONENT_IDS)
    print(f"✅ Copied streak crops:   {n_crops} → {OUTPUT_STREAK_CROPS_DIR}")

    # 3) Write CSV list of selected components
    write_selected_components_csv(COMPONENT_IDS, SELECTED_COMPONENTS_CSV_PATH)
    print(f"✅ Wrote component list CSV → {SELECTED_COMPONENTS_CSV_PATH}")

if __name__ == "__main__":
    main()
