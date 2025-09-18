#!/usr/bin/env python3
"""
Export assets for a selected set of GLOBAL component IDs.

Inputs:
  • COMPONENT_IDS: list of global component ids to export
  • EXCLUDE_COMPONENT_IDS: list of global component ids to skip (exclusions win)
  • (Optional) COMPONENT_IDS_TEXT: paste-friendly text with ids like "03499, 04331, 04744"
  • (Optional) EXCLUDE_COMPONENT_IDS_TEXT: paste-friendly text for exclusions
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

Edit COMPONENT_IDS (or the *_TEXT variants) and the paths below as needed.
"""

from __future__ import annotations
from pathlib import Path
import re
import shutil
import sys

# ───────────── GLOBAL PARAMETERS (EDIT THESE) ─────────────

# List of GLOBAL component ids you want to export
# (You can leave this empty and use COMPONENT_IDS_TEXT below if your notes have leading zeros)
COMPONENT_IDS: list[str] = [
    # e.g., "12", "47", "131" (strings allow leading zeros like "00012")
    "4925", "4935", "4893", "4894", "4956", "4936", "4916", "4933", "4914", "4905", "4981", "4872", "4919",
    "4842", "4828", "4875", "4845", "4744", "4760", "4873", "4823", "4893", "4925", "4894", "4895", "4935", "4936", "4956", "4904", "4916", "4884", "4886", "4914", "4918", "4876", "4891", "4871", "4923", "4930", "4933", "4934", "4905", "4887", "4908", "4331", "4872", "4919", "4833", "7467", "9201", "9158", "9113", "9209", "9110", "9125", "9118", "9198", "9061", "9171", "9137", "9194", "9208", "9166", "9134", "9199", "9182", "9196", "9186", "9206", "9233", "9213", "9222", "9139", "9136", "8997", "9220", "9143", "8978", "9157", "8988", "9003", "9096", "9027", "13406", "13388", "13475", "13452", "13499", "13475", "13452", "3499", "13513", "13514", "13520", "13535", "13532", "13528", "13498", "13542", "13538", "13311", "13061", "13554", "13549", "13541", "13462", "13382", "11145", "10648", "10452", "13382", "13462", "13444", "13472", "13541", "13447", "13549", "17593", "17615", "17677", "17676", "17575", "17407", "17624", "17019", "17687", "17469", "17689", "17692", "17621", "17613", "17618", "17672", "17670", "17659", "17653", "17664", "17657", "17639", "17638", "17656", "17671", "17645", "17629", "17635", "17576", "17577", "17567", "17541", "17514", "21632", "21611", "21782", "21660", "21683", "21762", "21754", "21752", "21735", "21748", "21760", "21736", "21152", "21760", "21735", "21776", "21753", "21667", "21674", "21765", "21757", "21745", "25535", "25634", "25636", "25623", "25648", "25651", "25646", "25662", "25671", "25681", "25664", "25676", "25697", "25700", "25706", "25711", "25707", "25714", "25620",
]

# List of GLOBAL component ids you want to EXCLUDE from export
EXCLUDE_COMPONENT_IDS: list[str] = [
    '03499', '04331', '04744', '04760', '04764', '04823', '04981', '07467', '08978', '09061', '10452', '11145', '10648' 
]

# OPTIONAL: Paste-friendly text blocks that allow leading zeros exactly as in your notes.
# Examples:
# COMPONENT_IDS_TEXT = "03499, 04331, 04744\n04760, 04764, 04823"
# EXCLUDE_COMPONENT_IDS_TEXT = "00012, 4935"
COMPONENT_IDS_TEXT = ""
EXCLUDE_COMPONENT_IDS_TEXT = ""

# Background component selection (separate lists for background patches)
BG_COMPONENT_IDS: list[str] = []
BG_EXCLUDE_COMPONENT_IDS: list[str] = []
BG_COMPONENT_IDS_TEXT = ""
BG_EXCLUDE_COMPONENT_IDS_TEXT = ""

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

# Background output folders (flat)
OUTPUT_BG_FLASH_PATCHES_DIR = BASE_DIR / "exported_flash_patches_background"
OUTPUT_BG_STREAK_CROPS_DIR  = BASE_DIR / "exported_streak_crops_background"

# CSV of the selected component “names” (ids)
SELECTED_COMPONENTS_CSV_PATH = BASE_DIR / "selected_components_list.csv"
SELECTED_BACKGROUND_COMPONENTS_CSV_PATH = BASE_DIR / "selected_background_components_list.csv"

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

def _purge_dir(p: Path) -> None:
    """Delete all contents of a directory (if it exists), then recreate it.
    Ensures a clean slate on every run.
    """
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        # Best-effort cleanup; proceed to recreate
        pass
    p.mkdir(parents=True, exist_ok=True)

def _format_cid(cid: str | int) -> str:
    """Format a component id as zero-padded 5-digit string.
    Accepts string inputs with or without leading zeros or an int.
    """
    cid_int = int(str(cid).strip())
    return f"{cid_int:05d}"

def _dedupe_preserve_order(ids: list[int]) -> list[int]:
    """Remove duplicates while preserving the original order."""
    seen: set[int] = set()
    out: list[int] = []
    for x in ids:
        xi = int(x)
        if xi not in seen:
            seen.add(xi)
            out.append(xi)
    return out

def _parse_ids_from_text(text: str) -> list[int]:
    """
    Parse a free-form string that may contain ids with leading zeros.
    Accepts separators like commas, spaces, newlines. Returns ints.
    """
    if not text:
        return []
    # Grab digit sequences; this tolerates commas/newlines/spaces/etc.
    tokens = re.findall(r"\d+", text)
    # Convert to int while preserving '0' correctly if it appears
    return [int(t.lstrip("0") or "0") for t in tokens]

def _resolve_ids(list_input: list[int] | list[str], text_input: str) -> list[int]:
    """
    Choose between the normal Python list input and the paste-friendly text block.
    If text_input is non-empty, it wins. Otherwise, coerce list_input items to int.
    """
    if text_input and text_input.strip():
        return _parse_ids_from_text(text_input)
    # Coerce any values (ints/strings) to int
    out: list[int] = []
    for x in list_input:
        out.append(int(str(x)))
    return out

# ───────────── EXPORT LOGIC ─────────────

def export_flash_patches_for_components(component_ids: list[int], dst_dir: Path) -> int:
    """
    Copy all 10×10 patches (flash frames) for the given component ids
    from COMPONENT_GIFS_DIR/<cid_padded>/*.png into dst_dir.
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
            _copy_file(src, dst_dir, new_name)
            copied += 1
    return copied

def export_streak_crops_for_components(component_ids: list[int], dst_dir: Path) -> int:
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
            _copy_file(src, dst_dir)
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
    # Always start with clean output folders for a fresh export
    _purge_dir(OUTPUT_FLASH_PATCHES_DIR)
    _purge_dir(OUTPUT_STREAK_CROPS_DIR)
    _purge_dir(OUTPUT_BG_FLASH_PATCHES_DIR)
    _purge_dir(OUTPUT_BG_STREAK_CROPS_DIR)

    # Resolve inputs (the *_TEXT variants allow pasting ids with leading zeros)
    include_ids = _resolve_ids(COMPONENT_IDS, COMPONENT_IDS_TEXT)
    exclude_ids = _resolve_ids(EXCLUDE_COMPONENT_IDS, EXCLUDE_COMPONENT_IDS_TEXT)
    bg_include_ids = _resolve_ids(BG_COMPONENT_IDS, BG_COMPONENT_IDS_TEXT)
    bg_exclude_ids = _resolve_ids(BG_EXCLUDE_COMPONENT_IDS, BG_EXCLUDE_COMPONENT_IDS_TEXT)

    if not include_ids:
        print("No COMPONENT_IDS provided. Edit the list at the top or use COMPONENT_IDS_TEXT.")
        sys.exit(1)

    # 1) De-duplicate while preserving order
    unique_ids = _dedupe_preserve_order(include_ids)

    # 2) Apply exclusions (exclusions win)
    if exclude_ids:
        exclude_set = {int(x) for x in exclude_ids}
        unique_ids = [cid for cid in unique_ids if cid not in exclude_set]

    print(f"Firefly components to export ({len(unique_ids)}): {unique_ids}")

    # 3) Copy 10×10 flash patches
    n_patches = export_flash_patches_for_components(unique_ids, OUTPUT_FLASH_PATCHES_DIR)
    print(f"✅ Copied flash patches: {n_patches} → {OUTPUT_FLASH_PATCHES_DIR}")

    # 4) Copy streak/shape crops
    n_crops = export_streak_crops_for_components(unique_ids, OUTPUT_STREAK_CROPS_DIR)
    print(f"✅ Copied streak crops:   {n_crops} → {OUTPUT_STREAK_CROPS_DIR}")

    # 5) Write CSV list of selected components
    write_selected_components_csv(unique_ids, SELECTED_COMPONENTS_CSV_PATH)
    print(f"✅ Wrote component list CSV → {SELECTED_COMPONENTS_CSV_PATH}")

    # Background set (optional)
    if bg_include_ids:
        bg_unique = _dedupe_preserve_order(bg_include_ids)
        if bg_exclude_ids:
            bg_ex_set = {int(x) for x in bg_exclude_ids}
            bg_unique = [cid for cid in bg_unique if cid not in bg_ex_set]

        print(f"Background components to export ({len(bg_unique)}): {bg_unique}")
        n_bg_patches = export_flash_patches_for_components(bg_unique, OUTPUT_BG_FLASH_PATCHES_DIR)
        print(f"✅ Copied BG flash patches: {n_bg_patches} → {OUTPUT_BG_FLASH_PATCHES_DIR}")
        n_bg_crops = export_streak_crops_for_components(bg_unique, OUTPUT_BG_STREAK_CROPS_DIR)
        print(f"✅ Copied BG streak crops:  {n_bg_crops} → {OUTPUT_BG_STREAK_CROPS_DIR}")
        write_selected_components_csv(bg_unique, SELECTED_BACKGROUND_COMPONENTS_CSV_PATH)
        print(f"✅ Wrote BG component list CSV → {SELECTED_BACKGROUND_COMPONENTS_CSV_PATH}")
    else:
        print("No background component ids provided; skipping BG export.")

if __name__ == "__main__":
    main()
