#!/usr/bin/env python3
"""
Combine species-level detection & ground-truth CSVs into two unified CSVs.

Input naming convention (any count per species is fine):
  <SPECIES>_<true|resnet>.{csv|tsv|txt}
  <SPECIES>_<true|resnet>_<anything>.{csv|tsv|txt}
Examples:
  frontalis_true.csv
  forresti_resnet_part1.csv
  tremulans_true_segmentA.tsv
  tremulans_resnet_run03.csv

Expected schemas
  • Detections ("resnet"): columns include at least
        t, x, y, background_logit, firefly_logit
  • Ground truth ("true"): columns include at least
        x, y, t
Notes:
  - Delimiter is auto-detected on read (comma or tab).
  - Outputs include a 'species' column so you can filter back per species.
  - Outputs are sorted by (t, species, y, x).

Configure paths & parameters in the GLOBAL CONFIG section below.
"""

# ────────────────────────── GLOBAL CONFIG ──────────────────────────
INPUT_DIR           = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/fixing stage9 val/hybrid'   # folder containing all 6 inputs (and possibly more)
SPECIES             = ["forresti", "frontalis", "tremulans"]

# Output files (single combined files):
OUTPUT_DETECTIONS   = "/Users/arnavps/Desktop/New DL project data to transfer to external disk/fixing stage9 val/hybrid/all_resnet_combined.csv"
OUTPUT_GROUND_TRUTH = "/Users/arnavps/Desktop/New DL project data to transfer to external disk/fixing stage9 val/hybrid/all_true_combined.csv"

# Write settings:
OUTPUT_DELIMITER    = ","   # "\t" for TSV, or "," for CSV
INDEX_IN_OUTPUT     = False  # set True to keep pandas index column

# Optional cleaning knobs:
DROP_DUPES_DET      = False  # drop duplicate rows in detections
DROP_DUPES_GT       = False  # drop duplicate rows in ground truth
STRICT_COLUMNS      = False  # if True, raise if required columns missing; if False, try to coerce/rename

# ───────────────────────── IMPLEMENTATION ──────────────────────────
from pathlib import Path
import re
import sys
import pandas as pd

def _debug(msg: str):
    print(f"[combine] {msg}")

def _read_any_csv(path: Path) -> pd.DataFrame:
    """
    Read CSV/TSV with automatic delimiter inference.
    Uses pandas engine='python' with sep=None to sniff delimiters.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase and strip whitespace from column names for consistent matching.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _normalize_det_df(df: pd.DataFrame, src: Path) -> pd.DataFrame:
    """
    Ensure detection dataframe contains: t, x, y, background_logit, firefly_logit
    Try common aliases if STRICT_COLUMNS is False.
    Cast types to sensible dtypes.
    """
    df = _standardize_columns(df)

    # Aliases for time column
    time_aliases = ["t", "time", "frame", "frame_num", "frameindex", "frame_idx"]
    x_aliases    = ["x", "cx", "xcenter", "x_center"]
    y_aliases    = ["y", "cy", "ycenter", "y_center"]

    def _pick(colnames, aliases):
        for a in aliases:
            if a in colnames:
                return a
        return None

    cols = set(df.columns)

    t_col = _pick(cols, ["t"]) or (None if STRICT_COLUMNS else _pick(cols, time_aliases))
    x_col = _pick(cols, ["x"]) or (None if STRICT_COLUMNS else _pick(cols, x_aliases))
    y_col = _pick(cols, ["y"]) or (None if STRICT_COLUMNS else _pick(cols, y_aliases))

    bg_col = "background_logit" if "background_logit" in cols else None
    ff_col = "firefly_logit"    if "firefly_logit"    in cols else None

    missing = [name for name, val in {
        "t": t_col, "x": x_col, "y": y_col,
        "background_logit": bg_col, "firefly_logit": ff_col
    }.items() if val is None]

    if missing:
        raise ValueError(
            f"{src.name}: required columns missing or not recognized: {missing}. "
            f"Have columns={sorted(df.columns)}"
        )

    out = df[[t_col, x_col, y_col, bg_col, ff_col]].rename(columns={
        t_col: "t", x_col: "x", y_col: "y", bg_col: "background_logit", ff_col: "firefly_logit"
    }).copy()

    # Cast dtypes
    out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("Int64")
    out["x"] = pd.to_numeric(out["x"], errors="coerce").astype("Int64")
    out["y"] = pd.to_numeric(out["y"], errors="coerce").astype("Int64")
    out["background_logit"] = pd.to_numeric(out["background_logit"], errors="coerce")
    out["firefly_logit"]    = pd.to_numeric(out["firefly_logit"], errors="coerce")

    # Drop rows with NA in core fields
    out = out.dropna(subset=["t", "x", "y", "background_logit", "firefly_logit"])

    return out

def _normalize_gt_df(df: pd.DataFrame, src: Path) -> pd.DataFrame:
    """
    Ensure ground-truth dataframe contains: x, y, t
    Try common aliases if STRICT_COLUMNS is False.
    """
    df = _standardize_columns(df)

    x_col = "x" if "x" in df.columns else (None if STRICT_COLUMNS else ( "cx" if "cx" in df.columns else "x_center" if "x_center" in df.columns else None))
    y_col = "y" if "y" in df.columns else (None if STRICT_COLUMNS else ( "cy" if "cy" in df.columns else "y_center" if "y_center" in df.columns else None))
    t_col = "t" if "t" in df.columns else (None if STRICT_COLUMNS else ( "time" if "time" in df.columns else "frame" if "frame" in df.columns else "frame_num" if "frame_num" in df.columns else None))

    missing = [name for name, val in {"x": x_col, "y": y_col, "t": t_col}.items() if val is None]
    if missing:
        raise ValueError(
            f"{src.name}: required GT columns missing or not recognized: {missing}. "
            f"Have columns={sorted(df.columns)}"
        )

    out = df[[x_col, y_col, t_col]].rename(columns={x_col: "x", y_col: "y", t_col: "t"}).copy()
    out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("Int64")
    out["x"] = pd.to_numeric(out["x"], errors="coerce").astype("Int64")
    out["y"] = pd.to_numeric(out["y"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["t", "x", "y"])
    return out

def _species_from_name(path: Path) -> str:
    """
    Extract species name by matching any of SPECIES tokens at the start of the filename.
    More tolerant: it will find species anywhere in the basename, preferring a leading match.
    """
    name = path.stem.lower()
    # prioritize exact leading match: ^species_
    for s in SPECIES:
        if re.match(rf"^{re.escape(s.lower())}[_\-\.]", name):
            return s
    # otherwise look for token anywhere
    for s in SPECIES:
        if s.lower() in name:
            return s
    raise ValueError(f"Could not determine species for file: {path.name}")

def _collect_files(kind: str) -> list[Path]:
    """
    kind: 'resnet' (detections) or 'true' (ground truth)
    Collect files from INPUT_DIR matching either:
      <SPECIES>_<kind>.{csv|tsv|txt}
      <SPECIES>_<kind>_<anything>.{csv|tsv|txt}
    """
    root = Path(INPUT_DIR)
    if not root.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {root}")

    exts = (".csv", ".tsv", ".txt")
    paths: list[Path] = []
    for s in SPECIES:
        for e in exts:
            paths.extend(root.glob(f"{s}_{kind}{e}"))       # exact filename
            paths.extend(root.glob(f"{s}_{kind}_*{e}"))     # with descriptor

    uniq = sorted(set(paths))
    _debug(f"Found {len(uniq)} '{kind}' file(s).")
    return uniq

def _combine(kind: str) -> pd.DataFrame:
    """
    Combine all files of a given kind ('resnet' or 'true') into a single DataFrame.
    Adds 'species' column.
    """
    files = _collect_files(kind)
    if not files:
        _debug(f"No files found for kind='{kind}'. Returning empty DataFrame.")
        return pd.DataFrame()

    frames = []
    for p in files:
        sp = _species_from_name(p)
        df = _read_any_csv(p)
        if kind == "resnet":
            df = _normalize_det_df(df, p)
        else:
            df = _normalize_gt_df(df, p)
        df.insert(0, "species", sp)  # put species as first column
        frames.append(df)
        _debug(f"Loaded {len(df):,} rows from {p.name} [{sp}]")

    out = pd.concat(frames, ignore_index=True)

    # Optional de-dup
    if kind == "resnet" and DROP_DUPES_DET:
        before = len(out)
        out = out.drop_duplicates()
        _debug(f"Detections: dropped {before - len(out):,} duplicate rows.")
    if kind == "true" and DROP_DUPES_GT:
        before = len(out)
        out = out.drop_duplicates()
        _debug(f"Ground truth: dropped {before - len(out):,} duplicate rows.")

    # Sort consistently
    sort_cols = ["t", "species", "y", "x"] if "t" in out.columns else ["species", "y", "x"]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out

def main():
    # Combine detections (resnet)
    det = _combine("resnet")
    if det.empty:
        _debug("WARNING: No detection data combined.")
    else:
        _debug(f"Detections combined total: {len(det):,} rows.")
        # Validate final columns
        required_det_cols = ["species", "t", "x", "y", "background_logit", "firefly_logit"]
        missing = [c for c in required_det_cols if c not in det.columns]
        if missing:
            raise RuntimeError(f"Combined detections missing columns: {missing}")
        Path(OUTPUT_DETECTIONS).parent.mkdir(parents=True, exist_ok=True)
        det.to_csv(OUTPUT_DETECTIONS, sep=OUTPUT_DELIMITER, index=INDEX_IN_OUTPUT)
        _debug(f"Wrote detections → {OUTPUT_DETECTIONS}")

    # Combine ground truth (true)
    gt = _combine("true")
    if gt.empty:
        _debug("WARNING: No ground-truth data combined.")
    else:
        _debug(f"Ground truth combined total: {len(gt):,} rows.")
        required_gt_cols = ["species", "x", "y", "t"]
        missing = [c for c in required_gt_cols if c not in gt.columns]
        if missing:
            raise RuntimeError(f"Combined ground truth missing columns: {missing}")
        Path(OUTPUT_GROUND_TRUTH).parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(OUTPUT_GROUND_TRUTH, sep=OUTPUT_DELIMITER, index=INDEX_IN_OUTPUT)
        _debug(f"Wrote ground truth → {OUTPUT_GROUND_TRUTH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _debug(f"ERROR: {e}")
        sys.exit(1)
