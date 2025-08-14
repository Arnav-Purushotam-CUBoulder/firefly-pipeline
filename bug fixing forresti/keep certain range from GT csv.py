#!/usr/bin/env python3
"""
Filter rows in an x,y,t CSV by inclusive frame ranges.

Input CSV schema (set DELIMITER accordingly):
    x,y,t
Example:
    333,757,0
    1489,604,0
    1585,525,0

Behavior:
  • Keep only rows whose t falls within any of the inclusive ranges in FRAME_RANGES.
  • If FRAME_RANGES is empty or normalizes to empty, keep ALL rows.
  • Writes a new CSV with the same three columns: x, y, t.
"""

from pathlib import Path
import csv

# ─── GLOBALS: set these to your environment ───────────────────

INPUT_CSV   = Path('/Users/arnavps/Desktop/to send/resnet forresti/ground truth csv/xyt ground truth.csv')
OUTPUT_CSV  = Path('/Users/arnavps/Desktop/to send/resnet forresti/ground truth csv/xyt ground truth forresti 4k-4_5k.csv')

# CSV delimiter — set to '\t' for TSV, ',' for CSV
DELIMITER   = ','

# Inclusive frame ranges (on column 't'); example: [[4000, 4500], [9000, 9050]]
# If empty, all rows are kept.
FRAME_RANGES = [[4000, 4500]]

# ─── helpers ───────────────────────────────────────────────────

def _normalize_ranges(ranges):
    """Clean, sort, and merge inclusive [start, end] ranges."""
    clean = []
    for pair in ranges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        if a > b:
            a, b = b, a
        clean.append([a, b])
    if not clean:
        return []
    clean.sort(key=lambda ab: ab[0])
    merged = [clean[0]]
    for s, e in clean[1:]:
        ms, me = merged[-1]
        if s <= me + 1:
            merged[-1][1] = max(me, e)
        else:
            merged.append([s, e])
    return merged

def _in_ranges(x, ranges_merged):
    """Return True if x is within any inclusive [s,e] range. If no ranges, allow all."""
    if not ranges_merged:
        return True
    for s, e in ranges_merged:
        if s <= x <= e:
            return True
        if x < s:
            return False
    return False

# ─── core ──────────────────────────────────────────────────────

def filter_csv_by_t_ranges(input_csv: Path, output_csv: Path, ranges) -> None:
    ranges_merged = _normalize_ranges(ranges)
    apply_filter = bool(ranges_merged)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total = kept = 0
    out_fields = ['x', 'y', 't']

    with input_csv.open('r', newline='') as fin, output_csv.open('w', newline='') as fout:
        reader = csv.DictReader(fin, delimiter=DELIMITER)
        if 't' not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain a 't' column.")
        # Write header
        writer = csv.DictWriter(fout, fieldnames=out_fields, delimiter=DELIMITER)
        writer.writeheader()

        for row in reader:
            total += 1
            try:
                t = int(round(float(row['t'])))
                x = int(round(float(row['x'])))
                y = int(round(float(row['y'])))
            except Exception:
                # skip malformed rows
                continue

            if apply_filter and not _in_ranges(t, ranges_merged):
                continue

            writer.writerow({'x': x, 'y': y, 't': t})
            kept += 1

    print("Done.")
    print(f"  Input rows  : {total}")
    print(f"  Kept rows   : {kept}")
    print(f"  Dropped rows: {total - kept}")
    print(f"  Ranges used : {ranges_merged if apply_filter else '[ALL FRAMES]'}")
    print(f"  Output path : {output_csv}")

# ─── main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    filter_csv_by_t_ranges(INPUT_CSV, OUTPUT_CSV, FRAME_RANGES)
