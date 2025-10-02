#!/usr/bin/env python3
"""
Stage 13 — Audit-trail explainer for FNs & FPs

What it does
------------
For each threshold folder produced by Stage 9 (e.g., ROOT/stage9 validation/<video>/thr_4.0/),
this stage scans FN and FP CSVs and traces each item across the audited stage snapshots
(saved by the AuditTrail). It prints a per-item trace showing:

• Presence/absence after each audited stage (01→08.7).
• If it disappeared between stage X → Y, tries to fetch the reason from Y/removed.csv.
• For FPs (which survived to the end), it dumps key attributes from Stage 4 (class/confidence/logits)
  and notes any Stage 7 merges (if pairs.csv is present).

Outputs
-------
Under AUDIT_ROOT/13_trace_reports/<video>/<thr_name>/:
  - fn_traces.txt : human-readable traces for all FNs at this threshold
  - fp_traces.txt : human-readable traces for all FPs at this threshold
  - fn_summary.csv: one line per FN with drop_stage and reason (if found)
  - fp_summary.csv: one line per FP with selected attributes (class/conf/logits, etc.)

Notes
-----
• Matching radius defaults to 4 px. You can override via radius_px.
• This stage is resilient to variations in file naming inside the audit folders:
  it will try snapshot.csv, <video>.csv, and <video>_snapshot.csv in each stage dir.
• Snapshots are filtered to the target video by filename match if multiple CSVs exist.

Add to orchestrator (imports):
    from stage13_audit_analysis import stage13_audit_trail_analysis

Call after Stage 12 (only if Stage 9 ran):
    if ran_stage9:
        stage13_audit_trail_analysis(
            stage9_video_dir=DIR_STAGE9_OUT / base,
            audit_root=AUDIT_ROOT,
            pred_csv_path=csv_path,
            gt_csv_path=GT_CSV_PATH,
            gt_t_offset=GT_T_OFFSET,
            radius_px=4.0,
            verbose=True,
        )
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import csv, json, math, re, sys

# ──────────────────────────────────────────────────────────────
# Helpers for robust file discovery
# ──────────────────────────────────────────────────────────────
def _read_csv_rows(p: Path) -> List[dict]:
    try:
        with p.open('r', newline='') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[stage13] WARN: failed reading {p}: {e}")
        return []

def _write_csv_rows(p: Path, rows: List[dict], fieldnames: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})

def _write_text(p: Path, lines: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

def _case_find_csv(dir_path: Path, contains: str) -> List[Path]:
    """Return all CSVs under dir_path whose name case-insensitively contains token 'contains'."""
    token = contains.lower()
    return sorted([p for p in dir_path.glob("*.csv") if token in p.name.lower()])

def _resolve_stage_file(stage_dir: Path, base_stem: str, kind: str) -> Optional[Path]:
    """
    Try to locate a file for a given 'kind' in a stage audit directory.
    kind ∈ {'snapshot','removed','kept','pairs'}.
    """
    prefer_map = {
        'snapshot': [ 'snapshot.csv', f'{base_stem}.csv', f'{base_stem}_snapshot.csv' ],
        'removed' : [ 'removed.csv',  f'{base_stem}_removed.csv', f'removed_{base_stem}.csv' ],
        'kept'    : [ 'kept.csv',     f'{base_stem}_kept.csv',    f'kept_{base_stem}.csv'    ],
        'pairs'   : [ 'pairs.csv',    f'{base_stem}_pairs.csv',   f'pairs_{base_stem}.csv'   ],
    }
    # Try preferred names first
    for name in prefer_map.get(kind, []):
        p = stage_dir / name
        if p.exists():
            return p
    # Fallback by token search
    hits = _case_find_csv(stage_dir, kind if kind != 'snapshot' else 'snapshot')
    if not hits and kind == 'snapshot':
        # snapshot could also just be the only CSV that is not removed/kept/pairs
        all_csv = sorted(stage_dir.glob("*.csv"))
        filtered = [p for p in all_csv if all(tok not in p.name.lower() for tok in ('removed','kept','pairs'))]
        if len(filtered) == 1:
            return filtered[0]
        # Prefer one that contains the base_stem
        for p in filtered:
            if base_stem.lower() in p.stem.lower():
                return p
        return filtered[0] if filtered else None
    if len(hits) == 1: return hits[0]
    # prefer a hit with base_stem in name
    for p in hits:
        if base_stem.lower() in p.stem.lower():
            return p
    return hits[0] if hits else None

def _parse_int(v, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default

def _row_center(r: dict) -> Tuple[float,float]:
    try:
        x = float(r.get('x', 0)); y = float(r.get('y', 0))
        w = float(r.get('w', 10)); h = float(r.get('h', 10))
        sem = str(r.get('xy_semantics','')).lower()
        if sem == 'center':
            return (x, y)
        return (x + w/2.0, y + h/2.0)
    except Exception:
        return (0.0, 0.0)

def _near(r: dict, t_norm: int, x: float, y: float, radius_px: float) -> bool:
    try:
        t = _parse_int(r.get('frame', r.get('t', -1)), -99)
        if t != t_norm: return False
        cx, cy = _row_center(r)
        return (cx - x)**2 + (cy - y)**2 <= radius_px*radius_px
    except Exception:
        return False

from dataclasses import dataclass
@dataclass
class StageSpec:
    tag: str               # folder name under audit root
    has_removed: bool      # whether we expect removed.csv
    has_pairs: bool        # whether we expect pairs.csv

# Order matters (pipeline order)
PIPELINE_STAGES: List[StageSpec] = [
    StageSpec('01_detect',            False, False),
    StageSpec('02_recenter',          True,  False),
    StageSpec('03_area_filter',       True,  False),
    StageSpec('04_cnn',               True,  False),
    StageSpec('07_merge',             True,  True),   # pairs.csv may exist
    StageSpec('08_gauss',             False, False),
    StageSpec('08_5_blob_area',       True,  False),
    StageSpec('08_7_large_flash_bfs', False, False),
]

# ──────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────
def _presence_trace(audit_root: Path, base_stem: str, t_norm: int, x: float, y: float, radius_px: float) -> Tuple[List[Tuple[str,bool]], Optional[str], Optional[dict]]:
    """
    Returns:
      - presence list [(stage_tag, present_bool), ...] in pipeline order
      - drop_stage (where it disappeared), or None if present through all
      - drop_reason_row (row from removed.csv if found), else None
    """
    presence: List[Tuple[str,bool]] = []
    drop_stage = None
    drop_reason = None

    # Cache last 'present' to find first drop
    last_present = False
    for i, spec in enumerate(PIPELINE_STAGES):
        stage_dir = audit_root / spec.tag
        snap = _resolve_stage_file(stage_dir, base_stem, 'snapshot')
        present = False
        if snap and snap.exists():
            rows = _read_csv_rows(snap)
            present = any(_near(r, t_norm, x, y, radius_px) for r in rows)
        presence.append((spec.tag, present))

        if i > 0 and last_present and not present and drop_stage is None:
            # Disappeared here
            drop_stage = spec.tag
            if spec.has_removed:
                rem = _resolve_stage_file(stage_dir, base_stem, 'removed')
                if rem and rem.exists():
                    rem_rows = _read_csv_rows(rem)
                    # Try exact frame + within radius
                    near_rows = [r for r in rem_rows if _near(r, t_norm, x, y, radius_px)]
                    # If we didn't find, relax: match only by frame and closest distance
                    if not near_rows:
                        frame_rows = [r for r in rem_rows if _parse_int(r.get('frame', r.get('t', -1)), -1) == t_norm]
                        best = None; bestd = 1e9
                        for r in frame_rows:
                            cx, cy = _row_center(r)
                            d2 = (cx - x)**2 + (cy - y)**2
                            if d2 < bestd:
                                bestd = d2; best = r
                        if best is not None: near_rows = [best]
                    if near_rows:
                        drop_reason = near_rows[0]
            # Once we've detected the first drop, we can stop early, but keep scanning to fill presence fully
        last_present = present

    return presence, drop_stage, drop_reason

def _pairs_note(audit_root: Path, base_stem: str, t_norm: int, x: float, y: float, radius_px: float) -> Optional[dict]:
    """Look into 07_merge/pairs.csv to see if our target was part of a merge group."""
    stage_dir = audit_root / '07_merge'
    pairs = _resolve_stage_file(stage_dir, base_stem, 'pairs')
    if not (pairs and pairs.exists()): return None
    rows = _read_csv_rows(pairs)
    # We try a few common schemas:
    #   kept_x, kept_y, kept_weight, dropped_x, dropped_y (maybe multiple); or JSON-ish fields.
    for r in rows:
        try:
            t = _parse_int(r.get('frame', r.get('t', -1)), -1)
            if t != t_norm: continue
            # check kept
            kx = float(r.get('kept_x', 'nan')); ky = float(r.get('kept_y', 'nan'))
            if not math.isnan(kx) and (kx - x)**2 + (ky - y)**2 <= radius_px*radius_px:
                return {'role':'kept', 'row': r}
            # check a single dropped
            dx = r.get('dropped_x', None); dy = r.get('dropped_y', None)
            if dx is not None and dy is not None:
                dx = float(dx); dy = float(dy)
                if (dx - x)**2 + (dy - y)**2 <= radius_px*radius_px:
                    return {'role':'dropped', 'row': r}
            # check list-like fields if present
            for k in ('dropped_list','group_rows','members'):
                if k in r and r[k]:
                    # try to parse as JSON or semi-colon tuples "x:y;..."
                    val = r[k]
                    try:
                        arr = json.loads(val)
                        # expect list of dicts with x,y
                        for it in arr:
                            dx = float(it.get('x', float('nan'))); dy = float(it.get('y', float('nan')))
                            if not math.isnan(dx) and (dx - x)**2 + (dy - y)**2 <= radius_px*radius_px:
                                return {'role':'dropped', 'row': r}
                    except Exception:
                        # naive parse x_y pairs
                        toks = re.split(r'[;|,]\s*', str(val))
                        for tok in toks:
                            m = re.match(r'.*?(-?\d+\.?\d*).+?(-?\d+\.?\d*)', tok)
                            if m:
                                dx = float(m.group(1)); dy = float(m.group(2))
                                if (dx - x)**2 + (dy - y)**2 <= radius_px*radius_px:
                                    return {'role':'dropped', 'row': r}
        except Exception:
            continue
    return None

def _format_reason(stage_tag: str, reason_row: Optional[dict]) -> str:
    if not reason_row:
        return "(no explicit reason found)"
    keys_by_stage = {
        '02_recenter':      ['reason','patch_max','x','y','w','h'],
        '03_area_filter':   ['reason','area','w','h','x','y'],
        '04_cnn':           ['reason','class','firefly_confidence','background_logit','firefly_logit','x','y'],
        '07_merge':         ['reason','kept_weight','x','y'],
        '08_5_blob_area':   ['reason','largest_cc_area','x','y','w','h'],
    }
    keys = keys_by_stage.get(stage_tag, [])
    if not keys:
        # dump a short JSON of the row
        # keep only a handful of common fields if present
        common = ['frame','x','y','w','h','reason']
        d = {k: reason_row.get(k) for k in reason_row.keys() if k in common}
        if not d:
            d = {k: reason_row.get(k) for k in list(reason_row.keys())[:6]}
        return json.dumps(d)
    out = []
    for k in keys:
        if k in reason_row and reason_row.get(k) != '':
            out.append(f"{k}={reason_row.get(k)}")
    return ", ".join(out) if out else json.dumps(reason_row)

def _extract_stage4_attrs(audit_root: Path, base_stem: str, t_norm: int, x: float, y: float, radius_px: float) -> Optional[dict]:
    stage_dir = audit_root / '04_cnn'
    snap = _resolve_stage_file(stage_dir, base_stem, 'snapshot')
    if not (snap and snap.exists()): return None
    rows = _read_csv_rows(snap)
    near = [r for r in rows if _near(r, t_norm, x, y, radius_px)]
    if not near:
        # relax: match by frame, closest
        frame_rows = [r for r in rows if _parse_int(r.get('frame', r.get('t', -1)), -1) == t_norm]
        best = None; bestd = 1e9
        for r in frame_rows:
            cx, cy = _row_center(r); d2 = (cx - x)**2 + (cy - y)**2
            if d2 < bestd: bestd = d2; best = r
        near = [best] if best is not None else []
    if not near: return None
    r = near[0]
    return {
        'class': r.get('class', ''),
        'firefly_confidence': r.get('firefly_confidence', ''),
        'background_logit': r.get('background_logit', ''),
        'firefly_logit': r.get('firefly_logit', ''),
    }

# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def stage13_audit_trail_analysis(
    *,
    stage9_video_dir: Path,
    audit_root: Path,
    pred_csv_path: Path,
    gt_csv_path: Path,
    gt_t_offset: int,
    radius_px: float = 4.0,
    verbose: bool = True,
) -> None:
    """
    Build FN/FP trace reports for one <video> (stage9_video_dir corresponds to that video).
    """
    base_stem = pred_csv_path.stem  # e.g., <video>.csv -> <video>
    # Prepare output root
    out_root = audit_root / '13_trace_reports' / base_stem
    out_root.mkdir(parents=True, exist_ok=True)

    # Find threshold folders under stage9_video_dir
    thr_dirs = [p for p in stage9_video_dir.iterdir() if p.is_dir() and p.name.lower().startswith('thr')]
    thr_dirs.sort(key=lambda p: p.name)

    if verbose:
        print(f"[stage13] base={base_stem}  thresholds={len(thr_dirs)}  audit_root={audit_root}")

    for thr_dir in thr_dirs:
        thr_name = thr_dir.name
        # Locate FN/FP CSVs (case-insensitive)
        fn_csvs = _case_find_csv(thr_dir, 'fn')
        fp_csvs = _case_find_csv(thr_dir, 'fp')
        if not fn_csvs and not fp_csvs:
            if verbose:
                print(f"[stage13] {thr_name}: no FN/FP CSVs found; skipping.")
            continue

        out_thr = out_root / thr_name
        out_thr.mkdir(parents=True, exist_ok=True)
        fn_lines: List[str] = []
        fp_lines: List[str] = []
        fn_summary_rows: List[dict] = []
        fp_summary_rows: List[dict] = []

        # Process FNs
        for fn_csv in fn_csvs:
            for r in _read_csv_rows(fn_csv):
                try:
                    t = _parse_int(r.get('t', r.get('frame', 0)), 0)
                    x = float(r.get('x', 0)); y = float(r.get('y', 0))
                except Exception:
                    continue
                pres, drop_stage, reason_row = _presence_trace(audit_root, base_stem, t, x, y, radius_px)
                lines = [f"FN @ {thr_name}: t={t}, x={int(round(x))}, y={int(round(y))}"]
                lines.append("  presence: " + " → ".join([f"{tag.split('_')[0]}:{'Y' if ok else '-'}" for tag, ok in pres]))
                if drop_stage:
                    lines.append(f"  DROPPED at stage: {drop_stage}")
                    lines.append(f"  reason: {_format_reason(drop_stage, reason_row)}")
                else:
                    lines.append("  Not present since Stage 1 (detector miss) or consolidated by 8.7; check 01_detect/snapshot and stage8.7 replacements.")
                fn_lines.extend(lines + [""])
                fn_summary_rows.append({
                    'thr': thr_name, 't': t, 'x': int(round(x)), 'y': int(round(y)),
                    'drop_stage': drop_stage or '',
                    'reason': _format_reason(drop_stage, reason_row) if drop_stage else 'not_found',
                })

        # Process FPs
        for fp_csv in fp_csvs:
            for r in _read_csv_rows(fp_csv):
                try:
                    t = _parse_int(r.get('t', r.get('frame', 0)), 0)
                    x = float(r.get('x', 0)); y = float(r.get('y', 0))
                except Exception:
                    continue
                pres, _, _ = _presence_trace(audit_root, base_stem, t, x, y, radius_px)
                pairs = _pairs_note(audit_root, base_stem, t, x, y, radius_px)
                s4 = _extract_stage4_attrs(audit_root, base_stem, t, x, y, radius_px)
                lines = [f"FP @ {thr_name}: t={t}, x={int(round(x))}, y={int(round(y))}"]
                lines.append("  presence: " + " → ".join([f"{tag.split('_')[0]}:{'Y' if ok else '-'}" for tag, ok in pres]))
                if s4:
                    lines.append(f"  stage4: class={s4.get('class','')}, conf={s4.get('firefly_confidence','')}, "
                                 f"bg_logit={s4.get('background_logit','')}, ff_logit={s4.get('firefly_logit','')}")
                if pairs:
                    role = pairs.get('role','')
                    lines.append(f"  stage7 merge: role={role} (see 07_merge/pairs.csv)")
                fp_lines.extend(lines + [""])
                fp_summary_rows.append({
                    'thr': thr_name, 't': t, 'x': int(round(x)), 'y': int(round(y)),
                    'stage4_class': (s4 or {}).get('class',''),
                    'stage4_conf': (s4 or {}).get('firefly_confidence',''),
                    'stage4_bg_logit': (s4 or {}).get('background_logit',''),
                    'stage4_ff_logit': (s4 or {}).get('firefly_logit',''),
                    'stage7_role': (pairs or {}).get('role',''),
                })

        # write outputs for this threshold
        if fn_lines:
            _write_text(out_thr / "fn_traces.txt", fn_lines)
            _write_csv_rows(out_thr / "fn_summary.csv", fn_summary_rows,
                            fieldnames=['thr','t','x','y','drop_stage','reason'])
        if fp_lines:
            _write_text(out_thr / "fp_traces.txt", fp_lines)
            _write_csv_rows(out_thr / "fp_summary.csv", fp_summary_rows,
                            fieldnames=['thr','t','x','y','stage4_class','stage4_conf','stage4_bg_logit','stage4_ff_logit','stage7_role'])

        if verbose:
            print(f"[stage13] Wrote: {out_thr}  (FN:{len(fn_summary_rows)}  FP:{len(fp_summary_rows)})")

    if verbose:
        print("[stage13] Done.")
