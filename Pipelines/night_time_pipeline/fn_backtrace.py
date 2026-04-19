#!/usr/bin/env python3
from __future__ import annotations
import csv, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

TOL_PX = 3.0

def _euclid(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    dx = float(a[0])-float(b[0]); dy = float(a[1])-float(b[1])
    return math.hypot(dx, dy)

def _read_xy(path: Path) -> List[Tuple[int,int,int]]:
    out = []
    with path.open('r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = int(row.get('t') or row.get('frame') or 0)
                x = int(round(float(row['x']))); y = int(round(float(row['y'])))
                out.append((t,x,y))
            except Exception:
                continue
    return out

def _index_rows(path: Path) -> Dict[int, List[Tuple[int,int]]]:
    out: Dict[int, List[Tuple[int,int]]] = {}
    if not path.exists():
        return out
    with path.open('r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = int(row.get('frame') or row.get('t') or 0)
                x = int(round(float(row['x']))); y = int(round(float(row['y'])))
                out.setdefault(t, []).append((x,y))
            except Exception:
                continue
    return out

def backtrace_fns(audit_root: Path, video_stem: str, fns_csv: Path, out_csv: Optional[Path] = None):
    """
    For each FN (t,x,y) in fns_csv, scan the audit folders for the earliest stage
    that removed a matching point (~3px), and report:
      t,x,y, culprit_stage, reason, culprit_file
    """
    fns = _read_xy(fns_csv)
    stages_order = [
        ('02_recenter', ['removed__dim_seed.csv']),
        ('03_area_filter', ['removed__area_below_thr.csv']),
        ('04_cnn', ['removed__classified_background.csv']),
        ('07_merge', ['groups_and_winners.csv']),
        ('08_5_blob_area', ['removed__blob_area_gate_removed.csv']),
        ('08_6_neighbor_hunt', ['removed__dedupe.csv']),
        ('08_7_large_flash_bfs', ['replacements.csv']),
    ]
    rows_out = []
    for (t,x,y) in fns:
        found = False
        for stage, files in stages_order:
            sdir = audit_root / video_stem / stage
            if not sdir.exists():
                continue
            for fname in files:
                p = sdir / fname
                if not p.exists():
                    continue
                idx = _index_rows(p)
                pts = idx.get(t, [])
                if any(_euclid((x,y), pt) <= TOL_PX for pt in pts):
                    rows_out.append({'t': t, 'x': x, 'y': y, 'culprit_stage': stage, 'culprit_file': str(p.name)})
                    found = True
                    break
            if found:
                break
        if not found:
            rows_out.append({'t': t, 'x': x, 'y': y, 'culprit_stage': 'UNK', 'culprit_file': ''})

    out_csv = out_csv or (fns_csv.parent / "fn_backtrace.csv")
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t','x','y','culprit_stage','culprit_file'])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"[fn_backtrace] wrote: {out_csv}")
