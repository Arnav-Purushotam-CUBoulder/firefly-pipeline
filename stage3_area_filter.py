import csv
from pathlib import Path

def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        fieldnames = ['frame','x','y','w','h']
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        for r in rows:
            wri.writerow(r)

def filter_boxes_by_area(csv_path: Path, area_threshold_px: int, snapshot_csv_path: Path | None = None):
    """
    Filters rows in-place by area (w*h >= area_threshold_px) and ALSO saves
    an identical 'snapshot' CSV that will NOT be modified by later stages.

    Parameters
    ----------
    csv_path : Path
        The working CSV path that later stages may overwrite/edit.
    area_threshold_px : int
        Minimum pixel area (inclusive) for a box to be kept.
    snapshot_csv_path : Path | None
        Where to save the frozen copy after area filtering. If None, uses
        '<csv_path.stem>_area_snapshot.csv' in the same directory.
    """
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return

    filtered = []
    for r in rows:
        try:
            w = int(r['w']); h = int(r['h'])
            if w * h >= area_threshold_px:
                filtered.append({
                    'frame': int(r['frame']),
                    'x': int(r['x']), 'y': int(r['y']),
                    'w': w, 'h': h
                })
        except Exception:
            continue

    # 1) overwrite the working CSV (as before)
    _write_csv(csv_path, filtered)

    # 2) save a frozen snapshot copy for visualization later
    if snapshot_csv_path is None:
        snapshot_csv_path = csv_path.with_name(csv_path.stem + '_area_snapshot.csv')
    _write_csv(snapshot_csv_path, filtered)
