from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Box:
    x: float  # top-left x (pixels)
    y: float  # top-left y (pixels)
    w: float  # width (pixels)
    h: float  # height (pixels)
    traj_id: int | None = None


def load_annotations_csv(csv_path: str | Path) -> dict[int, list[Box]]:
    """
    Load annotations from a CSV file.

    Expected columns:
      - x, y, w, h
      - frame (preferred) OR t

    Coordinates are expected to be in ORIGINAL pixel coordinates (top-left + width/height),
    matching `test1/tools/firefly flash annotation tool v2.py`.
    """
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    by_frame: dict[int, list[Box]] = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return by_frame

        fieldnames = {c.strip().lower() for c in reader.fieldnames}
        time_key = "frame" if "frame" in fieldnames else "t" if "t" in fieldnames else None
        if time_key is None:
            raise ValueError(
                f"CSV must contain a 'frame' or 't' column, got: {sorted(fieldnames)}"
            )

        traj_key = None
        for cand in ("traj_id", "track_id"):
            if cand in fieldnames:
                traj_key = cand
                break

        for row in reader:
            try:
                frame_index = int(float(row[time_key]))
                x = float(row["x"])
                y = float(row["y"])
                w = float(row["w"])
                h = float(row["h"])
            except Exception:
                continue

            if frame_index < 0 or w <= 0 or h <= 0:
                continue

            traj_id: int | None = None
            if traj_key is not None:
                try:
                    raw = row.get(traj_key)
                    if raw is not None and str(raw).strip() != "":
                        traj_id = int(float(raw))
                except Exception:
                    traj_id = None

            by_frame.setdefault(frame_index, []).append(Box(x=x, y=y, w=w, h=h, traj_id=traj_id))
    return by_frame


def iter_boxes_for_frame(
    annotations: dict[int, list[Box]], frame_index: int
) -> Iterable[Box]:
    return annotations.get(frame_index, [])
