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


def _norm_col_name(name: str) -> str:
    name = str(name).strip()
    # Handle UTF-8 BOM that sometimes appears in the first header cell.
    if name.startswith("\ufeff"):
        name = name.lstrip("\ufeff")
    return name.lower()


def _pick_unique_col(
    cols_norm_to_orig: dict[str, list[str]], candidates_norm: Iterable[str], label: str
) -> str | None:
    for cand in candidates_norm:
        key = _norm_col_name(cand)
        origs = cols_norm_to_orig.get(key)
        if not origs:
            continue
        if len(origs) > 1:
            raise ValueError(f"CSV has multiple columns matching {label}: {sorted(origs)}")
        return origs[0]
    return None


def load_annotations_csv(csv_path: str | Path) -> dict[int, list[Box]]:
    """
    Load annotations from a CSV file.

    Expected columns (extra columns are ignored):
      - x, y, w, h
      - frame/time column: frame | frame_idx | t
      - optional: traj_id | track_id

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

        cols_norm_to_orig: dict[str, list[str]] = {}
        for c in reader.fieldnames:
            cols_norm_to_orig.setdefault(_norm_col_name(c), []).append(c)

        time_key = _pick_unique_col(
            cols_norm_to_orig,
            ("frame", "frame_idx", "frame_index", "frame_id", "t"),
            label="frame/time column",
        )
        if time_key is None:
            raise ValueError(
                "CSV must contain a frame/time column (frame | frame_idx | t). "
                f"Got: {sorted(cols_norm_to_orig)}"
            )

        x_key = _pick_unique_col(cols_norm_to_orig, ("x",), label="x")
        y_key = _pick_unique_col(cols_norm_to_orig, ("y",), label="y")
        w_key = _pick_unique_col(cols_norm_to_orig, ("w",), label="w")
        h_key = _pick_unique_col(cols_norm_to_orig, ("h",), label="h")
        missing = [k for k, v in (("x", x_key), ("y", y_key), ("w", w_key), ("h", h_key)) if v is None]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Got: {sorted(cols_norm_to_orig)}")

        traj_key = _pick_unique_col(cols_norm_to_orig, ("traj_id", "track_id"), label="traj/track id")

        for row in reader:
            try:
                frame_index = int(float(row[time_key]))
                x = float(row[x_key])
                y = float(row[y_key])
                w = float(row[w_key])
                h = float(row[h_key])
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
