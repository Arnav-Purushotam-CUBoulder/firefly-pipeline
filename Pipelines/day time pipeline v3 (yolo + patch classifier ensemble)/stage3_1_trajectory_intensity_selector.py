#!/usr/bin/env python3
from __future__ import annotations

"""
Stage 3.1: group Stage 3 detections into trajectories and select "flash-like" ones.

This stage does NOT remove trajectories based on motion. It:
  1) links detections into trajectories in (x,y,t),
  2) computes an intensity curve per trajectory from the saved Stage3 crops,
  3) selects "flash-like" (hill-shaped) curves for downstream inspection.

Input:
  STAGE3_DIR/<video_stem>/<video_stem>_patches.csv

Output:
  STAGE3_DIR/<video_stem>/<video_stem>_patches_motion.csv          (selected only)
  STAGE3_DIR/<video_stem>/<video_stem>_patches_motion_all.csv

Output rows preserve all Stage 3 columns and add:
  traj_id, traj_size, traj_motion_xy, traj_intensity_range, traj_is_selected
"""

import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import params


@dataclass
class _Det:
    idx: int
    t: int
    cx: float
    cy: float
    det_id: int


@dataclass
class _Track:
    id: int
    det_id: int
    dets: List[_Det]

    @property
    def last_t(self) -> int:
        return self.dets[-1].t

    @property
    def last_cx(self) -> float:
        return self.dets[-1].cx

    @property
    def last_cy(self) -> float:
        return self.dets[-1].cy


def _load_stage3_csv(stem: str) -> tuple[Path, List[dict], List[_Det]]:
    s3_dir = params.STAGE3_DIR / stem
    in_csv = s3_dir / f"{stem}_patches.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Stage 3 CSV not found: {in_csv}")

    rows: List[dict] = []
    dets: List[_Det] = []
    with in_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            rows.append(row)
            try:
                t = int(row.get("frame_idx") or row.get("frame_number") or 0)
                x = float(row["x"])
                y = float(row["y"])
                w = float(row.get("w", getattr(params, "PATCH_SIZE_PX", 10)))
                h = float(row.get("h", getattr(params, "PATCH_SIZE_PX", 10)))
                det_id = int(float(row.get("det_id", -1)))
            except Exception:
                continue
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            dets.append(_Det(idx=i, t=t, cx=cx, cy=cy, det_id=det_id))
    return in_csv, rows, dets


def _group_by_frame(dets: List[_Det]) -> Dict[int, List[_Det]]:
    by_t: Dict[int, List[_Det]] = {}
    for d in dets:
        by_t.setdefault(int(d.t), []).append(d)
    return by_t


def _link_trajectories(
    by_t: Dict[int, List[_Det]],
    *,
    link_radius_px: float,
    max_frame_gap: int,
    time_scale: float,
) -> List[_Track]:
    active: Dict[int, _Track] = {}
    finished: List[_Track] = []
    next_id = 0

    def _finalize_stale(cur_t: int) -> None:
        nonlocal active, finished
        stale_ids = [tid for tid, tr in active.items() if (cur_t - tr.last_t) > max_frame_gap]
        for tid in stale_ids:
            finished.append(active.pop(tid))

    frames = sorted(by_t.keys())
    for t in frames:
        _finalize_stale(int(t))

        dets = by_t.get(t, [])
        if not dets:
            continue

        candidates: List[Tuple[float, int, int]] = []  # (dist, track_id, det_idx)
        active_items = list(active.items())
        for det_i, d in enumerate(dets):
            for tid, tr in active_items:
                if int(d.det_id) != int(tr.det_id):
                    continue
                gap = int(d.t - tr.last_t)
                if gap <= 0 or gap > max_frame_gap:
                    continue
                dx = float(d.cx - tr.last_cx)
                dy = float(d.cy - tr.last_cy)
                # Spatial linking threshold. Time gap is handled via max_frame_gap.
                # Treat link_radius_px as "max XY pixels per frame", so allow
                # proportional drift over larger gaps.
                dist_xy = math.sqrt(dx * dx + dy * dy)
                max_dist = float(link_radius_px) * max(1.0, float(gap))
                if dist_xy <= max_dist:
                    # Still prefer smaller temporal gaps if time_scale > 0
                    # by using a tie-break distance that lightly penalizes gap.
                    dist = float(dist_xy) + float(time_scale) * float(gap)
                    candidates.append((dist, tid, det_i))

        candidates.sort(key=lambda x: x[0])
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()

        for dist, tid, det_i in candidates:
            if tid in assigned_tracks or det_i in assigned_dets:
                continue
            active[tid].dets.append(dets[det_i])
            assigned_tracks.add(tid)
            assigned_dets.add(det_i)

        for det_i, d in enumerate(dets):
            if det_i in assigned_dets:
                continue
            active[next_id] = _Track(id=next_id, det_id=int(d.det_id), dets=[d])
            next_id += 1

    finished.extend(active.values())
    return finished


def _track_motion_xy(track: _Track) -> float:
    """A simple 2D motion metric: total path length in x/y (pixels)."""
    if len(track.dets) < 2:
        return 0.0
    motion = 0.0
    prev = track.dets[0]
    for cur in track.dets[1:]:
        dx = float(cur.cx - prev.cx)
        dy = float(cur.cy - prev.cy)
        motion += float(math.sqrt(dx * dx + dy * dy))
        prev = cur
    return float(motion)


_CROP_RE = re.compile(
    r"^f_(?P<t>\d+)_x(?P<x>-?\d+)_y(?P<y>-?\d+)_w(?P<w>\d+)_h(?P<h>\d+)_p(?P<p>\d+(?:\.\d+)?)\.png$"
)

_EXPORTED_CROP_RE = re.compile(
    r"^(?P<turn>\d+)_t(?P<t>\d+)_x(?P<x>-?\d+)_y(?P<y>-?\d+)\.png$"
)


def _index_stage3_crops(pos_dir: Path) -> Dict[tuple[int, int, int, int, int], Path]:
    """Map (t,x,y,w,h) -> crop path (choosing highest p if duplicates)."""
    best: Dict[tuple[int, int, int, int, int], tuple[float, Path]] = {}
    if not pos_dir.exists():
        return {}
    for p in pos_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        m = _CROP_RE.match(p.name)
        if not m:
            continue
        try:
            key = (
                int(m.group("t")),
                int(m.group("x")),
                int(m.group("y")),
                int(m.group("w")),
                int(m.group("h")),
            )
            prob = float(m.group("p"))
        except Exception:
            continue
        prev = best.get(key)
        if prev is None or prob > prev[0]:
            best[key] = (prob, p)
    return {k: v for k, (_, v) in best.items()}


def _safe_rmtree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass


def _collect_highvar_boxes(highvar_root: Path) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """Parse high-var trajectory crop folders to build boxes_by_frame.

    Returns mapping frame_idx -> list of (x, y, w, h) using PATCH_SIZE_PX.
    """
    boxes_by_t: Dict[int, List[Tuple[int, int, int, int]]] = {}
    if not highvar_root.exists() or not highvar_root.is_dir():
        return boxes_by_t

    w = int(getattr(params, "PATCH_SIZE_PX", 10))
    h = int(getattr(params, "PATCH_SIZE_PX", 10))

    for traj_dir in sorted(p for p in highvar_root.iterdir() if p.is_dir()):
        for p in traj_dir.iterdir():
            if not p.is_file() or p.suffix.lower() != ".png":
                continue
            m = _EXPORTED_CROP_RE.match(p.name)
            if not m:
                continue
            try:
                t = int(m.group("t"))
                x = int(m.group("x"))
                y = int(m.group("y"))
            except Exception:
                continue
            boxes_by_t.setdefault(t, []).append((x, y, w, h))
    return boxes_by_t


def _collect_stage31_boxes_from_csv(csv_path: Path) -> Dict[int, List[Tuple[int, int, int, int, int]]]:
    """Load boxes (and selection flag) from a Stage3.1 CSV.

    Returns mapping frame_idx -> list of (x, y, w, h, is_selected).
    """
    boxes_by_t: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    if not csv_path.exists():
        return boxes_by_t
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = int(row.get("frame_idx") or row.get("frame_number") or 0)
                x = int(float(row["x"]))
                y = int(float(row["y"]))
                w = int(float(row.get("w", getattr(params, "PATCH_SIZE_PX", 10))))
                h = int(float(row.get("h", getattr(params, "PATCH_SIZE_PX", 10))))
            except Exception:
                continue
            sel = 0
            sel_raw = row.get("traj_is_selected")
            if sel_raw is not None:
                s = str(sel_raw).strip()
                if s not in {"", "0", "False", "false"}:
                    sel = 1
            boxes_by_t.setdefault(t, []).append((x, y, w, h, sel))
    return boxes_by_t


def _render_boxes_video(
    *,
    video_path: Path,
    out_path: Path,
    boxes_by_t: Dict[int, List[Tuple[int, int, int, int, int]]],
) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    max_frames = int(getattr(params, "MAX_FRAMES", 0) or 0)
    if max_frames <= 0:
        max_frames = total if total > 0 else 0

    fps = float(getattr(params, "RENDER_FPS_HINT", None) or fps_src)
    codec = str(getattr(params, "RENDER_CODEC", "mp4v"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(W), int(H)), isColor=True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    t = 0
    try:
        while True:
            if max_frames and t >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if t in boxes_by_t:
                for (x, y, w, h, is_selected) in boxes_by_t[t]:
                    x0 = int(x)
                    y0 = int(y)
                    x1 = x0 + int(w)
                    y1 = y0 + int(h)
                    # selected=red, rejected=blue
                    color = (0, 0, 255) if is_selected else (255, 0, 0)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
            writer.write(frame)
            t += 1
    finally:
        cap.release()
        writer.release()


def _sum_png_pixels(path: Path) -> int:
    """Sum of all pixel values in the crop (all channels)."""
    from PIL import Image  # pillow

    with Image.open(path) as im:
        im = im.convert("RGB")
        # 10x10 is tiny; pure-Python summation is fine.
        return int(sum(r + g + b for (r, g, b) in im.getdata()))


def _write_trajectory_intensity_svg(
    *,
    out_path: Path,
    title: str,
    curves: List[Tuple[int, List[int], List[int], int]],
) -> None:
    """Write one SVG with many curves.

    curves: [(traj_id, xs_rel, ys_sum, is_selected), ...]
    """
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(18, 10))

    n = max(1, len(curves))
    cmap = plt.cm.hsv
    for i, (traj_id, xs, ys, is_selected) in enumerate(curves):
        if not xs or not ys:
            continue
        color = cmap(float(i) / float(n))
        alpha = 0.85 if is_selected else 0.25
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=alpha)
        try:
            peak_i = int(max(range(len(ys)), key=lambda k: ys[k]))
            ax.text(
                xs[peak_i],
                ys[peak_i],
                str(int(traj_id)),
                fontsize=6,
                color=color,
                alpha=min(1.0, alpha + 0.15),
                ha="left",
                va="bottom",
                clip_on=True,
            )
        except Exception:
            pass

    ax.set_title(title)
    ax.set_xlabel("frame (relative to trajectory start)")
    ax.set_ylabel("sum of pixel values in crop (RGB sum)")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format="svg", bbox_inches="tight")
    plt.close(fig)


def _moving_average(vals: List[int], win: int) -> List[float]:
    if win <= 1 or len(vals) <= 2:
        return [float(v) for v in vals]
    w = int(win)
    w = max(1, w)
    half = w // 2
    out: List[float] = []
    for i in range(len(vals)):
        a = max(0, i - half)
        b = min(len(vals), i + half + 1)
        seg = vals[a:b]
        out.append(float(sum(seg)) / float(len(seg)))
    return out


def _is_hill_curve(ys: List[int]) -> bool:
    """Heuristic 'hill' detector: rises then falls with a single main peak."""
    if len(ys) < 3:
        return False

    min_range = int(getattr(params, "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE", 0))
    if (max(ys) - min(ys)) < min_range:
        return False

    win = int(getattr(params, "STAGE3_1_HILL_SMOOTH_WINDOW", 3))
    y = _moving_average(ys, win)

    peak_i = int(max(range(len(y)), key=lambda i: y[i]))
    n = len(y)
    pmin = float(getattr(params, "STAGE3_1_HILL_PEAK_POS_MIN_FRAC", 0.10))
    pmax = float(getattr(params, "STAGE3_1_HILL_PEAK_POS_MAX_FRAC", 0.90))
    if peak_i <= int(math.floor(pmin * (n - 1))) or peak_i >= int(math.ceil(pmax * (n - 1))):
        return False

    diffs = [y[i + 1] - y[i] for i in range(n - 1)]
    # Reduce noise: treat tiny diffs as zero.
    eps = 1e-6
    signs = []
    for d in diffs:
        if abs(d) <= eps:
            signs.append(0)
        elif d > 0:
            signs.append(1)
        else:
            signs.append(-1)

    before = [s for s in signs[:peak_i] if s != 0]
    after = [s for s in signs[peak_i:] if s != 0]
    if not before or not after:
        return False

    min_up = int(getattr(params, "STAGE3_1_HILL_MIN_UP_STEPS", 2))
    min_down = int(getattr(params, "STAGE3_1_HILL_MIN_DOWN_STEPS", 2))
    up_steps = sum(1 for s in before if s > 0)
    down_steps = sum(1 for s in after if s < 0)
    if up_steps < min_up or down_steps < min_down:
        return False

    frac = float(getattr(params, "STAGE3_1_HILL_MIN_MONOTONIC_FRAC", 0.60))
    if (up_steps / max(1, len(before))) < frac:
        return False
    if (down_steps / max(1, len(after))) < frac:
        return False

    return True


def run_for_video(video_path: Path) -> Path:
    assert video_path.exists(), f"Video not found: {video_path}"
    stem = video_path.stem

    in_csv, rows, dets = _load_stage3_csv(stem)
    out_csv = in_csv.parent / f"{stem}_patches_motion.csv"  # selected only
    out_all_csv = in_csv.parent / f"{stem}_patches_motion_all.csv"

    fieldnames_base = list(rows[0].keys()) if rows else [
        "frame_idx",
        "video_name",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "det_id",
    ]
    extras = ["traj_id", "traj_size", "traj_motion_xy", "traj_intensity_range", "traj_is_selected"]
    fieldnames = list(fieldnames_base)
    for e in extras:
        if e not in fieldnames:
            fieldnames.append(e)

    if not dets:
        for p in [out_csv, out_all_csv]:
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
        print(f"Stage3.1 NOTE: No detections in {in_csv.name}; wrote empty → {out_csv.name}, {out_all_csv.name}")
        return out_csv

    link_radius_px = float(getattr(params, "STAGE3_1_LINK_RADIUS_PX", 12.0))
    max_frame_gap = int(getattr(params, "STAGE3_1_MAX_FRAME_GAP", 3))
    time_scale = float(getattr(params, "STAGE3_1_TIME_SCALE", 1.0))

    min_pts = int(getattr(params, "STAGE3_1_MIN_TRACK_POINTS", 3))
    min_range = int(getattr(params, "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_MIN_RANGE", 0))
    require_hill = bool(getattr(params, "STAGE3_1_HIGHVAR_REQUIRE_HILL_SHAPE", True))

    export_traj_crops = bool(getattr(params, "STAGE3_1_EXPORT_TRAJECTORY_CROPS", False))
    plot_intensity = bool(getattr(params, "STAGE3_1_PLOT_TRAJECTORY_INTENSITY", False))
    export_highvar = bool(getattr(params, "STAGE3_1_EXPORT_HIGHVAR_TRAJECTORY_CROPS", False))
    plot_highvar = bool(getattr(params, "STAGE3_1_PLOT_TRAJECTORY_INTENSITY_HIGHVAR", False))
    render_highvar_video = bool(getattr(params, "STAGE3_1_RENDER_HIGHVAR_VIDEO", False))

    tracks = _link_trajectories(
        _group_by_frame(dets),
        link_radius_px=link_radius_px,
        max_frame_gap=max_frame_gap,
        time_scale=time_scale,
    )

    pos_dir = in_csv.parent / "crops" / "positives"
    crop_index = _index_stage3_crops(pos_dir)

    traj_root = in_csv.parent / str(getattr(params, "STAGE3_1_TRAJECTORY_CROPS_DIRNAME", "stage3_1_trajectory_crops"))
    if export_traj_crops:
        traj_root.mkdir(parents=True, exist_ok=True)

    highvar_root = in_csv.parent / str(
        getattr(params, "STAGE3_1_HIGHVAR_TRAJECTORY_CROPS_DIRNAME", "stage3_1_highvar_trajectory_crops")
    )
    if export_highvar:
        _safe_rmtree(highvar_root)
        highvar_root.mkdir(parents=True, exist_ok=True)

    det_meta: Dict[int, Tuple[int, int, float, int, int]] = {}  # row_idx -> (traj_id,size,motion,range,selected)
    curves_for_plot: List[Tuple[int, List[int], List[int], int]] = []

    selected_tracks = 0
    selected_dets = 0
    missing_crops = 0
    bad_images = 0
    copy_fail = 0

    for tr in tracks:
        dets_sorted = sorted(tr.dets, key=lambda d: (int(d.t), int(d.idx)))
        if not dets_sorted:
            continue
        t0 = int(dets_sorted[0].t)
        t1 = int(dets_sorted[-1].t)
        motion_xy = float(_track_motion_xy(tr))

        entries: List[dict] = []
        xs_rel: List[int] = []
        ys_sum: List[int] = []
        copy_items: List[tuple[Path, str]] = []

        for turn, d in enumerate(dets_sorted):
            row = rows[d.idx]
            try:
                frame_idx = int(row.get("frame_idx") or row.get("frame_number") or 0)
                x = int(float(row["x"]))
                y = int(float(row["y"]))
                ww = int(float(row.get("w", getattr(params, "PATCH_SIZE_PX", 10))))
                hh = int(float(row.get("h", getattr(params, "PATCH_SIZE_PX", 10))))
                conf = float(row.get("conf", 0.0))
                det_id = int(float(row.get("det_id", 0)))
            except Exception:
                continue

            key = (frame_idx, x, y, ww, hh)
            src = crop_index.get(key)
            dst_name = f"{turn:03d}_t{frame_idx:06d}_x{x}_y{y}.png"

            crop_sum: int | None = None
            if src is not None and src.exists():
                try:
                    s = _sum_png_pixels(src)
                    crop_sum = int(s)
                    xs_rel.append(int(frame_idx - t0))
                    ys_sum.append(int(s))
                    copy_items.append((src, dst_name))
                except Exception:
                    bad_images += 1
            else:
                missing_crops += 1

            entries.append(
                {
                    "turn": int(turn),
                    "frame_idx": int(frame_idx),
                    "x": int(x),
                    "y": int(y),
                    "w": int(ww),
                    "h": int(hh),
                    "conf": float(conf),
                    "det_id": int(det_id),
                    "crop_src": src,
                    "crop_dst": dst_name,
                    "crop_sum": crop_sum,
                }
            )

        intensity_range = int(max(ys_sum) - min(ys_sum)) if ys_sum else 0
        hill_ok = (not require_hill) or _is_hill_curve(ys_sum)
        is_selected = int((len(ys_sum) >= min_pts) and (intensity_range >= min_range) and hill_ok)

        if plot_intensity and xs_rel and ys_sum:
            curves_for_plot.append((int(tr.id), xs_rel, ys_sum, is_selected))

        if is_selected:
            selected_tracks += 1
            selected_dets += len(entries)

        # Fill per-detection meta for output CSVs.
        for d in tr.dets:
            det_meta[d.idx] = (int(tr.id), int(len(tr.dets)), float(motion_xy), int(intensity_range), int(is_selected))

        status = "ACCEPT" if is_selected else "REJECT"

        # Export per-trajectory folder (all trajectories).
        if export_traj_crops:
            folder_name = (
                f"{int(tr.id):05d}_{status}_n{int(len(tr.dets)):03d}"
                f"_t{int(t0):06d}-{int(t1):06d}_m{float(motion_xy):.2f}"
                f"_range{intensity_range}"
            )
            out_dir = traj_root / folder_name
            _safe_rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            traj_csv = out_dir / "trajectory.csv"
            with traj_csv.open("w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "turn",
                        "frame_idx",
                        "x",
                        "y",
                        "w",
                        "h",
                        "conf",
                        "det_id",
                        "traj_id",
                        "traj_size",
                        "traj_motion_xy",
                        "traj_intensity_range",
                        "traj_is_selected",
                        "crop_file",
                        "crop_sum",
                    ],
                )
                w.writeheader()
                for e in entries:
                    crop_file = ""
                    src = e["crop_src"]
                    if src is not None and isinstance(src, Path) and src.exists():
                        try:
                            shutil.copy2(src, out_dir / e["crop_dst"])
                            crop_file = str(e["crop_dst"])
                        except Exception:
                            copy_fail += 1
                    w.writerow(
                        {
                            "turn": e["turn"],
                            "frame_idx": e["frame_idx"],
                            "x": e["x"],
                            "y": e["y"],
                            "w": e["w"],
                            "h": e["h"],
                            "conf": e["conf"],
                            "det_id": e["det_id"],
                            "traj_id": int(tr.id),
                            "traj_size": int(len(tr.dets)),
                            "traj_motion_xy": float(motion_xy),
                            "traj_intensity_range": int(intensity_range),
                            "traj_is_selected": int(is_selected),
                            "crop_file": crop_file,
                            "crop_sum": e["crop_sum"] if e["crop_sum"] is not None else "",
                        }
                    )

        # Export selected (high-var hill) trajectories into a separate root folder.
        if export_highvar and is_selected and highvar_root is not None:
            hv_folder = (
                f"{int(tr.id):05d}_{status}_n{int(len(tr.dets)):03d}"
                f"_t{int(t0):06d}-{int(t1):06d}_m{float(motion_xy):.2f}"
                f"_range{intensity_range}"
            )
            hv_dir = highvar_root / hv_folder
            _safe_rmtree(hv_dir)
            hv_dir.mkdir(parents=True, exist_ok=True)
            for src, dst_name in copy_items:
                try:
                    shutil.copy2(src, hv_dir / dst_name)
                except Exception:
                    copy_fail += 1

    # Write output CSVs.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_all_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows):
            meta = det_meta.get(i)
            if meta is None:
                continue
            traj_id, traj_size, motion_xy, intensity_range, is_selected = meta
            out_row = dict(row)
            out_row["traj_id"] = int(traj_id)
            out_row["traj_size"] = int(traj_size)
            out_row["traj_motion_xy"] = float(motion_xy)
            out_row["traj_intensity_range"] = int(intensity_range)
            out_row["traj_is_selected"] = int(is_selected)
            w.writerow(out_row)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows):
            meta = det_meta.get(i)
            if meta is None:
                continue
            traj_id, traj_size, motion_xy, intensity_range, is_selected = meta
            if not is_selected:
                continue
            out_row = dict(row)
            out_row["traj_id"] = int(traj_id)
            out_row["traj_size"] = int(traj_size)
            out_row["traj_motion_xy"] = float(motion_xy)
            out_row["traj_intensity_range"] = int(intensity_range)
            out_row["traj_is_selected"] = int(is_selected)
            w.writerow(out_row)

    print(
        "Stage3.1 Trajectories + intensity selection:",
        f"dets_in={len(dets)}",
        f"tracks={len(tracks)}",
        f"selected_tracks={selected_tracks}",
        f"selected_dets={selected_dets}",
        f"(min_pts={min_pts}, min_range={min_range}, require_hill={require_hill}, link_r={link_radius_px}, gap={max_frame_gap}, t_scale={time_scale})",
    )
    if missing_crops:
        print(f"Stage3.1 NOTE: missing_crops={missing_crops} (expected Stage3 crops in {pos_dir})")
    if bad_images:
        print(f"Stage3.1 NOTE: bad_images={bad_images}")
    if copy_fail:
        print(f"Stage3.1 NOTE: copy_failures={copy_fail}")
    print(f"Stage3.1 Wrote selected CSV → {out_csv}")
    print(f"Stage3.1 Wrote all CSV → {out_all_csv}")

    # Plot intensity curves SVG(s).
    if plot_intensity:
        svg_name = str(
            getattr(params, "STAGE3_1_TRAJECTORY_INTENSITY_SVG_NAME", "stage3_1_trajectory_intensity_curves.svg")
        )
        out_svg = in_csv.parent / svg_name
        title = f"{stem} – trajectory intensity curves (n={len(curves_for_plot)})"
        try:
            _write_trajectory_intensity_svg(out_path=out_svg, title=title, curves=curves_for_plot)
            print(f"Stage3.1 Wrote intensity curves SVG → {out_svg}")
        except Exception as e:
            print(f"Stage3.1 Warning: failed to write intensity curves SVG: {e}")

        if plot_highvar:
            curves_hi = [(tid, xs, ys, sel) for (tid, xs, ys, sel) in curves_for_plot if sel]
            svg_name_hi = str(
                getattr(
                    params,
                    "STAGE3_1_TRAJECTORY_INTENSITY_HIGHVAR_SVG_NAME",
                    "stage3_1_trajectory_intensity_curves_highvar.svg",
                )
            )
            out_svg_hi = in_csv.parent / svg_name_hi
            title_hi = f"{stem} – SELECTED (hill) intensity curves (n={len(curves_hi)})"
            try:
                _write_trajectory_intensity_svg(out_path=out_svg_hi, title=title_hi, curves=curves_hi)
                print(f"Stage3.1 Wrote selected intensity curves SVG → {out_svg_hi}")
            except Exception as e:
                print(f"Stage3.1 Warning: failed to write selected intensity curves SVG: {e}")

    # Render a full video showing selected vs rejected (red vs blue).
    if render_highvar_video:
        boxes_by_t = _collect_stage31_boxes_from_csv(out_all_csv)
        if not boxes_by_t:
            print(f"Stage3.1 NOTE: No Stage3.1 CSV boxes found to render video from {out_all_csv}")
        else:
            out_name = str(getattr(params, "STAGE3_1_HIGHVAR_VIDEO_NAME", "stage3_1_highvar_trajectories.mp4"))
            out_video = in_csv.parent / out_name
            try:
                _render_boxes_video(video_path=video_path, out_path=out_video, boxes_by_t=boxes_by_t)
                print(f"Stage3.1 Wrote selected-vs-rejected video → {out_video}")
            except Exception as e:
                print(f"Stage3.1 Warning: failed to render selected-vs-rejected video: {e}")

    return out_csv


__all__ = ["run_for_video"]
