#!/usr/bin/env python3
from __future__ import annotations

"""
Day-time pipeline Stage 1

Threshold → Background Subtraction → Long-Exposure (OR) trails
+ Connected Components visuals (all; area-filtered view)
+ Telemetry CSV for ML: per-pixel (x,y,t,global_component_id)
+ Per-component 10×10 patches from the original video using (x,y,t) telemetry.

Outputs are organized under: params.STAGE1_DIR/<video_stem>/
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

import params
from utils import open_video, make_writer, center_crop_with_pad, progress


# ───────────── Helpers ─────────────

def _ensure_dirs(out_root: Path) -> dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    d = {
        "root": out_root,
        "or_chunks": out_root / "or_chunks",
        "cc_all_chunks": out_root / "cc_all_chunks",
        "cc_filtered_chunks": out_root / "cc_filtered_chunks",
        "cc_crops": out_root / "cc_crops",
        "overlays": out_root / "overlays",
        "component_patches": out_root / "component_patches",
    }
    # Always create root and or_chunks (used internally to compute CC per chunk)
    d["root"].mkdir(parents=True, exist_ok=True)
    d["or_chunks"].mkdir(parents=True, exist_ok=True)
    # Create debug/visualization dirs only if extras are enabled
    if bool(getattr(params, 'SAVE_EXTRAS', True)):
        for key in ("cc_all_chunks", "cc_filtered_chunks", "cc_crops", "overlays", "component_patches"):
            d[key].mkdir(parents=True, exist_ok=True)
    return d


def _chunk_tag(or_path: Path) -> str:
    stem = or_path.stem
    if "_chunk_" in stem:
        return stem.split("_chunk_", 1)[1]
    return stem


def _idmap_path_for_chunk(or_path: Path) -> Path:
    return or_path.with_suffix("").with_name(or_path.stem + "_idmap.npy")


def _tmin_path_for_chunk(or_path: Path) -> Path:
    return or_path.with_suffix("").with_name(or_path.stem + "_tmin.npy")


# ───────────── Stage 1.1: Threshold ─────────────

def _write_threshold_video(inp: Path, out: Path, max_frames: int | None, fps_override: float | None) -> Tuple[int, int, float, int]:
    cap, W, H, fps_src, total = open_video(inp)
    fps = float(fps_override or fps_src)
    wr = make_writer(out, W, H, fps, codec=params.RENDER_CODEC, is_color=True)

    hard_cap = max_frames if max_frames is not None else total
    total_iter = min(total, hard_cap) if hard_cap is not None and total > 0 else (hard_cap or total or 0)

    i = 0
    try:
        while True:
            if hard_cap is not None and i >= hard_cap:
                break
            ok, bgr = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, int(params.THRESHOLD_8BIT), 255, cv2.THRESH_BINARY)
            out_frame = cv2.merge([mask, mask, mask])
            wr.write(out_frame)
            i += 1
            if i % 20 == 0:
                progress(i, total_iter or i, "stage1: threshold")
        progress(i, i or 1, "stage1: threshold done")
    finally:
        cap.release()
        wr.release()

    return W, H, fps, i


# ───────────── Stage 1.2: BG-sub (MOG2) ─────────────

def _write_bgsub_video(inp: Path, out: Path, force_w: int | None, force_h: int | None, fps_hint: float | None) -> int:
    cap, W0, H0, fps_src, total = open_video(inp)
    W = int(force_w or W0)
    H = int(force_h or H0)
    fps = float(fps_hint or fps_src)

    bgs = cv2.createBackgroundSubtractorMOG2(
        detectShadows=bool(params.BGS_DETECT_SHADOWS),
        history=int(params.BGS_HISTORY),
    )

    wr = make_writer(out, W, H, fps, codec=params.RENDER_CODEC, is_color=False)

    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
            fg = bgs.apply(frame, learningRate=float(params.BGS_LEARNING_RATE))
            if int(params.LONG_EXP_DILATE_ITERS) > 0:
                k = int(params.LONG_EXP_DILATE_KERNEL)
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=int(params.LONG_EXP_DILATE_ITERS))
            wr.write(fg)
            i += 1
            if i % 20 == 0:
                progress(i, total or i, "stage1: bgsub")
        progress(i, i or 1, "stage1: bgsub done")
    finally:
        cap.release()
        wr.release()

    return i


# ───────────── Stage 1.3: Long-exposure OR (chunked) + telemetry ─────────────

def _render_or_and_telemetry(bgsub_video: Path, dirs: dict[str, Path], stem: str) -> List[Path]:
    cap0, W, H, fps_src, total_est = open_video(bgsub_video)
    cap0.release()

    csv_path = dirs["root"] / f"{stem}_pixels_xy_t.csv"
    if bool(getattr(params, "STAGE1_WRITE_PER_FRAME_CSV", True)):
        with csv_path.open("w") as f:
            f.write("x,y,t,global_component_id\n")

    chunk_paths: list[Path] = []
    cid_offset = 0

    start_global = 0
    first_effective_start = max(int(params.LONG_EXP_START_FRAME), 0)

    while start_global < total_est:
        end_global = min(start_global + int(params.LONG_EXP_CHUNK_SIZE), total_est)
        start_eff = max(start_global, first_effective_start)

        trails = np.zeros((H, W), dtype=np.bool_)
        first_map = np.full((H, W), -1, dtype=np.int32)

        cap = cv2.VideoCapture(str(bgsub_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_eff))
        t = int(start_eff)

        kernel = None
        if int(params.LONG_EXP_DILATE_ITERS) > 0:
            k = int(params.LONG_EXP_DILATE_KERNEL)
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        total_chunk_frames = max(0, end_global - start_eff)
        seen = 0

        while t < end_global:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.ndim == 3:
                fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                fg = frame

            if int(params.LONG_EXP_BLUR_KSIZE) and int(params.LONG_EXP_BLUR_KSIZE) > 1:
                k = int(params.LONG_EXP_BLUR_KSIZE)
                if k % 2 == 0:
                    k += 1
                fg = cv2.GaussianBlur(fg, (k, k), 0)

            mask = fg >= int(params.FG_MASK_THRESHOLD)
            if kernel is not None and int(params.LONG_EXP_DILATE_ITERS) > 0:
                mask8 = (mask.astype(np.uint8) * 255)
                mask8 = cv2.dilate(mask8, kernel, iterations=int(params.LONG_EXP_DILATE_ITERS))
                mask = mask8 > 0

            trails |= mask
            new_on = mask & (first_map < 0)
            first_map[new_on] = t

            t += 1
            seen += 1
            if seen % 20 == 0:
                progress(seen, total_chunk_frames or seen, "stage1: OR chunk")
        progress(seen, seen or 1, "stage1: OR chunk done")
        cap.release()

        # Save chunk trails image
        out_chunk = dirs["or_chunks"] / f"{stem}_or_chunk_{start_global:06d}_{end_global-1:06d}.png"
        trails_img = (trails.astype(np.uint8) * 255)
        cv2.imwrite(str(out_chunk), trails_img)
        chunk_paths.append(out_chunk)

        # Save overlay (optional; only when extras enabled)
        if bool(getattr(params, 'SAVE_EXTRAS', True)) and bool(getattr(params, "STAGE1_SAVE_OVERLAY", False)):
            cap_in = cv2.VideoCapture(str(bgsub_video))
            cap_in.set(cv2.CAP_PROP_POS_FRAMES, float(start_eff))
            ok0, base = cap_in.read()
            cap_in.release()
            if ok0:
                rgb = cv2.merge([trails_img, trails_img, trails_img])
                overlay_img = cv2.addWeighted(base, 1.0, rgb, 0.7, 0.0)
                out_overlay = dirs["overlays"] / f"{stem}_overlay_chunk_{start_global:06d}_{end_global-1:06d}.png"
                cv2.imwrite(str(out_overlay), overlay_img)

        # Connected components for telemetry CSV on this chunk
        mask_u8 = (trails_img > 0).astype(np.uint8) * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        ys, xs = np.nonzero(mask_u8)
        local_ids = labels[ys, xs].astype(np.int32)
        m_fg = local_ids >= 1
        xs = xs[m_fg]; ys = ys[m_fg]; local_ids = local_ids[m_fg]
        ts = first_map[ys, xs].astype(np.int32)

        # Save idmap + tmin sidecars for this chunk
        idmap = np.arange(num, dtype=np.int64)
        idmap[1:] = idmap[1:] + int(cid_offset)
        np.save(str(_idmap_path_for_chunk(out_chunk)), idmap)
        tmin_local = np.full(num, -1, dtype=np.int64)
        if local_ids.size > 0:
            uniq = np.unique(local_ids)
            for li in uniq:
                tmin_local[int(li)] = int(ts[local_ids == li].min())
        np.save(str(_tmin_path_for_chunk(out_chunk)), tmin_local)

        if bool(getattr(params, "STAGE1_WRITE_PER_FRAME_CSV", True)) and xs.size > 0:
            data = np.column_stack([xs, ys, ts, local_ids + int(cid_offset)])
            with csv_path.open("ab") as fb:
                np.savetxt(fb, data, fmt="%d,%d,%d,%d")

        cid_offset += (num - 1)
        start_global = end_global
        first_effective_start = 0

    return chunk_paths


# ───────────── Stage 1.4: CC visuals per chunk + crops and kept comps ─────────────

def _cc_visuals_and_select(or_img: Path, dirs: dict[str, Path]) -> List[dict]:
    img = cv2.imread(str(or_img), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Cannot read OR image: {or_img}"
    H, W = img.shape[:2]

    mask_u8 = (img > 0).astype(np.uint8) * 255
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    # deterministically color all comps
    rng = np.random.default_rng(12345)
    colors = np.zeros((num, 3), dtype=np.uint8)
    for i in range(1, num):
        colors[i] = rng.integers(60, 255, size=3, endpoint=True, dtype=np.uint8)
    color_all = colors[labels]

    if bool(getattr(params, 'SAVE_EXTRAS', True)):
        all_vis = dirs["cc_all_chunks"] / f"long_exp_components_all_chunk_{_chunk_tag(or_img)}.png"
        cv2.imwrite(str(all_vis), color_all)

    min_a = int(max(0, params.CC_MIN_AREA))
    max_a = int(params.CC_MAX_AREA) if int(params.CC_MAX_AREA) > 0 else None
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_a:
            continue
        if max_a is not None and area > max_a:
            continue
        keep[i] = True

    color_filtered = np.zeros_like(color_all)
    kmask = keep[labels]
    color_filtered[kmask] = color_all[kmask]

    # load idmap + tmin
    idmap = None
    tmin_local = None
    try:
        idmap = np.load(str(_idmap_path_for_chunk(or_img)))
    except Exception:
        idmap = None
    try:
        tmin_local = np.load(str(_tmin_path_for_chunk(or_img)))
    except Exception:
        tmin_local = None

    # crops dir (extras only)
    save_extras = bool(getattr(params, 'SAVE_EXTRAS', True))
    crops_dir = dirs["cc_crops"] / f"chunk_{_chunk_tag(or_img)}"
    if save_extras:
        crops_dir.mkdir(parents=True, exist_ok=True)
    kept_rows: List[dict] = []

    for i in range(1, num):
        if not keep[i]:
            continue
        x, y, w, h, area = [int(v) for v in stats[i]]
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        crop = color_all[y0:y1, x0:x1].copy()
        gid = int(idmap[i]) if idmap is not None else i
        tmin = int(tmin_local[i]) if (tmin_local is not None and i < len(tmin_local) and tmin_local[i] >= 0) else -1
        if save_extras:
            out_name = f"gid_{gid:06d}_x{x0}_y{y0}_w{(x1-x0)}_h{(y1-y0)}_t{tmin}.png"
            cv2.imwrite(str(crops_dir / out_name), crop)
        kept_rows.append({
            "global_id": gid,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "first_t": tmin,
            "area": area,
        })

    # draw boxes+ids
    if True:
        for i in range(1, num):
            if not keep[i]:
                continue
            x, y, w, h, area = [int(v) for v in stats[i]]
            cv2.rectangle(color_filtered, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 1, cv2.LINE_AA)
            gid = int(idmap[i]) if idmap is not None else i
            label_text = str(gid)
            ty = y - 4 if y - 4 >= 10 else y + 14
            cv2.putText(color_filtered, label_text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(color_filtered, label_text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if save_extras:
        filt_vis = dirs["cc_filtered_chunks"] / f"long_exp_components_filtered_chunk_{_chunk_tag(or_img)}.png"
        cv2.imwrite(str(filt_vis), color_filtered)

    return kept_rows


# ───────────── Stage 1.5: Per-component patches (PNG) from original video ─────────────

def _make_component_patches(csv_path: Path, input_video_path: Path, out_dir: Path, patch_size: int = 10) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        return 0
    data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1, dtype=np.int64)
    if data.ndim == 1 and data.size == 4:
        data = data.reshape(1, 4)
    if data.size == 0:
        return 0
    xs = data[:, 0]; ys = data[:, 1]; ts = data[:, 2]; cids = data[:, 3]
    all_cids = np.unique(cids)
    for cid in all_cids:
        (out_dir / f"{int(cid):05d}").mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    centers_by_t = defaultdict(list)
    for x, y, t, cid in zip(xs, ys, ts, cids):
        centers_by_t[int(t)].append((int(cid), int(x), int(y)))

    times_sorted = sorted(centers_by_t.keys())
    if not times_sorted:
        return 0
    min_t, max_t = times_sorted[0], times_sorted[-1]

    cap, W, H, fps, total = open_video(input_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(min_t))
    made = 0
    cur_t = int(min_t)
    try:
        while cur_t <= max_t:
            ok, frame = cap.read()
            if not ok:
                break
            if cur_t in centers_by_t:
                for (cid, cx, cy) in centers_by_t[cur_t]:
                    patch, _, _ = center_crop_with_pad(frame, cx, cy, int(patch_size), int(patch_size))
                    out_path = out_dir / f"{int(cid):05d}" / f"t_{int(cur_t):06d}.png"
                    cv2.imwrite(str(out_path), patch)
                    made += 1
            cur_t += 1
            if (cur_t - min_t) % 20 == 0:
                progress(cur_t - min_t, max(1, max_t - min_t + 1), "stage1: patches")
        progress(max_t - min_t + 1, max_t - min_t + 1, "stage1: patches done")
    finally:
        cap.release()
    return made


# ───────────── Stage 1 extra: Render overlay video with (x,y,t) boxes ─────────────

def _render_xyts_overlay_video(video_path: Path, csv_path: Path, out_path: Path, *, box_size: int = 10,
                               color: tuple[int, int, int] = (0, 0, 255), thickness: int = 1) -> Path:
    """Render an overlay on the original video drawing a box of size box_size×box_size
    centered at (x,y) for each row at time t from the Stage 1 telemetry CSV (x,y,t,global_component_id).
    """
    boxes_by_t: dict[int, list[tuple[float, float]]] = {}
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1, dtype=np.float32)
            if data.ndim == 1 and data.size == 4:
                data = data.reshape(1, 4)
            if data.size > 0:
                xs = data[:, 0].astype(np.float32)
                ys = data[:, 1].astype(np.float32)
                ts = data[:, 2].astype(np.int64)
                for x, y, t in zip(xs, ys, ts):
                    boxes_by_t.setdefault(int(t), []).append((float(x), float(y)))
        except Exception as e:
            print(f"Stage1  WARN: could not read telemetry CSV {csv_path.name}: {e}")

    cap, W, H, fps_src, total = open_video(video_path)
    max_frames = int(params.MAX_FRAMES) if (params.MAX_FRAMES is not None) else total
    fps = float(params.RENDER_FPS_HINT or fps_src)
    writer = make_writer(out_path, W, H, fps, codec=params.RENDER_CODEC, is_color=True)

    t = 0
    try:
        while t < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if t in boxes_by_t:
                for (cx, cy) in boxes_by_t[t]:
                    w = int(box_size)
                    h = int(box_size)
                    x0 = int(round(cx - w / 2.0))
                    y0 = int(round(cy - h / 2.0))
                    x1 = x0 + w
                    y1 = y0 + h
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, int(thickness), cv2.LINE_AA)
            writer.write(frame)
            t += 1
            if t % 50 == 0:
                progress(t, max_frames or t, "stage1: xyts overlay")
        progress(t, max_frames or t or 1, "stage1: xyts overlay done")
    finally:
        cap.release()
        writer.release()
    print(f"Stage1  Wrote xyts overlay video → {out_path}")
    return out_path


# ───────────── Public API ─────────────

def run_for_video(video_path: Path) -> Path:
    stem = video_path.stem
    out_dirs = _ensure_dirs(params.STAGE1_DIR / stem)

    # Stage 1.1: threshold video (written as color for easy playback)
    thr_path = out_dirs["root"] / f"{stem}_threshold.mp4"
    W, H, fps, n1 = _write_threshold_video(
        video_path,
        thr_path,
        int(params.MAX_FRAMES) if params.MAX_FRAMES is not None else None,
        None,
    )

    # Stage 1.2: background subtraction on the thresholded video (single-channel)
    bg_path = out_dirs["root"] / f"{stem}_threshold_bgsub.mp4"
    n2 = _write_bgsub_video(thr_path, bg_path, W, H, fps)

    # Stage 1.3: long-exposure OR images (chunked) + telemetry CSV
    chunk_imgs = _render_or_and_telemetry(bg_path, out_dirs, stem)

    # Stage 1.4: CC visuals for each chunk + collect kept components into trajectories
    kept_all: List[dict] = []
    for or_img in chunk_imgs:
        kept = _cc_visuals_and_select(or_img, out_dirs)
        kept_all.extend(kept)

    # Write trajectories CSV (one row per kept component)
    traj_csv = out_dirs["root"] / f"{stem}_trajectories.csv"
    import csv as _csv
    with traj_csv.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["global_id", "x", "y", "w", "h", "first_t", "area"])  # area is extra; stage2 ignores it
        for row in kept_all:
            w.writerow([
                int(row["global_id"]), int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]), int(row["first_t"]), int(row["area"]),
            ])

    # Stage 1.5: optional patches (10×10 PNGs) saved per component id and time
    if bool(getattr(params, 'SAVE_EXTRAS', True)) and bool(getattr(params, "STAGE1_SAVE_PATCHES", True)):
        telem_csv = out_dirs["root"] / f"{stem}_pixels_xy_t.csv"
        _make_component_patches(telem_csv, video_path, out_dirs["component_patches"], patch_size=int(params.PATCH_SIZE_PX))

    # Extra: render an overlay video (only if extras enabled)
    if bool(getattr(params, 'SAVE_EXTRAS', True)):
        try:
            telem_csv = out_dirs["root"] / f"{stem}_pixels_xy_t.csv"
            out_overlay_vid = out_dirs["root"] / f"{stem}_xyts_overlay.mp4"
            _render_xyts_overlay_video(
                video_path,
                telem_csv,
                out_overlay_vid,
                box_size=int(params.PATCH_SIZE_PX),
                thickness=int(getattr(params, 'RENDER_BOX_THICKNESS', 1)),
            )
        except Exception as e:
            print(f"Stage1  WARN: failed to render xyts overlay: {e}")

    # Stage 1 stats
    telem_csv_path = out_dirs['root'] / f"{stem}_pixels_xy_t.csv"
    telem_rows = 0
    try:
        with open(telem_csv_path, 'r') as _f:
            telem_rows = max(0, sum(1 for _ in _f) - 1)
    except Exception:
        telem_rows = 0
    try:
        import numpy as _np
        areas = _np.asarray([int(r['area']) for r in kept_all], dtype=_np.int64)
        if areas.size:
            area_stats = f"area_px: avg={areas.mean():.1f} p50={_np.percentile(areas,50):.1f} p90={_np.percentile(areas,90):.1f} max={areas.max()}"
        else:
            area_stats = "area_px: n=0"
    except Exception:
        area_stats = "area_px: n/a"
    print("Stage1  Summary:")
    print(f"  Threshold frames: {n1}")
    print(f"  BG-sub frames:   {n2}")
    print(f"  OR chunks:       {len(chunk_imgs)} ({out_dirs['or_chunks']})")
    print(f"  Telemetry CSV:   {telem_csv_path} (rows={telem_rows})")
    print(f"  Kept components: {len(kept_all)} ({area_stats})")
    if bool(getattr(params, 'SAVE_EXTRAS', True)):
        print(f"  CC ALL dir:      {out_dirs['cc_all_chunks']}")
        print(f"  CC FILTERED dir: {out_dirs['cc_filtered_chunks']}")
        print(f"  CC CROPS dir:    {out_dirs['cc_crops']}")
    print(f"  Trajectories:    {traj_csv}")
    # Cleanup extras if disabled: remove videos and debug dirs
    if not bool(getattr(params, 'SAVE_EXTRAS', True)):
        import shutil
        def _rm(p: Path):
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
        # Remove directories with extra artifacts
        for key in ("cc_all_chunks", "cc_filtered_chunks", "cc_crops", "overlays", "component_patches"):
            _rm(out_dirs[key])
        # Remove intermediate videos and OR chunks images
        _rm(out_dirs['root'] / f"{stem}_threshold.mp4")
        _rm(out_dirs['root'] / f"{stem}_threshold_bgsub.mp4")
        _rm(out_dirs['or_chunks'])
    return traj_csv


__all__ = ["run_for_video"]
