#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import params
from utils import progress, center_crop_with_pad, open_video, make_writer


def _chunk_or_trails(video: Path, start: int, end: int, W: int, H: int):
    """Long-exposure OR of bright pixels in [start, end).
    Returns (trails_img_u8, first_map_int32) where first_map[y,x] is the
    first frame index that pixel lit up, or -1 if never.
    """
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))

    trails = np.zeros((H, W), dtype=np.bool_)
    first_map = np.full((H, W), -1, dtype=np.int32)

    t = int(start)
    total = max(0, int(end - start))
    seen = 0
    try:
        while t < end:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = gray >= int(params.TRAILS_INTENSITY_THRESHOLD)
            new_on = mask & (first_map < 0)
            first_map[new_on] = t
            trails |= mask

            t += 1
            seen += 1
            if seen % 20 == 0:
                progress(seen, total or seen, f"Stage1 OR {start}-{end-1}")
        progress(seen, seen or 1, f"Stage1 OR {start}-{end-1} done")
    finally:
        cap.release()

    img_u8 = (trails.astype(np.uint8) * 255)
    return img_u8, first_map


def run_for_video(video_path: Path) -> Path:
    """Run Stage 1 for a single video.

    Produces per-chunk OR images, CC visualizations, crops, and a CSV of
    trajectory candidate boxes with an assigned global id and first_t.
    Returns path to the per-video CSV.
    """
    assert video_path.exists(), f"Video not found: {video_path}"
    stem = video_path.stem
    out_root = params.STAGE1_DIR / stem
    or_dir = out_root / "or_chunks"
    cc_all_dir = out_root / "cc_all"
    cc_filt_dir = out_root / "cc_filtered"
    crops_dir = out_root / "crops"
    for d in (or_dir, cc_all_dir, cc_filt_dir, crops_dir):
        d.mkdir(parents=True, exist_ok=True)

    cap0 = cv2.VideoCapture(str(video_path))
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N_total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()
    # Capture for extracting original 10x10 patches at first_t
    cap_orig = cv2.VideoCapture(str(video_path))
    if not cap_orig.isOpened():
        raise RuntimeError(f"Cannot open video for patch extraction: {video_path}")
    # Respect global frame cap if provided
    N = min(N_total, int(params.MAX_FRAMES)) if (params.MAX_FRAMES is not None) else N_total

    csv_path = out_root / f"{stem}_trajectories.csv"
    f_csv = csv_path.open("w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["video", "global_id", "chunk_start", "chunk_end", "x", "y", "w", "h", "area", "first_t"])  # top-left (x,y)

    rng = np.random.default_rng(12345)
    id_offset = 0
    # For rendering: map frame index -> list of (cx, cy, w, h, gid)
    boxes_by_t: dict[int, list[tuple[float, float, int, int, int]]] = {}

    for start in range(0, N, int(params.CHUNK_SIZE)):
        end = min(N, start + int(params.CHUNK_SIZE))
        trails_img, first_map = _chunk_or_trails(video_path, start, end, W, H)

        out_chunk_path = or_dir / f"{stem}_chunk_{start:06d}_{end-1:06d}.png"
        cv2.imwrite(str(out_chunk_path), trails_img)

        mask = trails_img > 0
        mask_u8 = (mask.astype(np.uint8) * 255)
        num, labels, stats, cents = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        colors = np.zeros((num, 3), dtype=np.uint8)
        for i in range(1, num):
            colors[i] = rng.integers(60, 255, size=3, endpoint=True, dtype=np.uint8)
        color_all = colors[labels]
        cv2.imwrite(str(cc_all_dir / f"cc_all_{start:06d}_{end-1:06d}.png"), color_all)

        # Keep ALL components (no area filtering)
        keep = np.ones(num, dtype=bool)
        keep[0] = False  # background label

        keep_mask = keep[labels]
        color_filtered = np.zeros_like(color_all)
        color_filtered[keep_mask] = color_all[keep_mask]

        # Draw thin white boxes + GLOBAL component id label (like reference script)
        for i in range(1, num):
            if not keep[i]:
                continue
            x, y, w, h, area = stats[i]
            # Draw box in white
            cv2.rectangle(color_filtered, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 1, cv2.LINE_AA)
            # Label with GLOBAL id (offsetted by id_offset)
            g_id = int(i + id_offset)
            label_text = str(g_id)
            ty = y - 4 if (y - 4) >= 10 else (y + 14)
            cv2.putText(color_filtered, label_text, (x, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(color_filtered, label_text, (x, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(cc_filt_dir / f"cc_filtered_{start:06d}_{end-1:06d}.png"), color_filtered)

        chunk_tag = f"chunk_{start:06d}_{end-1:06d}"
        (crops_dir / chunk_tag).mkdir(parents=True, exist_ok=True)

        for i in range(1, num):
            if not keep[i]:
                continue
            x, y, w, h, area = map(int, stats[i])
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(W, x + w)
            y1 = min(H, y + h)
            comp_mask = labels == i
            fm_vals = first_map[comp_mask]
            fm_vals = fm_vals[fm_vals >= 0]
            first_t = int(fm_vals.min()) if fm_vals.size else int(start)

            crop_gray = trails_img[y0:y1, x0:x1]
            crop_bgr = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
            g_id = int(i + id_offset)
            out_name = f"gid_{g_id:06d}_x{x0}_y{y0}_w{(x1-x0)}_h{(y1-y0)}_t{first_t}.png"
            cv2.imwrite(str((crops_dir / chunk_tag) / out_name), crop_bgr)

            writer.writerow([stem, g_id, start, end - 1, x0, y0, x1 - x0, y1 - y0, area, first_t])

            # Also save 10x10 patch from the ORIGINAL video at first_t, centered on bbox center
            try:
                # Seek and read the specific frame
                cap_orig.set(cv2.CAP_PROP_POS_FRAMES, float(first_t))
                ok, frame = cap_orig.read()
                if ok and frame is not None:
                    cx = x0 + (x1 - x0) / 2.0
                    cy = y0 + (y1 - y0) / 2.0
                    patch, _, _ = center_crop_with_pad(frame, cx, cy, params.PATCH_SIZE_PX, params.PATCH_SIZE_PX)
                    # per-component subfolder like component_gifs layout
                    gid_dir = (out_root / "patches" / f"{g_id:05d}")
                    gid_dir.mkdir(parents=True, exist_ok=True)
                    patch_name = f"t_{first_t:06d}.png"
                    cv2.imwrite(str(gid_dir / patch_name), patch)
                    # record for overlay rendering later
                    boxes_by_t.setdefault(int(first_t), []).append((cx, cy, int(params.PATCH_SIZE_PX), int(params.PATCH_SIZE_PX), g_id))
            except Exception:
                # best-effort; continue silently on patch failure
                pass

        id_offset += (num - 1)

    f_csv.close()
    cap_orig.release()
    print(f"Stage1  Wrote trajectories CSV → {csv_path}")

    # --- Render overlay video with all 10x10 patches at their first_t ---
    try:
        cap, Wv, Hv, fps, total = open_video(video_path)
        # choose fps
        fps_use = float(params.RENDER_FPS_HINT or fps)
        out_video = out_root / f"{stem}_stage1_patches_overlay.mp4"
        max_frames = min(total, int(params.MAX_FRAMES) if (params.MAX_FRAMES is not None) else total)
        writer = make_writer(out_video, Wv, Hv, fps_use, codec=params.RENDER_CODEC, is_color=True)

        t = 0
        try:
            while t < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if t in boxes_by_t:
                    for (cx, cy, w, h, gid_draw) in boxes_by_t[t]:
                        x0 = int(round(cx - w / 2.0))
                        y0 = int(round(cy - h / 2.0))
                        x1 = x0 + int(w)
                        y1 = y0 + int(h)
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
                        # Label at top-left of 10x10 box
                        ty = y0 - 4 if (y0 - 4) >= 10 else (y0 + 14)
                        label_text = str(int(gid_draw))
                        cv2.putText(frame, label_text, (x0, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, label_text, (x0, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame)
                t += 1
                if t % 50 == 0:
                    progress(t, max_frames or t, "Stage1 overlay render")
            progress(t, max_frames or t or 1, "Stage1 overlay render done")
        finally:
            cap.release()
            writer.release()
        print(f"Stage1  Overlay video → {out_video}")
    except Exception as e:
        print(f"Stage1  Overlay render skipped due to error: {e}")
    return csv_path


__all__ = ["run_for_video"]
