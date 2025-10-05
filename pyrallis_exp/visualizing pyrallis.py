#!/usr/bin/env python3
"""
Threshold → Background Subtraction → Long-Exposure (OR) Trails Image
+ Connected Components visualizations (all; area-filtered view)
+ Telemetry CSV for ML: per-pixel (x,y,t,global_component_id)
+ Stage 5: Per-component 10×10 crops from the original video using (x,y,t) telemetry.

Install: pip install opencv-python numpy imageio
"""

from __future__ import annotations
from pathlib import Path
import sys
import cv2
import numpy as np
import imageio.v2 as imageio

# ───────────── GLOBAL PARAMETERS ─────────────

# I/O (kept exactly as provided)
INPUT_VIDEO_PATH    = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/raw videos/20240606_cam1_GS010064.mp4')
OUTPUT_THRESH_PATH  = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/pinfield videos/100f_20240606_cam1_GS010064_pinfield.mp4')
OUTPUT_BGSUB_PATH   = None  # auto = "<threshold_stem>_bgsub.mp4"

# Long-exposure stills (kept exactly as provided)
LONG_EXP_IMAGE_PATH         = Path("/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/pinfield videos/output_long_exposure.png")
LONG_EXP_OVERLAY_IMAGE_PATH = Path("/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/pinfield videos/output_long_exposure_overlay.png")

# Outputs for Connected Components (visuals)
CC_ALL_VIS_PATH       = LONG_EXP_IMAGE_PATH.with_name("long_exp_components_all.png")
CC_FILTERED_VIS_PATH  = LONG_EXP_IMAGE_PATH.with_name("long_exp_components_filtered.png")  # area-filtered + boxes + ids

# Telemetry CSV (per-pixel)
CC_PIXELS_CSV_PATH    = LONG_EXP_IMAGE_PATH.with_name("long_exp_components_pixels_xy_t.csv")

# ── NEW: Area filter params as globals ───────────────────────────────
CC_MIN_AREA           = 20         # draw boxes+IDs only for components with area >= this
CC_MAX_AREA           = 3000       # 0 = no upper cap; set >0 to limit very large blobs

# ── NEW: GIF generation globals ──────────────────────────────────────
GIF_OUTPUT_DIR        = LONG_EXP_IMAGE_PATH.with_name("component_gifs")
GIF_SIZE              = 10         # crop size (pixels), square (10×10)
GIF_FPS               = 8          # playback fps for GIFs
GIF_LIMIT_COMPONENTS  = 0          # 0 = all; else cap number of components to export

# ── NEW: Long-exposure chunking ──────────────────────────────────────
LONG_EXP_CHUNK_SIZE   = 500        # build one long-exposure image per 500 frames

# ── NEW: Output folders for each artifact type ───────────────────────
OR_CHUNKS_DIR         = LONG_EXP_IMAGE_PATH.parent / "or_chunks"
CC_ALL_CHUNKS_DIR     = LONG_EXP_IMAGE_PATH.parent / "cc_all_chunks"
CC_FILTERED_CHUNKS_DIR= LONG_EXP_IMAGE_PATH.parent / "cc_filtered_chunks"
CC_CROPS_DIR          = LONG_EXP_IMAGE_PATH.parent / "cc_crops"  # per-chunk subfolders

MAX_FRAMES            = 1000       # None = full video
OUTPUT_FPS            = None       # None = use source fps
CODEC                 = "mp4v"

# Stage 1: Threshold
THRESHOLD_8BIT        = 115        # 0..255 grayscale threshold
MODE                  = "mask"     # "mask" or "overlay"
INVERT                = False      # True => white where gray < threshold

# Stage 2: Background subtraction (MOG2)
BGS_DETECT_SHADOWS    = True       # shadows get value 127
BGS_HISTORY           = 1000
BGS_LEARNING_RATE     = -1.0       # -1 => OpenCV decides
MORPH_KERNEL_SIZE     = 3
MORPH_OPEN_ITERS      = 0          # keep 0 to preserve pixel detail

# Stage 3: Long-exposure OR (from Stage-2 video)
LONG_EXP_START_FRAME  = 30         # skip early frames for BG model warm-up
FG_MASK_THRESHOLD     = 200        # treat pixels >= this as foreground (ignore shadows=127)
LONG_EXP_DILATE_ITERS = 1          # 0=off; 1–2 thicken trails slightly
LONG_EXP_DILATE_KERNEL= 3          # odd size (3/5/7)
LONG_EXP_BLUR_KSIZE   = 0          # 0=off; else odd (3/5) slight blur before OR
LONG_EXP_OVERLAY      = False      # also save overlay on the first input frame
LONG_EXP_OVERLAY_ALPHA= 0.70

# Connected Components display
CC_DRAW_BOXES         = True       # draw thin white boxes around kept components (area-filtered)

SHOW_PROGRESS_EVERY   = 20

# ───────────── UTILITIES ─────────────

def _progress(i: int, total: int, tag: str = ""):
    total = max(int(total or 1), 1)
    i = min(i, total)
    bar_w = 36
    frac = i / total
    fill = int(frac * bar_w)
    bar = "█" * fill + "·" * (bar_w - fill)
    print(f"\r[{bar}] {i}/{total} {tag}", end="")
    if i == total:
        print("")

def _open_video(path: Path):
    assert path.exists(), f"Input not found: {path}"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, w, h, float(fps), count

def _make_writer(path: Path, w: int, h: int, fps: float, codec: str = CODEC, is_color: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path} (try a different CODEC).")
    return writer

def _crop_with_pad(frame_bgr: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    """Center crop size×size around (cx,cy), pad with black if near borders."""
    H, W = frame_bgr.shape[:2]
    s = int(size)
    x0 = int(cx) - s // 2
    y0 = int(cy) - s // 2
    x1 = x0 + s
    y1 = y0 + s

    # Compute valid region
    vx0 = max(0, x0); vy0 = max(0, y0)
    vx1 = min(W, x1); vy1 = min(H, y1)

    patch = np.zeros((s, s, 3), dtype=np.uint8)
    px0 = vx0 - x0; py0 = vy0 - y0  # where to place inside patch
    patch[py0:py0+(vy1-vy0), px0:px0+(vx1-vx0)] = frame_bgr[vy0:vy1, vx0:vy1 and vx1]  # keep original
    patch[py0:py0+(vy1-vy0), px0:px0+(vx1-vx0)] = frame_bgr[vy0:vy1, vx0:vx1]          # fix slice
    return patch

def _chunk_tag(or_path: Path) -> str:
    """Return suffix like 'chunk_000000_000499' (without leading underscore)."""
    stem = or_path.stem
    if "_chunk_" in stem:
        return stem.split("_chunk_", 1)[1]
    return stem

def _cc_paths_for_chunk(or_path: Path) -> tuple[Path, Path]:
    """
    From e.g. 'output_long_exposure_chunk_000000_000499.png'
    make:
      CC_ALL_CHUNKS_DIR/'long_exp_components_all_chunk_000000_000499.png'
      CC_FILTERED_CHUNKS_DIR/'long_exp_components_filtered_chunk_000000_000499.png'
    """
    tag = _chunk_tag(or_path)
    all_name = f"long_exp_components_all_chunk_{tag}.png"
    filt_name = f"long_exp_components_filtered_chunk_{tag}.png"
    CC_ALL_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    CC_FILTERED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    return CC_ALL_CHUNKS_DIR / all_name, CC_FILTERED_CHUNKS_DIR / filt_name

def _idmap_path_for_chunk(or_path: Path) -> Path:
    """Sidecar mapping file that maps LOCAL component id -> GLOBAL component id for this chunk."""
    return or_path.with_suffix("").with_name(or_path.stem + "_idmap.npy")

def _tmin_path_for_chunk(or_path: Path) -> Path:
    """Sidecar array tmin_local (index=local id) for this chunk."""
    return or_path.with_suffix("").with_name(or_path.stem + "_tmin.npy")

# ───────────── STAGE 1: THRESHOLD VIDEO ─────────────

def write_threshold_video(inp: Path, out: Path, max_frames: int | None, out_fps: float | None):
    cap, W, H, src_fps, total_est = _open_video(inp)
    fps = float(out_fps or src_fps)
    writer = _make_writer(out, W, H, fps, codec=CODEC, is_color=True)

    # respect MAX_FRAMES when reading source
    hard_cap = max_frames if max_frames is not None else total_est
    total = min(total_est, hard_cap) if (hard_cap is not None and total_est > 0) else (hard_cap or total_est or 0)

    frame_idx = 0
    print(f"Stage1  Input:  {inp} ({W}x{H} @ {src_fps:.3f} fps)")
    print(f"Stage1  Output: {out} (fps={fps:.3f}, codec={CODEC})")
    print(f"Stage1  Mode={MODE}, Threshold={THRESHOLD_8BIT}, Invert={INVERT}")

    try:
        while True:
            if hard_cap is not None and frame_idx >= hard_cap:
                break
            ok, bgr = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            th_type = cv2.THRESH_BINARY_INV if INVERT else cv2.THRESH_BINARY
            _, mask = cv2.threshold(gray, THRESHOLD_8BIT, 255, th_type)

            if MODE == "mask":
                out_frame = cv2.merge([mask, mask, mask])
            elif MODE == "overlay":
                out_frame = bgr.copy()
                out_frame[mask > 0] = (255, 255, 255)
            else:
                raise ValueError("MODE must be 'mask' or 'overlay'.")

            writer.write(out_frame)
            frame_idx += 1
            if frame_idx % max(1, SHOW_PROGRESS_EVERY) == 0:
                _progress(frame_idx, total or frame_idx, "threshold frames")

        _progress(frame_idx, frame_idx, "threshold done")
    finally:
        cap.release()
        writer.release()

    return W, H, fps, frame_idx

# ───────────── STAGE 2: BACKGROUND SUBTRACTION ─────────────

def write_bgsub_video(inp: Path, out: Path, force_w: int | None = None, force_h: int | None = None,
                      fps_hint: float | None = None):
    cap, W0, H0, src_fps, total_est = _open_video(inp)
    W = force_w or W0
    H = force_h or H0
    fps = float(fps_hint or src_fps)

    bgs = cv2.createBackgroundSubtractorMOG2(
        detectShadows=bool(BGS_DETECT_SHADOWS),
        history=int(BGS_HISTORY)
    )

    # Single-channel writer for FG mask
    writer = _make_writer(out, W, H, fps, codec=CODEC, is_color=False)

    frame_idx = 0
    total = total_est
    print(f"Stage2  Input:  {inp} ({W0}x{H0} @ {src_fps:.3f} fps)")
    print(f"Stage2  Output: {out} (fps={fps:.3f}, codec={CODEC})")
    print(f"Stage2  MOG2: history={BGS_HISTORY}, shadows={BGS_DETECT_SHADOWS}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)

            fg = bgs.apply(frame, learningRate=BGS_LEARNING_RATE)

            if MORPH_OPEN_ITERS > 0:
                k = int(MORPH_KERNEL_SIZE)
                if k % 2 == 0: k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=int(MORPH_OPEN_ITERS))

            writer.write(fg)

            frame_idx += 1
            if frame_idx % max(1, SHOW_PROGRESS_EVERY) == 0:
                _progress(frame_idx, total or frame_idx, "bgsub frames")

        _progress(frame_idx, frame_idx, "bgsub done")
    finally:
        cap.release()
        writer.release()

    return frame_idx

# ───────────── STAGE 3: LONG-EXPOSURE via LOGICAL OR (CHUNKED) + TELEMETRY CSV ─────────────

def render_long_exposure_or_from_video(inp: Path, out_img: Path,
                                       overlay_out: Path | None = None,
                                       overlay: bool = False):
    """
    Build OR-trails images in chunks of LONG_EXP_CHUNK_SIZE frames.
    Also appends per-pixel telemetry to CC_PIXELS_CSV_PATH with GLOBAL component ids.
    Returns list of chunk image paths.
    """
    cap0, W, H, src_fps, total_est = _open_video(inp)
    cap0.release()

    # reset CSV (header)
    with open(CC_PIXELS_CSV_PATH, "w") as f:
        f.write("x,y,t,global_component_id\n")

    # ensure output dirs
    OR_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    cid_offset = 0  # global component id offset across chunks

    # chunk loop
    start_global = 0
    # Respect LONG_EXP_START_FRAME by skipping only once globally
    first_effective_start = max(LONG_EXP_START_FRAME, 0)

    while start_global < total_est:
        end_global = min(start_global + int(LONG_EXP_CHUNK_SIZE), total_est)
        start_eff = max(start_global, first_effective_start)  # apply warm-up skip only within first chunk

        trails = np.zeros((H, W), dtype=np.bool_)
        first_map = np.full((H, W), -1, dtype=np.int32)

        # Read frames for this chunk
        cap = cv2.VideoCapture(str(inp))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_eff))
        t = int(start_eff)

        # Optional dilation kernel
        kernel = None
        if LONG_EXP_DILATE_ITERS > 0:
            k = int(LONG_EXP_DILATE_KERNEL)
            if k % 2 == 0: k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        total_chunk_frames = max(0, end_global - start_eff)
        seen = 0

        print(f"Stage3  OR-chunk {start_global:06d}-{end_global-1:06d}  from={start_eff}  (fg_thresh={FG_MASK_THRESHOLD})")

        try:
            while t < end_global:
                ok, frame = cap.read()
                if not ok:
                    break

                # Ensure single-channel
                if frame.ndim == 3:
                    fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    fg = frame

                if LONG_EXP_BLUR_KSIZE and LONG_EXP_BLUR_KSIZE > 1:
                    k = int(LONG_EXP_BLUR_KSIZE)
                    if k % 2 == 0: k += 1
                    fg = cv2.GaussianBlur(fg, (k, k), 0)

                mask = fg >= int(FG_MASK_THRESHOLD)

                if kernel is not None and LONG_EXP_DILATE_ITERS > 0:
                    mask8 = (mask.astype(np.uint8) * 255)
                    mask8 = cv2.dilate(mask8, kernel, iterations=int(LONG_EXP_DILATE_ITERS))
                    mask = mask8 > 0

                trails |= mask
                new_on = mask & (first_map < 0)
                first_map[new_on] = t

                t += 1
                seen += 1
                if seen % max(1, SHOW_PROGRESS_EVERY) == 0:
                    _progress(seen, total_chunk_frames or seen, "OR chunk")
            _progress(seen, seen or 1, "OR chunk done")
        finally:
            cap.release()

        # Save chunk trails image (into OR_CHUNKS_DIR)
        base = LONG_EXP_IMAGE_PATH
        base.parent.mkdir(parents=True, exist_ok=True)
        out_chunk = OR_CHUNKS_DIR / f"{base.stem}_chunk_{start_global:06d}_{end_global-1:06d}{base.suffix}"
        trails_img = (trails.astype(np.uint8) * 255)
        cv2.imwrite(str(out_chunk), trails_img)
        print(f"Stage3  Saved OR trails image → {out_chunk}")
        chunk_paths.append(out_chunk)

        # Optional overlay (kept next to overlay path)
        if overlay:
            cap_in = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
            cap_in.set(cv2.CAP_PROP_POS_FRAMES, int(start_eff))
            ok0, base_bgr = cap_in.read()
            cap_in.release()
            if ok0:
                trails_rgb = cv2.merge([trails_img, trails_img, trails_img])
                overlay_img = cv2.addWeighted(base_bgr, 1.0, trails_rgb, float(LONG_EXP_OVERLAY_ALPHA), 0.0)
                out_overlay = LONG_EXP_OVERLAY_IMAGE_PATH.with_name(
                    f"{LONG_EXP_OVERLAY_IMAGE_PATH.stem}_chunk_{start_global:06d}_{end_global-1:06d}{LONG_EXP_OVERLAY_IMAGE_PATH.suffix}"
                )
                cv2.imwrite(str(out_overlay), overlay_img)
                print(f"Stage3  Saved overlay image → {out_overlay}")

        # Connected components for telemetry on this chunk (global IDs via offset)
        mask_u8 = (trails_img > 0).astype(np.uint8) * 255
        num, labels, stats, cents = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        ys, xs = np.nonzero(mask_u8)
        local_ids = labels[ys, xs].astype(np.int32)   # 1..(num-1)
        # keep only foreground labels (>=1)
        m_fg = local_ids >= 1
        xs = xs[m_fg]; ys = ys[m_fg]; local_ids = local_ids[m_fg]
        ts = first_map[ys, xs].astype(np.int32)

        # Build and SAVE local->global id map for this chunk (so Stage 4 can draw GLOBAL ids)
        idmap = np.arange(num, dtype=np.int64)
        idmap[1:] = idmap[1:] + int(cid_offset)
        idmap_path = _idmap_path_for_chunk(out_chunk)
        np.save(str(idmap_path), idmap)
        print(f"Stage3  Saved idmap (local→global) → {idmap_path.name}  (offset={cid_offset})")

        # Also save tmin per LOCAL component (index = local id)
        tmin_local = np.full(num, -1, dtype=np.int64)
        if local_ids.size > 0:
            uniq = np.unique(local_ids)
            for li in uniq:
                tmin_local[int(li)] = int(ts[local_ids == li].min())
        tmin_path = _tmin_path_for_chunk(out_chunk)
        np.save(str(tmin_path), tmin_local)
        print(f"Stage3  Saved tmin (per local id) → {tmin_path.name}")

        global_ids = local_ids + int(cid_offset)

        data = np.column_stack([xs, ys, ts, global_ids])
        with open(CC_PIXELS_CSV_PATH, "ab") as fb:
            np.savetxt(fb, data, fmt="%d,%d,%d,%d")

        print(f"Stage3  Appended telemetry rows: {data.shape[0]} (chunk comps={num-1}, cid_offset={cid_offset})")
        cid_offset += (num - 1)

        # advance to next chunk
        start_global = end_global
        # after first loop, warm-up already applied
        first_effective_start = 0

    return chunk_paths

# ───────────── STAGE 4: CONNECTED COMPONENTS on each long-exposure (area-filtered view) ─────────────

def connected_components_visuals(or_image_path: Path,
                                 all_vis_path: Path,
                                 filtered_vis_path: Path,
                                 min_area: int, max_area: int, min_aspect: float,
                                 draw_boxes: bool = True):
    """
    Produces:
      • all_vis_path: all components colored (saved in CC_ALL_CHUNKS_DIR)
      • filtered_vis_path: components with area >= min_area (and <= max_area if >0),
        colored + boxes + **GLOBAL** ID labels (using saved idmap for this chunk) (saved in CC_FILTERED_CHUNKS_DIR)
      • NEW: crops (from the *all* image, WITHOUT boxes) for each kept component,
        saved in CC_CROPS_DIR/<chunk_tag>/ with filename:
        gid_<GLOBAL>_x<X>_y<Y>_w<W>_h<H>_t<TMIN>.png
    """
    CC_ALL_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    CC_FILTERED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    CC_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(or_image_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Cannot read OR image: {or_image_path}"
    H, W = img.shape[:2]

    # Binary mask (any nonzero counts as foreground)
    mask_u8 = (img > 0).astype(np.uint8) * 255

    # Connected components
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    print(f"Stage4  Connected components: total={num-1}  ({or_image_path.name})")

    # Load local->global id map and tmin (if present)
    idmap_path = _idmap_path_for_chunk(or_image_path)
    tmin_path = _tmin_path_for_chunk(or_image_path)
    idmap = None
    tmin_local = None
    if idmap_path.exists():
        try:
            idmap = np.load(str(idmap_path))
            if idmap.shape[0] != num:
                print(f"Stage4  ⚠️ idmap length mismatch for {or_image_path.name}; falling back to local ids.")
                idmap = None
        except Exception:
            print(f"Stage4  ⚠️ failed loading idmap {idmap_path.name}; falling back to local ids.")
            idmap = None
    if tmin_path.exists():
        try:
            tmin_local = np.load(str(tmin_path))
        except Exception:
            tmin_local = None

    # Color LUT (deterministic)
    rng = np.random.default_rng(12345)
    colors = np.zeros((num, 3), dtype=np.uint8)
    for i in range(1, num):
        colors[i] = rng.integers(60, 255, size=3, endpoint=True, dtype=np.uint8)

    # All components visualization (no filtering)
    color_all = colors[labels]  # (H,W,3)
    cv2.imwrite(str(all_vis_path), color_all)
    print(f"Stage4  Saved ALL components → {all_vis_path}")

    # Area-filtered visualization
    keep = np.zeros(num, dtype=bool)
    min_a = int(max(0, min_area))
    max_a = int(max_area) if int(max_area) > 0 else None
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_a:
            continue
        if max_a is not None and area > max_a:
            continue
        keep[i] = True

    keep_mask = keep[labels]
    color_filtered = np.zeros_like(color_all)
    color_filtered[keep_mask] = color_all[keep_mask]

    # NEW: save per-component crops (from ALL image, so boxes don't appear)
    chunk_tag = _chunk_tag(or_image_path)
    crops_dir = CC_CROPS_DIR / f"chunk_{chunk_tag}"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num):
        if not keep[i]:
            continue
        x, y, w, h, area = stats[i]
        # clamp bounds
        x0 = max(0, int(x)); y0 = max(0, int(y))
        x1 = min(W, int(x + w)); y1 = min(H, int(y + h))
        crop = color_all[y0:y1, x0:x1].copy()
        g_id = int(idmap[i]) if idmap is not None else i
        tmin = int(tmin_local[i]) if (tmin_local is not None and i < len(tmin_local) and tmin_local[i] >= 0) else -1
        out_name = f"gid_{g_id:06d}_x{x0}_y{y0}_w{(x1-x0)}_h{(y1-y0)}_t{tmin}.png"
        cv2.imwrite(str(crops_dir / out_name), crop)

    if draw_boxes:
        for i in range(1, num):
            if not keep[i]:
                continue
            x, y, w, h, area = stats[i]
            # Draw box
            cv2.rectangle(color_filtered, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 1, cv2.LINE_AA)
            # Draw **GLOBAL** component ID at top-left (with outline for readability)
            g_id = int(idmap[i]) if idmap is not None else i
            label_text = str(g_id)
            ty = y - 4 if y - 4 >= 10 else y + 14
            cv2.putText(color_filtered, label_text, (x, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(color_filtered, label_text, (x, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(filtered_vis_path), color_filtered)
    kept_count = int(keep.sum())
    print(f"Stage4  Saved AREA-FILTERED components (kept={kept_count}, min_area={min_a}"
          f"{'' if max_a is None else f', max_area={max_a}'} ) → {filtered_vis_path}")
    print(f"Stage4  Saved crops for kept components → {crops_dir}")

# ───────────── STAGE 5: 10×10 crops per component (sequential, fast) ─────────────

def make_component_gifs_from_csv(csv_path: Path,
                                 input_video_path: Path,
                                 out_dir: Path,
                                 gif_size: int = 10,
                                 gif_fps: int = 8,
                                 limit_components: int = 0):
    """
    Replacement: save 10×10 PNG patches per component (one folder per component).
    Uses sequential decoding for speed.

    Layout:
      out_dir/
        00001/t_000123.png, t_000130.png, ...
        00002/...
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV (x,y,t,global_component_id)
    data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1, dtype=np.int64)
    if data.ndim == 1 and data.size == 4:
        data = data.reshape(1, 4)
    if data.size == 0:
        print("Stage5  No rows in CSV; skipping.")
        return 0

    xs = data[:, 0]
    ys = data[:, 1]
    ts = data[:, 2]
    cids = data[:, 3]

    # Choose components (optionally limited)
    all_cids = np.unique(cids)
    if limit_components and limit_components > 0:
        keep_cids = set(all_cids[:int(limit_components)])
        mkeep = np.isin(cids, list(keep_cids))
        xs, ys, ts, cids = xs[mkeep], ys[mkeep], ts[mkeep], cids[mkeep]
        chosen_cids = np.unique(cids)
    else:
        chosen_cids = all_cids

    # Aggregate centers per (cid, t): mean of pixels that turned FG at t
    from collections import defaultdict
    accum = {}  # (cid,t) -> [sumx, sumy, n]
    for x, y, t, cid in zip(xs, ys, ts, cids):
        key = (int(cid), int(t))
        if key not in accum:
            accum[key] = [0.0, 0.0, 0]
        a = accum[key]
        a[0] += float(x); a[1] += float(y); a[2] += 1
    centers_by_t = defaultdict(list)  # t -> [(cid, cx, cy)]
    for (cid, t), (sx, sy, n) in accum.items():
        cx = int(np.round(sx / max(1, n)))
        cy = int(np.round(sy / max(1, n)))
        centers_by_t[int(t)].append((int(cid), cx, cy))

    # Prepare folders
    for cid in chosen_cids:
        (out_dir / f"{int(cid):05d}").mkdir(parents=True, exist_ok=True)

    # Nothing to do?
    if not centers_by_t:
        print("Stage5  No times found; skipping.")
        return 0

    # Sequential decode from min_t to max_t
    times_sorted = sorted(centers_by_t.keys())
    min_t, max_t = times_sorted[0], times_sorted[-1]

    cap, W, H, fps, total_est = _open_video(input_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(min_t))

    made = 0
    cur_t = int(min_t)
    try:
        while cur_t <= max_t:
            ok, frame = cap.read()
            if not ok:
                break

            if cur_t in centers_by_t:
                for (cid, cx, cy) in centers_by_t[cur_t]:
                    patch = _crop_with_pad(frame, cx, cy, gif_size)
                    out_path = out_dir / f"{int(cid):05d}" / f"t_{int(cur_t):06d}.png"
                    cv2.imwrite(str(out_path), patch)
                    made += 1

            cur_t += 1
            if (cur_t - min_t) % max(1, SHOW_PROGRESS_EVERY) == 0:
                _progress(cur_t - min_t, (max_t - min_t + 1), "patch frames")

        _progress(max_t - min_t + 1, max_t - min_t + 1, "patch frames done")
    finally:
        cap.release()

    print(f"Stage5  Saved {made} patches → {out_dir}")
    return made

# ───────────── MAIN ─────────────

def main():
    out_bg = OUTPUT_BGSUB_PATH or OUTPUT_THRESH_PATH.with_name(
        OUTPUT_THRESH_PATH.stem + "_bgsub" + OUTPUT_THRESH_PATH.suffix
    )

    # Stage 1: threshold video
    W, H, fps, n1 = write_threshold_video(INPUT_VIDEO_PATH, OUTPUT_THRESH_PATH, MAX_FRAMES, OUTPUT_FPS)

    # Stage 2: background subtraction on the thresholded video
    n2 = write_bgsub_video(OUTPUT_THRESH_PATH, out_bg, force_w=W, force_h=H, fps_hint=fps)

    # Stage 3: long-exposure OR images (chunked) + telemetry CSV from the BG-sub video
    overlay_out = LONG_EXP_OVERLAY_IMAGE_PATH if LONG_EXP_OVERLAY else None
    chunk_imgs = render_long_exposure_or_from_video(out_bg, LONG_EXP_IMAGE_PATH, overlay_out, LONG_EXP_OVERLAY)

    # Stage 4: connected components visualizations (area-filtered view) for EACH chunk image
    for or_img in chunk_imgs:
        all_vis, filt_vis = _cc_paths_for_chunk(or_img)
        connected_components_visuals(
            or_img,
            all_vis,
            filt_vis,
            min_area=CC_MIN_AREA,
            max_area=CC_MAX_AREA,
            min_aspect=0.0,      # ignored
            draw_boxes=CC_DRAW_BOXES,
        )

    # Stage 5: per-component 10×10 crops from the ORIGINAL video (saved as PNGs)
    made = make_component_gifs_from_csv(
        CC_PIXELS_CSV_PATH,
        INPUT_VIDEO_PATH,
        GIF_OUTPUT_DIR,
        gif_size=GIF_SIZE,
        gif_fps=GIF_FPS,
        limit_components=GIF_LIMIT_COMPONENTS
    )

    print("\nSummary:")
    print(f"  Threshold frames written: {n1}")
    print(f"  BG-sub frames written:    {n2}")
    print(f"  OR trails chunks folder:  {OR_CHUNKS_DIR}")
    print(f"  OR trails chunks count:   {len(chunk_imgs)} (first: {chunk_imgs[0] if chunk_imgs else 'None'})")
    print(f"  Telemetry CSV:            {CC_PIXELS_CSV_PATH}")
    print(f"  CC ALL folder:            {CC_ALL_CHUNKS_DIR}")
    print(f"  CC FILTERED folder:       {CC_FILTERED_CHUNKS_DIR}")
    print(f"  CC CROPS folder:          {CC_CROPS_DIR}")
    print(f"  Component patches:        {made}  → {GIF_OUTPUT_DIR}")
    if LONG_EXP_OVERLAY:
        print(f"  Overlay images:           *_chunk_*.png next to {LONG_EXP_OVERLAY_IMAGE_PATH}")

if __name__ == "__main__":
    main()
