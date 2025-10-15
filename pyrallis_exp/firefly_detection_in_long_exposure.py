#!/usr/bin/env python3
"""
Given a long-exposure image path whose file/folder name encodes the video stem,
mode, and frame range, go back to the original video and extract crops for the
specified bounding boxes across that frame range. Instead of saving GIFs, this
script saves the individual cropped frames into a folder per box.

Expected image filename (as produced by long_exposure_from_raw_video.py):
  <interval>_<video_stem>_<mode>_<start>-<end>.png

This script parses that info from the provided image path. If the parent folder
encodes the same pattern, it will parse from there as well.

Detections input (YOLO format):
- Provide a text file path with one detection per line in YOLO format:
  class x_center y_center width height (all normalized 0..1). A 6th column
  with confidence is ignored if present.

Edit globals below and run from your IDE.
"""

from __future__ import annotations
from pathlib import Path
import re
import cv2
import numpy as np

# ===================== Globals (edit these) =====================

# Path to the long-exposure image (inside a folder that encodes metadata)
INPUT_IMAGE_PATH = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/long exposure shots/100_20240606_cam1_GS010064_lighten_000000-000099.png')

# Path to the ORIGINAL source video (set explicitly; no auto-search)
SOURCE_VIDEO_PATH = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/raw input videos/20240606_cam1_GS010064.mp4')

# Path to YOLO-format detections file corresponding to INPUT_IMAGE_PATH
# Each line: class x_center y_center width height [confidence]
DETECTIONS_FILE = Path("/Users/arnavps/Downloads/streaks.v1i.yolov8/train/labels/100_20240606_cam1_GS010064_lighten_000000-000099_png.rf.1f21359384de7bf4983250e919b898a1.txt")

# Output root directory. The script creates a subfolder with the same
# name as the input image (including extension) inside this directory.
OUTPUT_DIR = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/long exposure shots/temp testing dataset collection')

# Optional: target GIF FPS and resize (None to keep original crop size)
GIF_FPS = 8  # kept for pacing; not used for file output timing
RESIZE_TO: tuple[int, int] | None = None  # e.g., (128, 128)

# Frame output format
FRAME_PREFIX = "t_"          # filename prefix
FRAME_EXT = ".png"           # output image extension

# Video extensions to search for when auto-locating the source video
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}

# Final 10x10 brightest-centered crops output
REFINED_OUTPUT_DIR = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/long exposure shots/temp testing dataset collection/10x10 patches')
PATCH_SIZE = 10


# ===================== Helpers =====================

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


def _bbox_from_center(cx: float, cy: float, w: float, h: float) -> tuple[int, int, int, int]:
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    return int(x), int(y), int(round(w)), int(round(h))


def _crop_with_pad(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + int(w), y0 + int(h)
    vx0, vy0 = max(0, x0), max(0, y0)
    vx1, vy1 = min(W, x1), min(H, y1)
    patch = np.zeros((max(1, int(h)), max(1, int(w)), 3), dtype=np.uint8)
    if vx1 > vx0 and vy1 > vy0:
        px0, py0 = vx0 - x0, vy0 - y0
        patch[py0:py0 + (vy1 - vy0), px0:px0 + (vx1 - vx0)] = frame_bgr[vy0:vy1, vx0:vx1]
    return patch


def _parse_meta_from_image_path(img_path: Path) -> dict:
    """Parse interval, video_stem, mode, start, end from parent folder or filename.

    Patterns handled (greedy video group):
      - "<interval>_<video_stem>_<mode>_<start>-<end>" in parent folder name
      - same pattern in filename (if present)
    """
    name_candidates = [img_path.parent.name, img_path.stem]
    pat = re.compile(r"^(?P<interval>\d+?)_(?P<video>.+?)_(?P<mode>lighten|average|trails)_(?P<start>\d+)-(?P<end>\d+)$")
    for s in name_candidates:
        m = pat.match(s)
        if m:
            d = m.groupdict()
            return {
                "interval": int(d["interval"]),
                "video_stem": d["video"],
                "mode": d["mode"],
                "start": int(d["start"]),
                "end": int(d["end"]),
            }
    raise ValueError(
        f"Could not parse metadata from '{img_path}'. Expected parent folder or filename to match: <interval>_<video_stem>_<mode>_<start>-<end>"
    )


def _find_source_video(video_stem: str, img_path: Path) -> Path:
    """Heuristically locate the source video file by stem.

    Search strategy:
      1) Look in up to three ancestor directories of the image path and search recursively.
      2) Prefer exact stem match with known video extensions.
    """
    roots: list[Path] = []
    for p in [img_path.parent, img_path.parent.parent if img_path.parent.parent else None,
              img_path.parent.parent.parent if img_path.parent.parent else None]:
        if p and p.exists() and p.is_dir():
            roots.append(p)
    # Dedup while preserving order
    seen = set()
    uniq_roots = []
    for r in roots:
        if r not in seen:
            uniq_roots.append(r)
            seen.add(r)

    candidates: list[Path] = []
    for root in uniq_roots:
        for ext in VIDEO_EXTS:
            for p in root.rglob(f"{video_stem}{ext}"):
                if p.is_file():
                    candidates.append(p)
        if candidates:
            break

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate source video for stem '{video_stem}'. Please place the video under a nearby ancestor folder, or modify the search strategy."
        )
    # Prefer shortest path (closest root)
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def _save_frames_for_bbox(video_path: Path, bbox_xywh: tuple[int, int, int, int],
                          start: int, end: int, out_dir: Path,
                          resize_to: tuple[int, int] | None = RESIZE_TO) -> int:
    """Save cropped frames for [start, end] inclusive into out_dir.

    Returns number of frames written.
    """
    assert end >= start, "end must be >= start"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    s = max(0, int(start))
    e = min(int(end), max(0, total - 1)) if total > 0 else int(end)
    x, y, w, h = bbox_xywh

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    cur = s
    try:
        while cur <= e:
            ok, frame = cap.read()
            if not ok:
                break
            patch = _crop_with_pad(frame, x, y, w, h)
            if resize_to:
                patch = cv2.resize(patch, resize_to, interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"{FRAME_PREFIX}{int(cur):06d}{FRAME_EXT}"
            okw = cv2.imwrite(str(out_path), patch)
            if not okw:
                raise RuntimeError(f"Failed writing frame: {out_path}")
            written += 1
            cur += 1
            if (cur - s) % 50 == 0:
                _progress(cur - s, (e - s + 1), tag="crop frames")
        _progress(e - s + 1, e - s + 1, tag="crop frames done")
    finally:
        cap.release()
    return written


def _center_brightest_crop(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Return a size×size crop centered on the brightest pixel (grayscale).
    Pads with black if near borders.
    """
    H, W = img_bgr.shape[:2]
    if H == 0 or W == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _minVal, _maxVal, _minLoc, maxLoc = cv2.minMaxLoc(gray)
    cx, cy = int(maxLoc[0]), int(maxLoc[1])
    s = int(size)
    x0 = cx - s // 2
    y0 = cy - s // 2
    x1 = x0 + s
    y1 = y0 + s
    vx0, vy0 = max(0, x0), max(0, y0)
    vx1, vy1 = min(W, x1), min(H, y1)
    out = np.zeros((s, s, 3), dtype=np.uint8)
    if vx1 > vx0 and vy1 > vy0:
        px0, py0 = vx0 - x0, vy0 - y0
        out[py0:py0 + (vy1 - vy0), px0:px0 + (vx1 - vx0)] = img_bgr[vy0:vy1, vx0:vx1]
    return out


def refine_crops_to_brightest(in_base: Path, out_dir: Path, size: int = PATCH_SIZE) -> tuple[int, int]:
    """For each crop folder under in_base, write brightest-centered size×size frames
    into a single folder out_dir.

    Output filenames encode the original crop folder and frame name to avoid collisions:
      <crop_folder>__<frame_name>.png

    Returns (num_folders_processed, num_frames_written)
    """
    if not in_base.exists() or not in_base.is_dir():
        return (0, 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    folders = [d for d in sorted(in_base.iterdir()) if d.is_dir()]
    total_written = 0
    processed = 0
    for crop_dir in folders:
        frames = sorted([p for p in crop_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not frames:
            continue
        processed += 1
        for fp in frames:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            patch = _center_brightest_crop(img, int(size))
            out_name = f"{crop_dir.name}__{fp.name}"
            out_path = out_dir / out_name
            okw = cv2.imwrite(str(out_path), patch)
            if not okw:
                raise RuntimeError(f"Failed to write refined frame: {out_path}")
            total_written += 1
    return (processed, total_written)


def _image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image for size: {path}")
    h, w = img.shape[:2]
    return w, h


def _read_yolo_boxes(lbl_path: Path, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
    """Read YOLO labels and return pixel (cx,cy,w,h) floats.

    Accepts 5 or 6 columns per line. Ignores empty/invalid lines.
    """
    assert lbl_path.exists() and lbl_path.is_file(), f"Labels file not found: {lbl_path}"
    boxes: list[tuple[float, float, float, float]] = []
    with open(lbl_path, "r") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                # YOLO: class cx cy w h [conf]
                cls = float(parts[0])  # unused
                cx_n = float(parts[1]); cy_n = float(parts[2])
                w_n  = float(parts[3]); h_n  = float(parts[4])
            except ValueError:
                continue
            # Clamp to [0,1]
            cx_n = max(0.0, min(1.0, cx_n)); cy_n = max(0.0, min(1.0, cy_n))
            w_n  = max(0.0, min(1.0, w_n));  h_n  = max(0.0, min(1.0, h_n))
            # Convert to pixels based on image size
            cx = cx_n * img_w; cy = cy_n * img_h
            ww = max(1.0, w_n * img_w); hh = max(1.0, h_n * img_h)
            boxes.append((cx, cy, ww, hh))
    return boxes


# ===================== Main =====================

def main():
    img_path = INPUT_IMAGE_PATH
    assert img_path.exists(), f"Input image not found: {img_path}"

    meta = _parse_meta_from_image_path(img_path)
    interval = meta["interval"]
    video_stem = meta["video_stem"]
    mode = meta["mode"]
    start = meta["start"]
    end = meta["end"]

    print(f"Parsed: interval={interval}, video='{video_stem}', mode={mode}, frames=[{start},{end}]")

    # Use the explicitly provided source video path
    video_path = SOURCE_VIDEO_PATH
    assert video_path.exists() and video_path.is_file(), f"Source video not found: {video_path}"
    print(f"Using source video: {video_path}")

    # Output base folder named after the input image filename
    out_base = OUTPUT_DIR / img_path.name
    out_base.mkdir(parents=True, exist_ok=True)

    # Load sizes for normalization conversion
    img_w, img_h = _image_size(img_path)
    # Also get video size for sanity/scaling
    cap = cv2.VideoCapture(str(video_path))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    sx = float(vw) / float(img_w) if img_w > 0 else 1.0
    sy = float(vh) / float(img_h) if img_h > 0 else 1.0

    # Read YOLO boxes (normalized) and convert to pixel centers/sizes
    yolo_boxes = _read_yolo_boxes(DETECTIONS_FILE, img_w, img_h)
    if not yolo_boxes:
        print(f"No boxes found in {DETECTIONS_FILE}; nothing to do.")
        return

    made = []
    for i, (cx, cy, bw, bh) in enumerate(yolo_boxes):
        # Scale to video size if image and video differ
        cx2 = cx * sx; cy2 = cy * sy; bw2 = bw * sx; bh2 = bh * sy
        x, y, w, h = _bbox_from_center(cx2, cy2, bw2, bh2)
        folder_name = (
            f"crop_{video_stem}_{mode}_{start:06d}-{end:06d}_"
            f"xc{int(round(cx2))}_yc{int(round(cy2))}_w{int(round(bw2))}_h{int(round(bh2))}"
        )
        out_dir = out_base / folder_name
        n = _save_frames_for_bbox(video_path, (x, y, w, h), start, end, out_dir)
        print(f"Saved {n} frames → {out_dir}")
        made.append(out_dir)

    print(f"Done. Saved crops for {len(made)} box(es) under {out_base}")

    # Final brightest-centered crops into a single folder named '<image_name>_10x10_patches'
    refined_dir = REFINED_OUTPUT_DIR / f"{img_path.name}_10x10_patches"
    refined_dir.mkdir(parents=True, exist_ok=True)
    nfolders, nframes = refine_crops_to_brightest(out_base, refined_dir, PATCH_SIZE)
    print(f"Refined {nfolders} folders, wrote {nframes} frames → {refined_dir}")


if __name__ == "__main__":
    main()
