#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import cv2


# ================= Global Configuration (edit these) =================
INPUT_VIDEO_PATH = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/raw input videos/20240606_cam1_GS010064.mp4'
OUTPUT_DIR = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/raw input videos/20240606_cam1_GS010064 frames'

# How many frames to save starting from START_FRAME
NUM_FRAMES = 20               # Set to the number of frames you want
START_FRAME = 0                # 0-based index of the first frame to save

# Saving options
OUTPUT_EXT = ".png"           # Only PNG requested, keep as .png
FILENAME_PREFIX = "frame"     # Output filename prefix
ZERO_PAD = 6                   # Zero-padding width in filenames
OVERWRITE_EXISTING = False     # Overwrite existing files if True
DRY_RUN = False                # Preview without writing files

# Progress bar settings
BAR_LEN = 50                   # Visual length of the progress bar
# ====================================================================


def progress(i: int, total: int, tag: str = "", live: str = "") -> None:
    total = max(1, int(total))
    i = min(max(0, int(i)), total)
    frac = i / total
    fill = int(frac * BAR_LEN)
    bar = "=" * fill + " " * (BAR_LEN - fill)
    sys.stdout.write(f"\r{tag} [{bar}] {int(frac*100):3d}% {live}")
    sys.stdout.flush()
    if i >= total:
        sys.stdout.write("\n")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    vid_path = Path(INPUT_VIDEO_PATH).expanduser()
    out_dir = Path(OUTPUT_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if NUM_FRAMES is None or NUM_FRAMES <= 0:
        log("NUM_FRAMES must be a positive integer.")
        return

    t0 = time.time()
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {vid_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    log(f"Input : {vid_path}")
    log(f"Output: {out_dir}")
    log(f"Meta  : frames={total_frames}, res={width}x{height} @ {fps:.2f} fps")

    # Seek to START_FRAME (best effort)
    if START_FRAME > 0:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_FRAME))
        if not ok:
            # Fallback: read-and-discard until START_FRAME
            current = 0
            while current < START_FRAME:
                ok, _ = cap.read()
                if not ok:
                    break
                current += 1

    # Determine how many frames to attempt
    target = int(NUM_FRAMES)
    if total_frames > 0:
        # If metadata available, do not exceed stream length
        remaining = max(0, total_frames - max(0, int(START_FRAME)))
        target = min(target, remaining)
    if target <= 0:
        log("Nothing to save (target <= 0).")
        cap.release()
        return

    saved = 0
    read_count = 0
    start_time = time.time()
    progress(0, target, tag="save-frames", live="starting…")

    while saved < target:
        ok, frame = cap.read()
        read_count += 1
        if not ok:
            break

        # Compute absolute frame index for naming (if available)
        # Note: CAP_PROP_POS_FRAMES may be one ahead on some backends
        abs_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or (START_FRAME + read_count)) - 1
        fname = f"{FILENAME_PREFIX}_{abs_frame_idx:0{ZERO_PAD}d}{OUTPUT_EXT}"
        fpath = out_dir / fname

        if DRY_RUN:
            wrote = True
        else:
            if fpath.exists() and not OVERWRITE_EXISTING:
                wrote = True
            else:
                wrote = cv2.imwrite(str(fpath), frame)

        if wrote:
            saved += 1

        elapsed = max(1e-6, time.time() - start_time)
        spd = saved / elapsed
        live = f"saved {saved}/{target} | {spd:.1f} fps"
        progress(saved, target, tag="save-frames", live=live)

    cap.release()

    total_elapsed = time.time() - t0
    log("=== Summary ===")
    log(f"Requested: {NUM_FRAMES} from start={START_FRAME}")
    log(f"Saved    : {saved} to {out_dir}")
    log(f"Time     : {total_elapsed:.2f}s")
    if DRY_RUN:
        log("[DRY RUN] No files written.")
    log("===============")


if __name__ == "__main__":
    main()

