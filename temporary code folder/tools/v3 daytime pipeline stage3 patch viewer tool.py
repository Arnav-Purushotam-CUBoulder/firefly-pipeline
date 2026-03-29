#!/usr/bin/env python3
"""
Stage 3 Patch Viewer – video + crop
───────────────────────────────────

Purpose
-------
Interactively inspect Stage 3 positive patches alongside their source video
frames, so you can:
  • See exactly where each 10×10 patch lives in the full frame.
  • Visually decide whether it is a real firefly or confusing noise.
  • Mark selected patches and copy them into a separate folder for retraining.

Expected patch layout (per video)
---------------------------------
This tool assumes Stage 3 saved positive patches with filenames like:
  f_000123_x10_y20_w10_h10_p0.995.png
under a folder such as:
  .../stage3_patch_classifier/<video_stem>/crops/positives

Filename fields:
  • frame index:   f_<frame_idx:06d>
  • top-left x,y:  x<int>_y<int>
  • size:          w<int>_h<int>
  • prob:          p<float with 3 decimals>

You only need:
  • VIDEO_PATH: path to the original video.
  • PATCHES_DIR: folder containing those Stage 3 positive crops.
  • OUTPUT_DIR: where to copy marked patches (for training data).

All three are configurable via globals below and via the GUI text boxes.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ───────── Global configuration (edit these as needed) ─────────
# Point these near your data; you can also override them from the GUI.

VIDEO_PATH = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/raw input videos/20240606_cam1_GS010064.mp4'  # e.g. "/path/to/original_video.mp4"
PATCHES_DIR = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset/v2/initial dataset/firefly'  # e.g. "/path/to/stage3_patch_classifier/<stem>/crops/positives"
OUTPUT_DIR = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset/v2/initial dataset/firefly refined'  # e.g. "/path/to/stage3_patch_classifier/<stem>/marked_for_training"

# How much to enlarge the tiny 10×10 patch for display
PATCH_BLOWUP_FACTOR = 5

# Maximum size for the full-frame display panel
FULL_MAX_WIDTH = 960
FULL_MAX_HEIGHT = 720


# ───────── Data structures ─────────
@dataclass
class PatchMeta:
    path: Path
    frame_idx: int
    x: int
    y: int
    w: int
    h: int
    conf: float
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None
    roi_w: Optional[int] = None
    roi_h: Optional[int] = None
    kind: str = "stage3"  # "stage3" | "crop"


class AppState:
    def __init__(self) -> None:
        self.video_path: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.patches: List[PatchMeta] = []
        self.index: int = 0
        self.photo_patch: Optional[ImageTk.PhotoImage] = None
        # External viewer state
        self.viewer_snapshot_path: Optional[Path] = None
        self.viewer_opened: bool = False

    def close_video(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.total_frames = 0


state = AppState()


# ───────── Helpers ─────────
_PATCH_RE = re.compile(
    r"^f_(?P<frame>\d+)_x(?P<x>-?\d+)_y(?P<y>-?\d+)_w(?P<w>\d+)_h(?P<h>\d+)_p(?P<p>[0-9.]+)\.(?:png|jpg|jpeg)$",
    re.IGNORECASE,
)

# Example:
#   crop_20240606_cam1_GS010064_lighten_008100-008199_xc4759_yc1451_w28_h29__t_008193.png
_CROP_RE = re.compile(
    r"^crop_(?P<video>.+?)_(?P<mode>lighten|average|trails)_(?P<start>\d+)-(?P<end>\d+)_"
    r"xc(?P<xc>-?\d+)_yc(?P<yc>-?\d+)_w(?P<w>\d+)_h(?P<h>\d+)__t_(?P<frame>\d+)"
    r"(?:\.(?:png|jpg|jpeg))?$",
    re.IGNORECASE,
)


def parse_patch_filename(p: Path) -> Optional[PatchMeta]:
    name = p.name
    # Ignore composite patches like "comp_04828_t_000255*"
    if name.startswith("comp_"):
        return None

    # 1) Stage3-style filename: f_<frame>_x<x>_y<y>_w<w>_h<h>_p<conf>.png
    m = _PATCH_RE.match(name)
    if m:
        try:
            frame_idx = int(m.group("frame"))
            x = int(m.group("x"))
            y = int(m.group("y"))
            w = int(m.group("w"))
            h = int(m.group("h"))
            conf = float(m.group("p"))
        except Exception:
            return None
        return PatchMeta(
            path=p,
            frame_idx=frame_idx,
            x=x,
            y=y,
            w=w,
            h=h,
            conf=conf,
            kind="stage3",
        )

    # 2) Long-exposure crop-derived filename:
    #    crop_<video>_<mode>_<start>-<end>_xc<xc>_yc<yc>_w<w>_h<h>__t_<frame>[.png]
    m2 = _CROP_RE.match(name)
    if m2:
        try:
            frame_idx = int(m2.group("frame"))
            xc = int(m2.group("xc"))
            yc = int(m2.group("yc"))
            roi_w = int(m2.group("w"))
            roi_h = int(m2.group("h"))
            # ROI (x,y,w,h) used when extracting crops from the original video
            roi_x = int(round(xc - roi_w / 2.0))
            roi_y = int(round(yc - roi_h / 2.0))
        except Exception:
            return None
        # Note: actual 10×10 patch location per frame comes from brightest
        # pixel inside this ROI; we recompute that when opening the frame.
        return PatchMeta(
            path=p,
            frame_idx=frame_idx,
            x=0,
            y=0,
            w=10,
            h=10,
            conf=0.0,
            roi_x=roi_x,
            roi_y=roi_y,
            roi_w=roi_w,
            roi_h=roi_h,
            kind="crop",
        )

    return None


def list_patches(dir_path: Path) -> List[PatchMeta]:
    metas: List[PatchMeta] = []
    if not dir_path.exists():
        return metas
    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        meta = parse_patch_filename(entry)
        if meta is not None:
            metas.append(meta)
    metas.sort(key=lambda m: (m.frame_idx, -m.conf, m.x, m.y))
    return metas


def load_video(path: Path) -> tuple[Optional[cv2.VideoCapture], int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    return cap, total


def pixel_blowup(img: Image.Image, factor: int = PATCH_BLOWUP_FACTOR) -> Image.Image:
    """Upscale very small crops with nearest-neighbour so pixels look blocky."""
    factor = max(1, int(factor))
    resample = getattr(Image, "NEAREST", Image.Resampling.NEAREST)
    return img.resize((img.width * factor, img.height * factor), resample)


def copy_marked_patch(meta: PatchMeta, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / meta.path.name
    if not out_path.exists():
        shutil.copy2(meta.path, out_path)


# ───────── GUI callbacks ─────────
def on_browse_video() -> None:
    p = filedialog.askopenfilename(
        title="Select original video",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")],
    )
    if p:
        video_var.set(p)


def on_browse_patches() -> None:
    p = filedialog.askdirectory(title="Select Stage3 positives folder (crops/positives)")
    if p:
        patches_var.set(p)


def on_browse_output() -> None:
    p = filedialog.askdirectory(title="Select output folder for marked patches")
    if p:
        output_var.set(p)


def on_load() -> None:
    video_path = Path(video_var.get()).expanduser()
    patches_dir = Path(patches_var.get()).expanduser()

    if not video_path.exists():
        messagebox.showerror("Error", f"Video not found:\n{video_path}")
        return
    if not patches_dir.exists():
        messagebox.showerror("Error", f"Patches folder not found:\n{patches_dir}")
        return

    cap, total = load_video(video_path)
    if cap is None or total <= 0:
        messagebox.showerror("Error", f"Could not open video or video has no frames:\n{video_path}")
        return

    patches = list_patches(patches_dir)
    if not patches:
        messagebox.showwarning("No patches", f"No Stage3 patch files found in:\n{patches_dir}")
        return

    state.close_video()
    state.video_path = video_path
    state.cap = cap
    state.total_frames = total
    state.patches = patches
    state.index = 0

    status_var.set(f"Loaded {len(patches)} patches from {patches_dir.name}")
    show_current()


def get_current_patch() -> Optional[PatchMeta]:
    if not state.patches:
        return None
    idx = max(0, min(state.index, len(state.patches) - 1))
    return state.patches[idx]


def show_current() -> None:
    meta = get_current_patch()
    if meta is None:
        patch_canvas.config(width=100, height=100)
        patch_canvas.delete("all")
        patch_info_var.set("No patches loaded.")
        return

    # Left: patch image (blown up)
    try:
        patch_img = Image.open(meta.path).convert("RGB")
    except Exception:
        patch_canvas.config(width=100, height=100)
        patch_canvas.delete("all")
        patch_info_var.set(f"#{state.index + 1}/{len(state.patches)}  (failed to read patch)")
        return

    blown = pixel_blowup(patch_img, PATCH_BLOWUP_FACTOR)
    state.photo_patch = ImageTk.PhotoImage(blown)
    patch_canvas.config(width=blown.width, height=blown.height)
    patch_canvas.delete("all")
    patch_canvas.create_image(0, 0, image=state.photo_patch, anchor=tk.NW)

    if meta.kind == "crop":
        info = (
            f"#{state.index + 1}/{len(state.patches)}  "
            f"frame={meta.frame_idx}  (crop-based; ROI "
            f"x={meta.roi_x} y={meta.roi_y} w={meta.roi_w} h={meta.roi_h})"
        )
    else:
        info = (
            f"#{state.index + 1}/{len(state.patches)}  "
            f"frame={meta.frame_idx}  conf={meta.conf:.3f}  "
            f"x={meta.x} y={meta.y} w={meta.w} h={meta.h}"
        )
    patch_info_var.set(info)

    # Auto-open full frame in external viewer for this patch
    open_current_frame(manual=False)

    # Update mark checkbox based on whether a copy exists in the output folder
    out_dir_str = output_var.get().strip() if "output_var" in globals() else ""
    if out_dir_str:
        out_dir = Path(out_dir_str).expanduser()
        out_path = out_dir / meta.path.name
        mark_var.set(out_path.exists())
    else:
        mark_var.set(False)


def next_patch(_event=None) -> None:
    if not state.patches:
        return
    if state.index + 1 < len(state.patches):
        state.index += 1
        show_current()


def prev_patch(_event=None) -> None:
    if not state.patches:
        return
    if state.index > 0:
        state.index -= 1
        show_current()


def open_current_frame(manual: bool = False) -> None:
    """Decode the current patch's frame from the video, draw its bbox,
    save as PNG, and open it in the OS default image viewer.

    When manual=False, errors are silenced so automated calls don't spam dialogs.
    """
    meta = get_current_patch()
    if meta is None:
        return

    video_path = Path(video_var.get()).expanduser()
    if not video_path.exists():
        if manual:
            messagebox.showerror("Error", f"Video not found:\n{video_path}")
        return

    # Ensure we have a capture for this video
    if state.cap is None or not state.cap.isOpened() or state.video_path != video_path:
        cap, total = load_video(video_path)
        if cap is None or total <= 0:
            if manual:
                messagebox.showerror("Error", f"Could not open video or video has no frames:\n{video_path}")
            return
        state.close_video()
        state.cap = cap
        state.video_path = video_path
        state.total_frames = total

    cap = state.cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(meta.frame_idx))
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        if manual:
            messagebox.showerror("Error", f"Could not read frame {meta.frame_idx} from video.")
        return

    H, W = frame_bgr.shape[:2]

    # Compute patch bbox depending on patch kind
    if meta.kind == "crop" and meta.roi_x is not None and meta.roi_y is not None and meta.roi_w is not None and meta.roi_h is not None:
        # Recompute brightest-centered 10×10 patch inside the original ROI
        rx0 = max(0, int(meta.roi_x))
        ry0 = max(0, int(meta.roi_y))
        rx1 = min(W, int(meta.roi_x + meta.roi_w))
        ry1 = min(H, int(meta.roi_y + meta.roi_h))
        roi = frame_bgr[ry0:ry1, rx0:rx1]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _minVal, _maxVal, _minLoc, maxLoc = cv2.minMaxLoc(gray)
            bx = rx0 + int(maxLoc[0])
            by = ry0 + int(maxLoc[1])
            s = 10
            x0 = int(round(bx - s / 2.0))
            y0 = int(round(by - s / 2.0))
            x0 = max(0, min(x0, W - s))
            y0 = max(0, min(y0, H - s))
            x1 = x0 + s
            y1 = y0 + s
        else:
            # Fallback: use ROI center
            bx = rx0 + max(0, (rx1 - rx0) // 2)
            by = ry0 + max(0, (ry1 - ry0) // 2)
            s = 10
            x0 = max(0, min(int(round(bx - s / 2.0)), W - s))
            y0 = max(0, min(int(round(by - s / 2.0)), H - s))
            x1 = x0 + s
            y1 = y0 + s
    else:
        # Stage3-style patch: x,y,w,h are already the patch box
        x0 = int(meta.x)
        y0 = int(meta.y)
        x1 = int(meta.x + meta.w)
        y1 = int(meta.y + meta.h)

    # Draw the patch bbox on the frame
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 0, 255), 1)

    # Determine snapshot directory: default under patches dir / "frame_snaps"
    try:
        patches_root = Path(patches_var.get() or PATCHES_DIR or ".").expanduser()
    except Exception:
        patches_root = Path(".")
    snap_dir = patches_root / "frame_snaps"
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Reuse a single snapshot path so the external viewer shows one window
    # that updates as we move through patches.
    out_path = snap_dir / "current_frame.png"
    try:
        cv2.imwrite(str(out_path), frame_bgr)
    except Exception as e:
        if manual:
            messagebox.showerror("Error", f"Failed to write frame image:\n{e}")
        return

    status_var.set(f"Saved frame snapshot → {out_path}")

    # Open with OS default viewer once; subsequent calls just update the file.
    if not state.viewer_opened:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(out_path)], check=False)
            elif os.name == "nt":
                os.startfile(str(out_path))  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(out_path)], check=False)
            state.viewer_opened = True
            state.viewer_snapshot_path = out_path
        except Exception as e:
            if manual:
                messagebox.showerror("Error", f"Failed to open frame image:\n{e}")


def on_mark_toggle() -> None:
    meta = get_current_patch()
    if meta is None:
        return
    out_dir_str = output_var.get().strip()
    if not out_dir_str:
        messagebox.showerror("Error", "Please set an output folder first.")
        mark_var.set(False)
        return

    out_dir = Path(out_dir_str).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / meta.path.name

    if mark_var.get():
        # Marked: ensure patch is copied into the output folder.
        try:
            if not out_path.exists():
                copy_marked_patch(meta, out_dir)
            status_var.set(f"Copied patch → {out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy patch:\n{e}")
            mark_var.set(False)
    else:
        # Unmarked: delete the copied patch from the output folder if present.
        try:
            if out_path.exists():
                out_path.unlink()
                status_var.set(f"Deleted patch copy → {out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete patch copy:\n{e}")
            # If deletion failed, keep it logically marked
            mark_var.set(True)


# ───────── GUI setup ─────────
root = tk.Tk()
root.title("Stage 3 Patch Viewer – video + crop")

video_var = tk.StringVar(value=VIDEO_PATH)
patches_var = tk.StringVar(value=PATCHES_DIR)
output_var = tk.StringVar(value=OUTPUT_DIR)
patch_info_var = tk.StringVar(value="No patches loaded.")
status_var = tk.StringVar(value="")
mark_var = tk.BooleanVar(value=False)

# Top controls frame
top = ttk.Frame(root)
top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

row0 = ttk.Frame(top)
row0.pack(fill=tk.X, pady=2)
ttk.Label(row0, text="Video:").pack(side=tk.LEFT)
ttk.Entry(row0, textvariable=video_var, width=80).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
ttk.Button(row0, text="Browse…", command=on_browse_video).pack(side=tk.LEFT, padx=4)

row1 = ttk.Frame(top)
row1.pack(fill=tk.X, pady=2)
ttk.Label(row1, text="Patches dir:").pack(side=tk.LEFT)
ttk.Entry(row1, textvariable=patches_var, width=80).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
ttk.Button(row1, text="Browse…", command=on_browse_patches).pack(side=tk.LEFT, padx=4)

row2 = ttk.Frame(top)
row2.pack(fill=tk.X, pady=2)
ttk.Label(row2, text="Output dir:").pack(side=tk.LEFT)
ttk.Entry(row2, textvariable=output_var, width=80).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
ttk.Button(row2, text="Browse…", command=on_browse_output).pack(side=tk.LEFT, padx=4)

row3 = ttk.Frame(top)
row3.pack(fill=tk.X, pady=4)
ttk.Button(row3, text="Load video + patches", command=on_load).pack(side=tk.LEFT)
ttk.Label(row3, textvariable=status_var, foreground="gray").pack(side=tk.LEFT, padx=10)

# Main content: left (patch) and right (full frame)
main = ttk.Frame(root)
main.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

left_frame = ttk.Frame(main)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

ttk.Label(left_frame, text="Patch (10×10 blown up)").pack(anchor="w")
patch_canvas = tk.Canvas(left_frame, width=160, height=160, bg="black")
patch_canvas.pack()
ttk.Label(left_frame, textvariable=patch_info_var, justify="left").pack(anchor="w", pady=4)

# Bottom controls: navigation + mark checkbox
bottom = ttk.Frame(root)
bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

ttk.Button(bottom, text="Prev", command=prev_patch).pack(side=tk.LEFT)
ttk.Button(bottom, text="Next", command=next_patch).pack(side=tk.LEFT, padx=(4, 10))
ttk.Button(bottom, text="Open full frame in viewer", command=open_current_frame).pack(side=tk.LEFT, padx=(4, 10))
ttk.Checkbutton(
    bottom,
    text="Mark (copy patch to output folder)",
    variable=mark_var,
    command=on_mark_toggle,
).pack(side=tk.LEFT)

ttk.Label(bottom, text="Use Left / Right arrow keys to navigate.").pack(side=tk.RIGHT)

# Keyboard bindings
root.bind("<Left>", prev_patch)
root.bind("<Right>", next_patch)


def _on_close():
    state.close_video()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", _on_close)

if __name__ == "__main__":
    root.mainloop()
