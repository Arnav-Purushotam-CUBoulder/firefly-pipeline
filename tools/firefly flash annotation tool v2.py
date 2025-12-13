#!/usr/bin/env python3
"""
Firefly-flash annotator – video v2
──────────────────────────────────
Based on the image-folder annotator v4.7, but:
  • Takes a single video file as input instead of a folder of frames.
  • Loads frames on demand via OpenCV (no pre-extraction step).
  • Continues to append annotations directly to CSV on every change.

Usage notes
  • Left / Right arrows: move to previous / next frame.
  • Space: undo last box on the current frame.
  • Click: add a 10×10 box, snapped to the brightest spot near the click.
  • Hover: shows R,G,B and luminance of the brightest pixel in a 10×10 patch
    centred on the cursor.

CSV format
  Columns: x, y, w, h, frame
  • Coordinates (x, y) are top-left of the box in ORIGINAL pixel coordinates.
  • w, h are the box width and height (typically 10×10).
  • frame is the 0-based video frame index.
"""

# ───────── imports ─────────
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ───────── helper ─────────
def _basename(p: str | Path) -> str:
    """Return just the filename, stripping any leading directories."""
    return Path(p).name


# ───────── global paths ─────────
# Keep these pointing near your data; auto-load only if the paths exist.
DEFAULT_VIDEO_PATH = (
    '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset collection/raw input videos/20240606_cam1_GS010064.mp4'
)
DEFAULT_CSV_FILE = (
    '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/testing v2 firefly annotation tool/test1.csv'
)

BOX_SIZE = 10       # also used as hover patch size
SEARCH_RADIUS = 30  # radius (in pixels) around click for brightest search

# Max number of decoded frames to keep in memory
# for fast backwards stepping.
MAX_CACHE_FRAMES = 64

state: dict[str, object] = {
    "video_path":   None,   # Path of currently loaded video
    "cap":          None,   # cv2.VideoCapture
    "total_frames": 0,
    "idx":          0,      # current 0-based frame index
    "boxes":        {},     # {frame_index: [(x,y), …]} in original pixels
    "photo":        None,
    "current_np":   None,   # RGB numpy array of full-res frame
    "scale":        1.0,
    "hover_id":     None,   # canvas text item for hover read-out
    "cap_pos":      None,   # index of last decoded frame in cap
    "frame_cache":  {},     # {frame_index: np.ndarray RGB}
    "cache_order":  [],     # [frame_index, …] LRU-ish order
}


# ───────── helper functions ─────────
def get_csv_path() -> Path:
    return Path(csv_file_var.get()).expanduser()


def append_rows(rows: list[dict]) -> None:
    """Append list-of-dicts to CSV, creating the file + header if needed."""
    if not rows:
        return
    csv_path = get_csv_path()
    newfile = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["x", "y", "w", "h", "frame"])
        if newfile:
            wr.writeheader()
        wr.writerows(rows)
        fh.flush()  # ensure on-disk immediately


def _load_boxes_from_csv() -> dict[int, list[tuple[int, int]]]:
    """
    Read existing annotations from CSV, grouping them by frame index.

    Expected CSV columns: x, y, w, h, frame
    """
    csv_path = get_csv_path()
    boxes: dict[int, list[tuple[int, int]]] = {}
    if not csv_path.exists():
        return boxes

    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                f = int(r["frame"])
                x = int(round(float(r["x"])))
                y = int(round(float(r["y"])))
            except Exception:
                continue
            boxes.setdefault(f, []).append((x, y))
    return boxes


def update_stats() -> None:
    total = int(state.get("total_frames") or 0)
    if total <= 0:
        stats_var.set("No video loaded.")
        return

    boxes: dict[int, list[tuple[int, int]]] = state["boxes"]  # type: ignore[assignment]
    annotated = sum(
        1 for f, v in boxes.items() if v and 0 <= f < total
    )
    total_boxes = sum(
        len(v) for f, v in boxes.items() if 0 <= f < total
    )
    avg = total_boxes / annotated if annotated else 0.0
    remain = total - annotated
    pct = annotated / total * 100 if total else 0.0
    stats_var.set(
        f"Frames annotated: {annotated}/{total}  ({pct:.1f}%)\n"
        f"Total boxes: {total_boxes}\n"
        f"Average boxes / annotated frame: {avg:.2f}\n"
        f"Frames remaining: {remain}"
    )


def _close_cap() -> None:
    cap = state.get("cap")
    if cap is not None:
        try:
            cap.release()  # type: ignore[call-arg]
        except Exception:
            pass
    state["cap"] = None
    state["cap_pos"] = None
    state["frame_cache"] = {}
    state["cache_order"] = []
    state["cap_pos"] = None


def load_video(p: str | Path) -> None:
    """Open a video file, load existing CSV annotations, and jump to first unannotated frame."""
    p = Path(p).expanduser()
    if not p.exists():
        messagebox.showerror("Error", f"Video {p} not found")
        return

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video {p}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        messagebox.showerror("Error", f"Video {p} has no frames")
        return

    _close_cap()

    boxes = _load_boxes_from_csv()

    # Choose start frame: first frame without any boxes, or last frame if all have boxes
    start = 0
    if total > 0:
        start = next(
            (i for i in range(total) if not boxes.get(i)),
            total - 1,
        )

    state.update(
        video_path=p,
        cap=cap,
        total_frames=total,
        idx=start,
        boxes=boxes,
        current_np=None,
        scale=1.0,
        hover_id=None,
        photo=None,
        cap_pos=None,
        frame_cache={},
        cache_order=[],
    )
    show_frame()
    update_stats()


def save_box(x: int, y: int) -> None:
    """Append one new box for the current frame straight to disk."""
    frame_idx = int(state["idx"])
    append_rows(
        [
            {
                "x": x,
                "y": y,
                "w": BOX_SIZE,
                "h": BOX_SIZE,
                "frame": frame_idx,
            }
        ]
    )
    update_stats()


def rewrite_csv_without(frame_idx: int, keep: list[tuple[int, int]]) -> None:
    """
    Used only by UNDO: remove every row for *frame_idx* then re-append the
    surviving ‘keep’ boxes for that frame — all other frames stay untouched.
    """
    csv_path = get_csv_path()
    if not csv_path.exists():
        return

    # Read everything except rows for this frame
    rows: list[dict] = []
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                f = int(r["frame"])
            except Exception:
                continue
            if f != frame_idx:
                rows.append(r)

    # Add the retained boxes for this frame
    for (x, y) in keep:
        rows.append(
            {
                "x": x,
                "y": y,
                "w": BOX_SIZE,
                "h": BOX_SIZE,
                "frame": frame_idx,
            }
        )

    # Atomically rewrite
    tmp = csv_path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["x", "y", "w", "h", "frame"])
        wr.writeheader()
        wr.writerows(rows)
    tmp.replace(csv_path)
    update_stats()


def find_nearest_brightest(gray: np.ndarray, cx: int, cy: int) -> tuple[int, int]:
    h, w = gray.shape
    x0, y0 = max(cx - SEARCH_RADIUS, 0), max(cy - SEARCH_RADIUS, 0)
    x1, y1 = min(cx + SEARCH_RADIUS, w - 1), min(cy + SEARCH_RADIUS, h - 1)
    sub = gray[y0 : y1 + 1, x0 : x1 + 1]
    maxv = sub.max()
    ys, xs = np.where(sub == maxv)
    best = min(
        zip(xs, ys),
        key=lambda c: (c[0] + x0 - cx) ** 2 + (c[1] + y0 - cy) ** 2,
    )
    return best[0] + x0, best[1] + y0


def draw_box(x: int, y: int) -> None:
    r = BOX_SIZE // 2
    canvas.create_rectangle(
        x - r,
        y - r,
        x + r,
        y + r,
        outline="red",
        width=1,
    )


def _cache_put(idx: int, frame_rgb: np.ndarray) -> None:
    """Store a decoded RGB frame in a small LRU cache."""
    cache: dict[int, np.ndarray] = state["frame_cache"]  # type: ignore[assignment]
    order: list[int] = state["cache_order"]  # type: ignore[assignment]
    cache[idx] = frame_rgb
    if idx in order:
        order.remove(idx)
    order.append(idx)
    # Trim cache to max size
    while len(order) > MAX_CACHE_FRAMES:
        old = order.pop(0)
        cache.pop(old, None)


def _cache_get(idx: int) -> np.ndarray | None:
    cache: dict[int, np.ndarray] = state["frame_cache"]  # type: ignore[assignment]
    frame = cache.get(idx)
    if frame is None:
        return None
    # Touch in order list (simple LRU)
    order: list[int] = state["cache_order"]  # type: ignore[assignment]
    if idx in order:
        order.remove(idx)
    order.append(idx)
    return frame


def _read_frame(idx: int) -> np.ndarray | None:
    """
    Efficiently read frame *idx*.

    For sequential navigation (idx == last_decoded + 1) we just call
    cap.read() without a random seek, which is much faster for large videos.
    For other jumps (initial load, big skips, moving backwards), we fall
    back to cap.set(..., idx) then read once. Recently-visited frames are
    also cached in RAM, so stepping backwards within a small window is fast.
    """
    # First try cache for quick backwards navigation.
    cached = _cache_get(idx)
    if cached is not None:
        return cached

    cap = state.get("cap")
    if cap is None or not cap.isOpened():  # type: ignore[union-attr]
        video_path = state.get("video_path")
        if not video_path:
            return None
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not reopen video {video_path}")
            state["cap"] = None
            return None
        state["cap"] = cap
        state["cap_pos"] = None

    cap_pos = state.get("cap_pos")

    if isinstance(cap_pos, int) and idx == cap_pos + 1:
        # Fast path: next frame in sequence
        ok, frame = cap.read()  # type: ignore[call-arg]
        if not ok or frame is None:
            return None
        state["cap_pos"] = idx
    else:
        # Random access (first frame, large jump, or moving backwards)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # type: ignore[call-arg]
        ok, frame = cap.read()  # type: ignore[call-arg]
        if not ok or frame is None:
            return None
        state["cap_pos"] = idx

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _cache_put(idx, frame_rgb)
    return frame_rgb


# ───────── display logic ─────────
def show_frame() -> None:
    total = int(state.get("total_frames") or 0)
    if total <= 0 or not state.get("video_path"):
        canvas.config(width=100, height=100)
        canvas.delete("all")
        fname_var.set("")
        prog_var.set("")
        return

    idx = int(state["idx"])
    full = _read_frame(idx)
    if full is None:
        canvas.config(width=100, height=100)
        canvas.delete("all")
        fname_var.set("Could not read frame")
        prog_var.set(f"Frame {idx + 1}/{total}")
        return

    h, w = full.shape[:2]

    # Fit to 95 % of monitor
    sw, sh = int(root.winfo_screenwidth() * 0.95), int(
        root.winfo_screenheight() * 0.95
    )
    scale = min(sw / w, sh / h)
    disp_w, disp_h = int(w * scale), int(h * scale)

    disp = cv2.resize(
        full,
        (disp_w, disp_h),
        interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA,
    )

    state.update(current_np=full, scale=scale)
    state["hover_id"] = None  # reset hover label on new frame

    canvas.config(width=disp_w, height=disp_h)
    canvas.delete("all")
    state["photo"] = ImageTk.PhotoImage(Image.fromarray(disp))
    canvas.create_image(0, 0, image=state["photo"], anchor=tk.NW)

    boxes: dict[int, list[tuple[int, int]]] = state["boxes"]  # type: ignore[assignment]
    for (x, y) in boxes.get(idx, []):
        draw_box(int(x * scale), int(y * scale))

    video_path = state.get("video_path")
    fname_var.set(f"{_basename(video_path)}  (frame {idx})")  # type: ignore[arg-type]
    prog_var.set(f"Frame {idx + 1}/{total}")
    canvas.focus_set()


# ───────── hover helper ─────────
def hover_readout(event) -> None:
    """Show R,G,B and luminance of brightest pixel in 10×10 patch."""
    total = int(state.get("total_frames") or 0)
    if total <= 0 or not state.get("video_path"):
        return
    s = state["scale"]
    ox, oy = int(event.x / s), int(event.y / s)
    frame = state["current_np"]
    if frame is None:
        return
    h, w = frame.shape[:2]
    if not (0 <= ox < w and 0 <= oy < h):
        return

    half = BOX_SIZE // 2
    x0, y0 = max(ox - half, 0), max(oy - half, 0)
    x1, y1 = min(ox + half, w - 1), min(oy + half, h - 1)
    patch = frame[y0 : y1 + 1, x0 : x1 + 1]  # RGB

    if patch.size == 0:
        return

    lumin = (
        0.299 * patch[:, :, 0]
        + 0.587 * patch[:, :, 1]
        + 0.114 * patch[:, :, 2]
    )
    idx = np.unravel_index(np.argmax(lumin), lumin.shape)
    r, g, b = patch[idx]
    intensity = int(round(lumin[idx]))

    # Create or update hover label
    text = f"{r},{g},{b}  {intensity}"
    if state["hover_id"] is None:
        state["hover_id"] = canvas.create_text(
            event.x + 10,
            event.y + 10,
            text=text,
            fill="white",
            anchor=tk.NW,
            font=("TkDefaultFont", 8),
        )
    else:
        canvas.coords(state["hover_id"], event.x + 10, event.y + 10)
        canvas.itemconfig(state["hover_id"], text=text)


def clear_hover(_event=None):
    """Remove hover label when leaving canvas."""
    if state["hover_id"] is not None:
        canvas.delete(state["hover_id"])
        state["hover_id"] = None


# ───────── callbacks ─────────
def on_click(event) -> None:
    total = int(state.get("total_frames") or 0)
    if total <= 0 or not state.get("video_path"):
        return
    s = state["scale"]
    ox_img, oy_img = int(event.x / s), int(event.y / s)
    current = state["current_np"]
    if current is None:
        return
    h, w = current.shape[:2]
    if not (0 <= ox_img < w and 0 <= oy_img < h):
        return
    gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    bx, by = find_nearest_brightest(gray, ox_img, oy_img)
    idx = int(state["idx"])
    boxes: dict[int, list[tuple[int, int]]] = state["boxes"]  # type: ignore[assignment]
    boxes.setdefault(idx, []).append((bx, by))
    save_box(bx, by)
    draw_box(int(bx * s), int(by * s))


def prev_frame(event=None):
    if int(state.get("idx", 0)) > 0:
        state["idx"] = int(state["idx"]) - 1
        show_frame()
    return "break"


def next_frame(event=None):
    total = int(state.get("total_frames") or 0)
    if int(state.get("idx", 0)) < total - 1:
        state["idx"] = int(state["idx"]) + 1
        show_frame()
    return "break"


def undo_last(event=None):
    total = int(state.get("total_frames") or 0)
    if total <= 0 or not state.get("video_path"):
        return "break"
    idx = int(state["idx"])
    boxes: dict[int, list[tuple[int, int]]] = state["boxes"]  # type: ignore[assignment]
    if not boxes.get(idx):
        return "break"
    boxes[idx].pop()
    rewrite_csv_without(idx, boxes[idx])
    show_frame()
    return "break"


def browse_video() -> None:
    p = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[
            ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
            ("All files", "*.*"),
        ],
    )
    if p:
        path_var.set(p)
        load_video(p)


def browse_csv_file() -> None:
    p = filedialog.asksaveasfilename(
        initialfile="annotations_video.csv",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
    )
    if p:
        csv_file_var.set(p)


def on_tab_change(_event):
    if nb.index(nb.select()) == 0:
        update_stats()


# ───────── GUI setup ─────────
root = tk.Tk()
root.title("Firefly Flash Annotator – video v2")

nb = ttk.Notebook(root)
tab_load = ttk.Frame(nb)
tab_annot = ttk.Frame(nb)
nb.add(tab_load, text="Load Video")
nb.add(tab_annot, text="Annotate")
nb.pack(fill=tk.BOTH, expand=True)
nb.bind("<<NotebookTabChanged>>", on_tab_change)

# --- Header inside Annotate tab (filename + progress) ---
header = ttk.Frame(tab_annot)
header.pack(anchor="w", fill=tk.X, padx=4, pady=(4, 0))

fname_var = tk.StringVar()
prog_var = tk.StringVar()
ttk.Label(
    header,
    textvariable=fname_var,
    font=("TkDefaultFont", 10, "bold"),
).pack(side=tk.LEFT)
ttk.Label(header, textvariable=prog_var).pack(side=tk.LEFT, padx=10)

# --- Canvas (resized per frame) ---
canvas = tk.Canvas(tab_annot, bg="black", highlightthickness=0, cursor="crosshair")
canvas.pack()
canvas.bind("<Button-1>", on_click)
canvas.bind("<Left>", prev_frame)
canvas.bind("<Right>", next_frame)
canvas.bind("<Motion>", hover_readout)
canvas.bind("<Leave>", clear_hover)

# --- Load-tab widgets ---
path_var = tk.StringVar(value=DEFAULT_VIDEO_PATH)
csv_file_var = tk.StringVar(value=DEFAULT_CSV_FILE)

ttk.Label(tab_load, text="Video file:").grid(
    row=0,
    column=0,
    sticky="w",
    padx=4,
    pady=4,
)
ttk.Entry(tab_load, textvariable=path_var, width=60).grid(
    row=0,
    column=1,
    padx=4,
    pady=4,
)
ttk.Button(tab_load, text="Browse…", command=browse_video).grid(
    row=0,
    column=2,
    padx=4,
    pady=4,
)

ttk.Label(tab_load, text="CSV file:").grid(
    row=1,
    column=0,
    sticky="w",
    padx=4,
    pady=4,
)
ttk.Entry(tab_load, textvariable=csv_file_var, width=60).grid(
    row=1,
    column=1,
    padx=4,
    pady=4,
)
ttk.Button(tab_load, text="Browse…", command=browse_csv_file).grid(
    row=1,
    column=2,
    padx=4,
    pady=4,
)

stats_var = tk.StringVar(value="No video loaded.")
ttk.Label(tab_load, textvariable=stats_var, justify="left").grid(
    row=2,
    column=0,
    columnspan=3,
    sticky="w",
    padx=4,
    pady=8,
)

# --- Global keybinds ---
root.bind_all("<space>", undo_last)

# Preload defaults if video exists
if Path(DEFAULT_VIDEO_PATH).exists():
    load_video(DEFAULT_VIDEO_PATH)

root.mainloop()
