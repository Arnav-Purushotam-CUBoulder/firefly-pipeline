#!/usr/bin/env python3
"""
Firefly-flash annotator  –  v4.7
────────────────────────────────
• Image resizes to 95 % of current monitor while preserving aspect ratio.
• Filename and progress counter are shown in a header above the canvas,
  leaving the full area below for the picture.
• Boxes are always stored in ORIGINAL coordinates.
• Keys:  ← / →  navigate, space = undo last, click = add 10×10 box.
• Hover shows **R,G,B** *and* a single-channel intensity next to cursor.
      – intensity = round(0.299 R + 0.587 G + 0.114 B)

Changes since v4.6-p
  ✱ Keeps session-persistence fix (basename comparison).
  ✱ Adds real-time hover read-out of the brightest pixel in the 10×10
    window centred on the cursor.
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
DEFAULT_IMAGE_FOLDER = (
    '/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/raw frames/20240606_cam1_GS010064')
DEFAULT_CSV_FILE = (
'/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/raw frames/20240606_cam1_GS010064.csv')

BOX_SIZE      = 10                    # also used as hover patch size
SEARCH_RADIUS = 30

state: dict[str, object] = {
    "folder":      None,
    "frames":      [],
    "idx":         0,
    "boxes":       {},   # {frame_name: [(x,y), …]} in original pixels
    "photo":       None,
    "current_np":  None, # RGB numpy array of full-res frame
    "scale":       1.0,
    "hover_id":    None, # canvas text item for hover read-out
}

# ───────── helper functions ─────────
def get_csv_path() -> Path:
    return Path(csv_file_var.get()).expanduser()

def append_rows(rows: list[dict]) -> None:
    """append list-of-dicts to CSV, creating the file + header if needed."""
    if not rows:
        return
    csv_path = get_csv_path()
    newfile  = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=["x", "y", "w", "h", "frame"])
        if newfile:
            wr.writeheader()
        wr.writerows(rows)
        fh.flush()          # ensure on-disk immediately

def update_stats() -> None:
    if not state["frames"]:
        stats_var.set("No folder loaded.")
        return
    total       = len(state["frames"])
    annotated   = sum(1 for v in state["boxes"].values() if v)
    total_boxes = sum(len(v) for v in state["boxes"].values())
    avg         = total_boxes / annotated if annotated else 0
    remain      = total - annotated
    pct         = annotated / total * 100
    stats_var.set(
        f"Images annotated: {annotated}/{total}  ({pct:.1f}%)\n"
        f"Total boxes: {total_boxes}\n"
        f"Average boxes / annotated image: {avg:.2f}\n"
        f"Images remaining: {remain}"
    )

def load_folder(p: str | Path) -> None:
    p = Path(p).expanduser()
    if not p.exists():
        messagebox.showerror("Error", f"Folder {p} not found")
        return

    imgs = sorted(
        f for f in p.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not imgs:
        messagebox.showerror("Error", "No image files found")
        return

    boxes: dict[str, list[tuple[int, int]]] = {}
    csv_path = get_csv_path()
    if csv_path.exists():
        with csv_path.open(newline='') as fh:
            for r in csv.DictReader(fh):
                key = _basename(r["frame"])
                boxes.setdefault(key, []).append((int(r["x"]), int(r["y"])))

    start = next((i for i, f in enumerate(imgs) if f.name not in boxes),
                 len(imgs) - 1)

    state.update(folder=p, frames=imgs, idx=start, boxes=boxes)
    show_frame()
    update_stats()

def save_box(x: int, y: int) -> None:
    """append one new box for the current frame straight to disk."""
    append_rows([{
        "x": x, "y": y, "w": BOX_SIZE, "h": BOX_SIZE,
        "frame": state["frames"][state["idx"]].name
    }])
    update_stats()

def rewrite_csv_without(frame_name: str,
                        keep: list[tuple[int, int]]) -> None:
    """
    used only by UNDO: remove every row for *frame_name* then re-append the
    surviving ‘keep’ boxes for that frame — all other frames stay untouched.
    """
    csv_path = get_csv_path()
    if not csv_path.exists():
        return

    # read everything except rows for this frame
    with csv_path.open(newline='') as fh:
        rows = [r for r in csv.DictReader(fh)
                if _basename(r["frame"]) != frame_name]

    # add the retained boxes for this frame
    for (x, y) in keep:
        rows.append({"x": x, "y": y, "w": BOX_SIZE,
                     "h": BOX_SIZE, "frame": frame_name})

    # atomically rewrite
    tmp = csv_path.with_suffix(".tmp")
    with tmp.open("w", newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=["x", "y", "w", "h", "frame"])
        wr.writeheader(); wr.writerows(rows)
    tmp.replace(csv_path)
    update_stats()

def find_nearest_brightest(gray: np.ndarray,
                           cx: int, cy: int) -> tuple[int, int]:
    h, w   = gray.shape
    x0, y0 = max(cx - SEARCH_RADIUS, 0), max(cy - SEARCH_RADIUS, 0)
    x1, y1 = min(cx + SEARCH_RADIUS, w - 1), min(cy + SEARCH_RADIUS, h - 1)
    sub    = gray[y0:y1 + 1, x0:x1 + 1]
    maxv   = sub.max()
    ys, xs = np.where(sub == maxv)
    best   = min(
        zip(xs, ys),
        key=lambda c: (c[0] + x0 - cx) ** 2 + (c[1] + y0 - cy) ** 2
    )
    return best[0] + x0, best[1] + y0

def draw_box(x: int, y: int) -> None:
    r = BOX_SIZE // 2
    canvas.create_rectangle(
        x - r, y - r, x + r, y + r,
        outline="red", width=1
    )

# ───────── display logic ─────────
def show_frame() -> None:
    if not state["frames"]:
        canvas.config(width=100, height=100)
        canvas.delete("all")
        fname_var.set("")
        prog_var.set("")
        return

    img_path = state["frames"][state["idx"]]
    full     = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w     = full.shape[:2]

    # fit to 95 % of monitor
    sw, sh   = (int(root.winfo_screenwidth()*0.95),
                int(root.winfo_screenheight()*0.95))
    scale    = min(sw / w, sh / h)
    disp_w, disp_h = int(w * scale), int(h * scale)

    disp = cv2.resize(
        full, (disp_w, disp_h),
        interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA
    )

    state.update(current_np=full, scale=scale)
    state["hover_id"] = None            # reset hover label on new frame

    canvas.config(width=disp_w, height=disp_h)
    canvas.delete("all")
    state["photo"] = ImageTk.PhotoImage(Image.fromarray(disp))
    canvas.create_image(0, 0, image=state["photo"], anchor=tk.NW)

    for (x, y) in state["boxes"].get(img_path.name, []):
        draw_box(int(x * scale), int(y * scale))

    fname_var.set(img_path.name)
    prog_var.set(f"Frame {state['idx'] + 1}/{len(state['frames'])}")
    canvas.focus_set()

# ───────── hover helper ─────────
def hover_readout(event) -> None:
    """Show R,G,B and luminance of brightest pixel in 10×10 patch."""
    if not state["frames"]:
        return
    s = state["scale"]
    ox, oy = int(event.x / s), int(event.y / s)
    frame  = state["current_np"]
    if frame is None:
        return
    h, w = frame.shape[:2]
    if not (0 <= ox < w and 0 <= oy < h):
        return

    half = BOX_SIZE // 2
    x0, y0 = max(ox - half, 0), max(oy - half, 0)
    x1, y1 = min(ox + half, w - 1), min(oy + half, h - 1)
    patch  = frame[y0:y1 + 1, x0:x1 + 1]        # RGB

    if patch.size == 0:
        return

    lumin = (0.299 * patch[:, :, 0] +
             0.587 * patch[:, :, 1] +
             0.114 * patch[:, :, 2])
    idx   = np.unravel_index(np.argmax(lumin), lumin.shape)
    r, g, b = patch[idx]
    intensity = int(round(lumin[idx]))

    # create or update hover label
    text = f"{r},{g},{b}  {intensity}"
    if state["hover_id"] is None:
        state["hover_id"] = canvas.create_text(
            event.x + 10, event.y + 10,
            text=text, fill="white", anchor=tk.NW,
            font=("TkDefaultFont", 8)
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
    if not state["frames"]:
        return
    s = state["scale"]
    ox_img, oy_img = int(event.x / s), int(event.y / s)
    h, w           = state["current_np"].shape[:2]
    if not (0 <= ox_img < w and 0 <= oy_img < h):
        return
    gray = cv2.cvtColor(state["current_np"], cv2.COLOR_RGB2GRAY)
    bx, by = find_nearest_brightest(gray, ox_img, oy_img)
    fname  = state["frames"][state["idx"]].name
    state["boxes"].setdefault(fname, []).append((bx, by))
    save_box(bx, by)
    draw_box(int(bx * s), int(by * s))

def prev_frame(event=None):
    if state["idx"] > 0:
        state["idx"] -= 1
        show_frame()
    return "break"

def next_frame(event=None):
    if state["idx"] < len(state["frames"]) - 1:
        state["idx"] += 1
        show_frame()
    return "break"

def undo_last(event=None):
    if not state["frames"]:
        return "break"
    fname = state["frames"][state["idx"]].name
    if not state["boxes"].get(fname):
        return "break"
    state["boxes"][fname].pop()
    rewrite_csv_without(fname, state["boxes"][fname])
    show_frame()
    return "break"

def browse_images() -> None:
    p = filedialog.askdirectory()
    if p:
        path_var.set(p)
        load_folder(p)

def browse_csv_file() -> None:
    p = filedialog.asksaveasfilename(
        initialfile="annotations_small.csv",
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
root.title("Firefly Flash Annotator")

nb = ttk.Notebook(root)
tab_load  = ttk.Frame(nb)
tab_annot = ttk.Frame(nb)
nb.add(tab_load,  text="Load Images")
nb.add(tab_annot, text="Annotate")
nb.pack(fill=tk.BOTH, expand=True)
nb.bind("<<NotebookTabChanged>>", on_tab_change)

# --- Header inside Annotate tab (filename + progress) ---
header = ttk.Frame(tab_annot)
header.pack(anchor="w", fill=tk.X, padx=4, pady=(4, 0))

fname_var = tk.StringVar()
prog_var  = tk.StringVar()
ttk.Label(header, textvariable=fname_var,
          font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
ttk.Label(header, textvariable=prog_var).pack(side=tk.LEFT, padx=10)

# --- Canvas (resized per frame) ---
canvas = tk.Canvas(tab_annot, bg="black",
                   highlightthickness=0, cursor="crosshair")
canvas.pack()
canvas.bind("<Button-1>", on_click)
canvas.bind("<Left>",  prev_frame)
canvas.bind("<Right>", next_frame)
canvas.bind("<Motion>", hover_readout)
canvas.bind("<Leave>",  clear_hover)

# --- Load-tab widgets ---
path_var     = tk.StringVar(value=DEFAULT_IMAGE_FOLDER)
csv_file_var = tk.StringVar(value=DEFAULT_CSV_FILE)

ttk.Label(tab_load, text="Image folder:")\
    .grid(row=0, column=0, sticky="w", padx=4, pady=4)
ttk.Entry(tab_load, textvariable=path_var, width=60)\
    .grid(row=0, column=1, padx=4, pady=4)
ttk.Button(tab_load, text="Browse…", command=browse_images)\
    .grid(row=0, column=2, padx=4, pady=4)

ttk.Label(tab_load, text="CSV file:")\
    .grid(row=1, column=0, sticky="w", padx=4, pady=4)
ttk.Entry(tab_load, textvariable=csv_file_var, width=60)\
    .grid(row=1, column=1, padx=4, pady=4)
ttk.Button(tab_load, text="Browse…", command=browse_csv_file)\
    .grid(row=1, column=2, padx=4, pady=4)

stats_var = tk.StringVar(value="No folder loaded.")
ttk.Label(tab_load, textvariable=stats_var, justify="left")\
    .grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=8)

# --- Global keybinds ---
root.bind_all("<space>", undo_last)

# preload defaults if folder exists
if Path(DEFAULT_IMAGE_FOLDER).exists():
    load_folder(DEFAULT_IMAGE_FOLDER)

root.mainloop()
