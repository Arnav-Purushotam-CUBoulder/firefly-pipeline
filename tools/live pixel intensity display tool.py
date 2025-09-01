#!/usr/bin/env python3
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image

# ───── GLOBAL CONFIG ─────────────────────────────────────────────────────────
DEFAULT_IMAGE_PATH = '/Users/arnavps/Desktop/New DL project data to transfer to external disk/orc pipeline frontalis only inference data/stage9 validation/val_9k-14k_frontalis_clip/thr_10.0px/crops/FP_t000064_x797_y901_conf0.9990_max239_area61.png'
DEFAULT_THRESHOLD  = 20          # initial luminance threshold
PIXEL_SCALE        = 20          # preferred per-pixel block size
MAX_VIEW_SIZE      = 600         # hard cap on displayed width/height

# Intensity formula
def compute_luminance(r, g, b):
    return round(0.299 * r + 0.587 * g + 0.114 * b, 2)

class PixelIntensityViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Live Pixel Intensity Viewer")
        # state
        self.threshold  = DEFAULT_THRESHOLD
        self.image_path = None
        self.has_image  = False
        self.orig_w = self.orig_h = None

        # Notebook
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        # ── Tab 1: Viewer ───────────────────────────────────────────────────────
        viewer = ttk.Frame(notebook); notebook.add(viewer, text="Viewer")

        # image name / status
        self.name_label = ttk.Label(viewer, text="No image loaded")
        self.name_label.pack(pady=(10,0))

        # canvas
        self.canvas = tk.Canvas(viewer, bg="white")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Motion>", self._on_mouse_move)

        # info
        self.intensity_label = ttk.Label(viewer, text="Hover over the image"); self.intensity_label.pack()
        self.max_label       = ttk.Label(viewer, text="Max luminance: N/A");    self.max_label.pack()
        self.count_label     = ttk.Label(viewer, text=f"Pixels > {self.threshold}: N/A")
        self.count_label.pack(pady=(0,10))

        # ── Tab 2: Load Image & Threshold ───────────────────────────────────────
        loader = ttk.Frame(notebook); notebook.add(loader, text="Load Image")
        ttk.Label(loader, text="Image:").pack(anchor="w", padx=10, pady=(10,0))
        btn_frame = ttk.Frame(loader); btn_frame.pack(fill="x", pady=5, padx=10)
        self.path_label = ttk.Label(btn_frame, text=DEFAULT_IMAGE_PATH or "(none)", anchor="w")
        self.path_label.pack(side="left", fill="x", expand=True)
        ttk.Button(btn_frame, text="Browse…", command=self._browse_image).pack(side="right")

        thresh_frame = ttk.Frame(loader); thresh_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(thresh_frame, text="Intensity threshold:").pack(side="left")
        self.threshold_entry = ttk.Entry(thresh_frame, width=6)
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.pack(side="left", padx=(5,0))
        ttk.Button(thresh_frame, text="Apply", command=self._apply_threshold).pack(side="left", padx=10)

        # load default (optional)
        if DEFAULT_IMAGE_PATH:
            self._load_image(DEFAULT_IMAGE_PATH)

    # ───────────────────────── UI Actions ──────────────────────────────────────
    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.path_label.config(text=path)
        self._load_image(path)

    def _apply_threshold(self):
        try:
            t = float(self.threshold_entry.get())
            self.threshold = t
            self._update_stats()
        except ValueError:
            messagebox.showerror("Invalid Threshold", "Please enter a numeric threshold.")

    # ───────────────────────── Core Logic ──────────────────────────────────────
    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to open image:\n{e}")
            self.has_image = False
            self.name_label.config(text="No image loaded")
            return

        self.image_path = path
        self.orig_w, self.orig_h = img.size

        # 1) Downsize to fit within MAX_VIEW_SIZE (keep aspect ratio)
        disp_w, disp_h = self.orig_w, self.orig_h
        if disp_w > MAX_VIEW_SIZE or disp_h > MAX_VIEW_SIZE:
            ratio = min(MAX_VIEW_SIZE / disp_w, MAX_VIEW_SIZE / disp_h)
            disp_w = max(1, int(disp_w * ratio))
            disp_h = max(1, int(disp_h * ratio))
            img = img.resize((disp_w, disp_h), Image.NEAREST)

        # 2) Choose integer scale so final canvas ≤ MAX_VIEW_SIZE
        self.img_w, self.img_h = img.size
        self.scale = max(1, min(
            PIXEL_SCALE,
            MAX_VIEW_SIZE // self.img_w if self.img_w else 1,
            MAX_VIEW_SIZE // self.img_h if self.img_h else 1
        ))

        self.pixels = img.load()
        self.canvas.config(width=self.img_w * self.scale, height=self.img_h * self.scale)
        self.has_image = True

        self._draw_pixels()
        name = os.path.basename(self.image_path)
        self.name_label.config(
            text=f"{name}  (orig {self.orig_w}×{self.orig_h} → disp {self.img_w}×{self.img_h}, scale×{self.scale})"
        )
        self._update_stats()

    def _draw_pixels(self):
        self.canvas.delete("all")
        if not self.has_image:
            return
        for y in range(self.img_h):
            for x in range(self.img_w):
                r, g, b = self.pixels[x, y]
                color = f"#{r:02x}{g:02x}{b:02x}"
                x0, y0 = x * self.scale, y * self.scale
                self.canvas.create_rectangle(
                    x0, y0, x0 + self.scale, y0 + self.scale, fill=color, outline="black"
                )

    def _on_mouse_move(self, event):
        if not self.has_image:
            return
        px = min(self.img_w - 1, max(0, event.x // self.scale))
        py = min(self.img_h - 1, max(0, event.y // self.scale))
        r, g, b = self.pixels[px, py]
        lum = compute_luminance(r, g, b)
        self.intensity_label.config(
            text=f"Pixel ({px},{py}) → R={r},G={g},B={b}  Lum={lum}"
        )

    def _update_stats(self):
        if not self.has_image:
            self.max_label.config(text="Max luminance: N/A")
            self.count_label.config(text=f"Pixels > {self.threshold}: N/A")
            return
        lums = [
            compute_luminance(*self.pixels[x, y])
            for y in range(self.img_h)
            for x in range(self.img_w)
        ]
        self.max_label.config(text=f"Max luminance: {max(lums)}")
        self.count_label.config(
            text=f"Pixels > {self.threshold}: {sum(v > self.threshold for v in lums)}"
        )

if __name__ == "__main__":
    app = PixelIntensityViewer()
    app.mainloop()
