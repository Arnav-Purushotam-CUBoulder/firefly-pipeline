#!/usr/bin/env python3
"""
IoU Crop Explorer – TP / FP / FN visualiser with pixel-intensity read-out & zoom.

Controls
--------
← / →              step through cases
Mouse-wheel / pinch   zoom current image (full or crop)
Radio buttons         Full-frame vs Crop view
Tabs                  IoU threshold  →  TP / FP / FN
Open Folder…          jump to another IOU_CROPS_ROOT

Author: 2025
"""

import os, re, argparse, tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# ─── change to your typical path ───
DEFAULT_IOU_ROOT = (
    '/Volumes/DL Project SSD/back subtracted method data/resnet forresti data/main validation script output data for observation/>50 threshold resnet18 run'
)

# ■■■ utility helpers ■■■ ------------------------------------------------------
def pixel_blowup(img, factor=8):
    """Upscale very small crops with nearest-neighbour so pixels look blocky."""
    resample = getattr(Image, "NEAREST", Image.Resampling.NEAREST)
    return img.resize((img.width * factor, img.height * factor), resample)


def brightest_intensity(img):
    """Return single-channel intensity of the brightest pixel (0-255)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return int(np.round(lum.max()))


def resize_for_display(img, max_w=1150, max_h=700):
    # 1) enlarge tiny crops for easy viewing
    if min(img.width, img.height) < 64:
        img = pixel_blowup(img)
    # 2) shrink overly large images to fit window
    ratio = max(img.width / max_w, img.height / max_h, 1)
    if ratio > 1:
        resample = getattr(Image, "LANCZOS", Image.Resampling.LANCZOS)
        img = img.resize((int(img.width / ratio), int(img.height / ratio)), resample)
    return img


def pick_img(case_dict, *bases):
    """Return first existing image path among candidate base-names."""
    for base in bases:
        for ext in (".png", ".jpg", ".jpeg"):
            p = case_dict.get(base + ext)
            if p and p.exists():
                return p
    return None


# ■■■ main Tk app ■■■ -----------------------------------------------------------
class IoUCropExplorer(tk.Tk):
    ZOOM_UP = 1.25
    ZOOM_DN = 1 / 1.25

    def __init__(self, root_dir: str):
        super().__init__()
        self.title("IoU Crop Explorer")
        self.geometry("1280x800")

        # state
        self.root_dir = Path(root_dir)
        self.view_mode = tk.StringVar(value="full")  # full | crop
        self.cases, self.index, self.labels = {}, {}, {}
        self.zoom_base = None      # original PIL.Image for current view
        self.zoom_factor = 1.0
        self.current_label = None
        self.current_info = ""

        # global key-bindings
        self.bind("<Left>", self.prev_case)
        self.bind("<Right>", self.next_case)

        # top bar --------------------------------------------------
        topbar = ttk.Frame(self)
        topbar.pack(side="top", fill="x")
        ttk.Label(topbar, text="View:").pack(side="left", padx=(10, 2))
        for mode in ("full", "crop"):
            ttk.Radiobutton(
                topbar,
                text=mode.capitalize(),
                variable=self.view_mode,
                value=mode,
                command=self.refresh_current,
            ).pack(side="left")
        ttk.Button(topbar, text="Open Folder…", command=self.pick_root).pack(
            side="right", padx=10
        )

        # tab notebooks -------------------------------------------
        self.nb_iou = ttk.Notebook(self)
        self.nb_iou.pack(fill="both", expand=True)
        self.build_tabs()
        self.nb_iou.bind("<<NotebookTabChanged>>", lambda e: self.refresh_current())

    # ─── folder picker ──────────────────────────────────────────
    def pick_root(self):
        new_path = filedialog.askdirectory(title="Select IoU_CROPS_ROOT")
        if new_path:
            self.root_dir = Path(new_path)
            self.rebuild()

    # ─── rebuild whole UI when root changes ─────────────────────
    def rebuild(self):
        for child in self.nb_iou.winfo_children():
            child.destroy()
        self.cases.clear()
        self.index.clear()
        self.labels.clear()
        self.build_tabs()

    # ─── create IoU tabs and TP/FP/FN inner tabs ────────────────
    def build_tabs(self):
        iou_dirs = sorted(
            [d for d in self.root_dir.glob("IoU_*") if d.is_dir()],
            key=lambda p: float(p.name.split("_")[1]),
        )
        pat = re.compile(r"IoU_(\d\.\d+)")
        for iou_dir in iou_dirs:
            thr = pat.match(iou_dir.name).group(1)
            outer = ttk.Frame(self.nb_iou)
            self.nb_iou.add(outer, text=thr)
            inner = ttk.Notebook(outer)
            inner.pack(fill="both", expand=True)
            for ctype in ("TP", "FP", "FN"):
                tab = ttk.Frame(inner)
                inner.add(tab, text=ctype)
                lbl = ttk.Label(tab, compound="bottom")  # allow text under image
                lbl.pack(fill="both", expand=True)
                self.labels[(thr, ctype)] = lbl
                self.cases[(thr, ctype)] = self.scan_cases(iou_dir / ctype)
                self.index[(thr, ctype)] = 0
                self.show_case(thr, ctype)

    @staticmethod
    def scan_cases(path: Path):
        return (
            [{f.name: f for f in c.iterdir()} for c in sorted(path.iterdir()) if c.is_dir()]
            if path.exists()
            else []
        )

    # ─── helpers to know which tab we're on ─────────────────────
    def current_keys(self):
        thr = self.nb_iou.tab(self.nb_iou.select(), "text")
        inner = self.nb_iou.nametowidget(self.nb_iou.select()).winfo_children()[0]
        ctype = inner.tab(inner.select(), "text")
        return thr, ctype

    # ─── navigation ─────────────────────────────────────────────
    def prev_case(self, *_):
        thr, ctype = self.current_keys()
        if self.index[(thr, ctype)] > 0:
            self.index[(thr, ctype)] -= 1
            self.show_case(thr, ctype)

    def next_case(self, *_):
        thr, ctype = self.current_keys()
        if self.index[(thr, ctype)] + 1 < len(self.cases[(thr, ctype)]):
            self.index[(thr, ctype)] += 1
            self.show_case(thr, ctype)

    def refresh_current(self):
        thr, ctype = self.current_keys()
        self.show_case(thr, ctype)

    # ─── zoom handler ───────────────────────────────────────────
    def on_zoom(self, event):
        if not self.zoom_base or not self.current_label:
            return
        delta = event.delta if event.delta else (120 if event.num == 4 else -120)
        factor = self.ZOOM_UP if delta > 0 else self.ZOOM_DN
        new_zoom = max(0.1, min(10.0, self.zoom_factor * factor))
        if abs(new_zoom - self.zoom_factor) < 0.01:
            return
        self.zoom_factor = new_zoom
        resample = getattr(Image, "LANCZOS", Image.Resampling.LANCZOS)
        new_img = self.zoom_base.resize(
            (int(self.zoom_base.width * self.zoom_factor), int(self.zoom_base.height * self.zoom_factor)),
            resample,
        )
        photo = ImageTk.PhotoImage(new_img)
        self.current_label.config(image=photo, text=self.current_info, compound="bottom")
        self.current_label.image = photo  # keep reference

    # ─── core display logic ─────────────────────────────────────
    def show_case(self, thr, ctype):
        lbl = self.labels[(thr, ctype)]
        cases = self.cases[(thr, ctype)]
        if not cases:
            lbl.config(text="(no samples)", image="")
            lbl.image = None
            return

        case = cases[self.index[(thr, ctype)]]
        view = self.view_mode.get()
        info = ""

        # Determine which files to use ----------------------------------
        if ctype == "TP":
            pred_crop = pick_img(case, "pred_crop")
            gt_crop = pick_img(case, "gt_crop")
            pred_full = pick_img(case, "pred_full")
            gt_full = pick_img(case, "gt_full")

            if view == "crop" and (pred_crop or gt_crop):
                im_pred = Image.open(pred_crop) if pred_crop else None
                im_gt = Image.open(gt_crop) if gt_crop else None
                pred_int = brightest_intensity(im_pred) if im_pred else "-"
                gt_int = brightest_intensity(im_gt) if im_gt else "-"
                info = f"P: {pred_int}   |   G: {gt_int}"
                im1 = resize_for_display(im_pred) if im_pred else None
                im2 = resize_for_display(im_gt) if im_gt else None
            else:  # full view
                im_pred = Image.open(pred_full) if pred_full else None
                im_gt = Image.open(gt_full) if gt_full else None
                # intensity from crops if available
                pred_int = (
                    brightest_intensity(Image.open(pred_crop))
                    if pred_crop
                    else brightest_intensity(im_pred)
                    if im_pred
                    else "-"
                )
                gt_int = (
                    brightest_intensity(Image.open(gt_crop))
                    if gt_crop
                    else brightest_intensity(im_gt)
                    if im_gt
                    else "-"
                )
                info = f"P: {pred_int}   |   G: {gt_int}"
                im1 = resize_for_display(im_pred) if im_pred else None
                im2 = resize_for_display(im_gt) if im_gt else None

            if not (im1 or im2):
                lbl.config(text="(no matching images)", image="")
                lbl.image = None
                return
            if im1 and im2:
                merged = Image.new("RGB", (im1.width + im2.width + 5, max(im1.height, im2.height)), (255, 255, 255))
                merged.paste(im1, (0, 0))
                merged.paste(im2, (im1.width + 5, 0))
                disp = merged
            else:
                disp = im1 or im2

        elif ctype == "FP":
            img_crop = pick_img(case, "pred_crop")
            img_full = pick_img(case, "pred_full")
            if view == "crop" and img_crop:
                im = Image.open(img_crop)
            else:
                im = Image.open(img_full) if img_full else Image.open(img_crop)
            intensity = brightest_intensity(im)
            info = f"I: {intensity}"
            disp = resize_for_display(im)

        else:  # FN
            img_crop = pick_img(case, "gt_crop")
            img_full = pick_img(case, "gt_full")
            if view == "crop" and img_crop:
                im = Image.open(img_crop)
            else:
                im = Image.open(img_full) if img_full else Image.open(img_crop)
            intensity = brightest_intensity(im)
            info = f"I: {intensity}"
            disp = resize_for_display(im)

        # prepare zoom handling ----------------------------------------
        self.zoom_base = disp.copy()  # keep original (already blown-up tiny crops)
        self.zoom_factor = 1.0
        self.current_label = lbl
        self.current_info = info

        # create Tk image & display ------------------------------------
        photo = ImageTk.PhotoImage(disp)
        lbl.config(image=photo, text=info, compound="bottom")
        lbl.image = photo

        # bind zoom events (re-bind each time so current_label correct)
        lbl.unbind("<MouseWheel>")
        lbl.unbind("<Button-4>")
        lbl.unbind("<Button-5>")
        lbl.bind("<MouseWheel>", self.on_zoom)
        lbl.bind("<Button-4>", self.on_zoom)  # Linux scroll-up
        lbl.bind("<Button-5>", self.on_zoom)  # Linux scroll-down

        # window title update
        self.title(f"IoU {thr} • {ctype} • {self.index[(thr, ctype)] + 1}/{len(cases)}")

    # ─── static helper ends -------------------------------------------


# ■■■ entry point ■■■ ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=DEFAULT_IOU_ROOT, help="Path to IOU_CROPS_ROOT (default set in script)"
    )
    args = parser.parse_args()
    if not os.path.isdir(args.root):
        messagebox.showerror("Error", f"Directory not found:\n{args.root}")
        return
    IoUCropExplorer(args.root).mainloop()


if __name__ == "__main__":
    main()
