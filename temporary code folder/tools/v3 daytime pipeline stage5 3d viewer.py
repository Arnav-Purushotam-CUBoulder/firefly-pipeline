#!/usr/bin/env python3
"""
Stage 5 3D Viewer – PyVista
────────────────────────────

Interactive viewer for the 3D detection geometry exported by
`stage5_3d_render.py` in the daytime v3 pipeline.

Usage (from terminal, inside your RA01 env)
------------------------------------------

  • To open a single .vtp file:

      python "test1/tools/stage5 3d viewer.py" /path/to/file.vtp

  • To browse all .vtp files in a folder:

      python "test1/tools/stage5 3d viewer.py" /path/to/stage5_3d_render/<video_stem>

    Then use Left / Right arrow keys in the 3D window to move between
    blocks while keeping your current camera / zoom.

  • If no path is provided on the command line, a file-open dialog
    will pop up to let you pick a .vtp file.

Requirements
------------
  • pyvista (and its VTK dependency) installed in the active Python env.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import tkinter as tk
from tkinter import filedialog, messagebox


# ───────── Global configuration ─────────
# Set this to a specific .vtp file or a folder containing .vtp files
# to avoid passing the path on the command line every time.
# If left empty (""), the script will fall back to CLI args or a file dialog.
VTP_PATH: str = '/Users/arnavps/Desktop/RA inference data/v3 daytime pipeline inference data/stage5_3d_render/20240606_cam1_GS010064/20240606_cam1_GS010064_block_000000-000999.vtp'


def _ensure_pyvista():
    try:
        import pyvista as pv  # type: ignore[import]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "This viewer requires pyvista.\n"
            "Install it in your current environment, e.g.:\n\n"
            "  python3 -m pip install pyvista\n"
        ) from e
    return pv


def _pick_file_via_dialog() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    try:
        p = filedialog.askopenfilename(
            title="Select Stage5 3D file (.vtp)",
            filetypes=[("VTK PolyData files", "*.vtp"), ("All files", "*.*")],
        )
        if not p:
            return None
        return Path(p).expanduser()
    finally:
        root.destroy()


def _collect_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() != ".vtp":
            raise ValueError(f"File does not look like a VTK PolyData (.vtp): {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Path is neither file nor directory: {path}")

    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".vtp")
    if not files:
        raise FileNotFoundError(f"No .vtp files found in directory: {path}")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive viewer for Stage5 3D detection files (.vtp).")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="VTK .vtp file or directory containing .vtp files. If omitted, a file dialog will open.",
    )
    args = parser.parse_args(argv)

    if args.path is None:
        if VTP_PATH:
            path = Path(VTP_PATH).expanduser()
        else:
            chosen = _pick_file_via_dialog()
            if chosen is None:
                return 0
            path = chosen
    else:
        path = Path(args.path).expanduser()

    pv = _ensure_pyvista()

    try:
        files = _collect_files(path)
    except Exception as e:
        # Use a simple Tk popup for user-friendly error if Tk is available.
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Stage5 3D Viewer", str(e))
            root.destroy()
        except Exception:
            print(f"Error: {e}")
        return 1

    print(f"[stage5-viewer] Loaded {len(files)} file(s).")

    def _add_mesh_with_colors(plotter, data, name: str):
        # If 'rejected' data is present, use it to color points:
        #   1 => rejected trajectories (blue)
        #   0 => kept trajectories (red)
        #
        # (Legacy) If 'uniform' data is present, use it to color points:
        #   1 => uniform-brightness trajectories (blue)
        #   0 => varying brightness (red)
        scalars_name = None
        if "rejected" in data.point_data:
            scalars_name = "rejected"
        elif "rejected" in data.cell_data:
            scalars_name = "rejected"
        elif "uniform" in data.point_data:
            scalars_name = "uniform"
        elif "uniform" in data.cell_data:
            scalars_name = "uniform"

        if scalars_name is not None:
            plotter.add_mesh(
                data,
                scalars=scalars_name,
                cmap=["red", "blue"],  # 0 -> red, 1 -> blue
                categories=True,
                smooth_shading=True,
            )
        else:
            plotter.add_mesh(data, color="red", smooth_shading=True)
        plotter.add_axes()
        plotter.add_text(name, font_size=10)

    # Single-file viewer (no navigation) is straightforward
    if len(files) == 1:
        data = pv.read(str(files[0]))
        plotter = pv.Plotter()
        _add_mesh_with_colors(plotter, data, files[0].name)
        plotter.show(title=f"Stage5 3D Viewer – {files[0].name}")
        return 0

    # Multi-file viewer with Left/Right navigation
    idx = 0
    data = pv.read(str(files[idx]))
    plotter = pv.Plotter()
    _add_mesh_with_colors(plotter, data, f"{files[idx].name}   ({idx+1}/{len(files)})")

    def _update(delta: int) -> None:
        nonlocal idx
        if not files:
            return
        idx = (idx + delta) % len(files)
        try:
            new_data = pv.read(str(files[idx]))
        except Exception as e:
            print(f"[stage5-viewer] Failed to load {files[idx]}: {e}")
            return

        # Preserve current camera position when switching
        cam = plotter.camera_position
        plotter.clear()
        _add_mesh_with_colors(plotter, new_data, f"{files[idx].name}   ({idx+1}/{len(files)})")
        plotter.camera_position = cam
        plotter.render()

    plotter.add_key_event("Right", lambda: _update(+1))
    plotter.add_key_event("Left", lambda: _update(-1))
    plotter.add_text("Use Left/Right arrows to switch blocks", position="lower_left", font_size=8)

    plotter.show(title="Stage5 3D Viewer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
