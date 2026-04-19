#!/usr/bin/env python3
from __future__ import annotations

"""Repo-wide defaults for annotated video rendering."""

VIDEO_RENDER_BBOX_THICKNESS_PX: int = 1
ENFORCE_SOURCE_VIDEO_RESOLUTION: bool = True


def normalize_video_bbox_thickness(value: int | None = None) -> int:
    """Return a safe bbox thickness, defaulting to the repo standard."""
    if value is None:
        value = VIDEO_RENDER_BBOX_THICKNESS_PX
    try:
        return max(1, int(value))
    except Exception:
        return VIDEO_RENDER_BBOX_THICKNESS_PX


def resolve_video_render_size(
    src_width: int,
    src_height: int,
    *,
    requested_width: int | None = None,
    requested_height: int | None = None,
) -> tuple[int, int]:
    """Return the output video size, enforcing source geometry by default."""
    src_width = int(src_width)
    src_height = int(src_height)
    if src_width <= 0 or src_height <= 0:
        raise ValueError(f"Invalid source video size: {src_width}x{src_height}")
    if ENFORCE_SOURCE_VIDEO_RESOLUTION:
        return src_width, src_height
    out_width = src_width if requested_width is None else int(requested_width)
    out_height = src_height if requested_height is None else int(requested_height)
    return max(1, out_width), max(1, out_height)
