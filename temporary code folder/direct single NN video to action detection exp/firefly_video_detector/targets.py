from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .annotations import Box


def gaussian_radius(det_size: tuple[float, float], min_overlap: float = 0.7) -> float:
    """
    CenterNet-style gaussian radius.

    det_size is (height, width) in OUTPUT (downsampled) coordinates.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1**2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0.0, b2**2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(0.0, b3**2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2d(shape: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(
    heatmap: np.ndarray, center: tuple[int, int], radius: int, k: float = 1.0
) -> None:
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = center
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) <= 0 or min(masked_heatmap.shape) <= 0:
        return

    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def generate_centernet_targets(
    boxes_xywh: Iterable[Box],
    *,
    input_size: tuple[int, int],
    downsample: int,
    min_overlap: float = 0.7,
) -> dict[str, np.ndarray]:
    """
    Build CenterNet-style targets for the *center frame*.

    - input_size is (H, W) in resized pixel coordinates.
    - boxes are in resized pixel coordinates (top-left x,y + w,h).
    """
    in_h, in_w = input_size
    out_h, out_w = in_h // downsample, in_w // downsample

    heatmap = np.zeros((out_h, out_w), dtype=np.float32)
    wh = np.zeros((2, out_h, out_w), dtype=np.float32)
    offset = np.zeros((2, out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.float32)

    for box in boxes_xywh:
        if box.w <= 0 or box.h <= 0:
            continue

        cx = box.x + box.w * 0.5
        cy = box.y + box.h * 0.5

        if not (0 <= cx < in_w and 0 <= cy < in_h):
            continue

        cx_out = cx / downsample
        cy_out = cy / downsample
        cx_int = int(cx_out)
        cy_int = int(cy_out)

        if not (0 <= cx_int < out_w and 0 <= cy_int < out_h):
            continue

        w_out = box.w / downsample
        h_out = box.h / downsample
        rad = gaussian_radius((h_out, w_out), min_overlap=min_overlap)
        radius = max(0, int(rad))

        draw_umich_gaussian(heatmap, (cx_int, cy_int), radius)
        wh[0, cy_int, cx_int] = box.w
        wh[1, cy_int, cx_int] = box.h
        offset[0, cy_int, cx_int] = cx_out - cx_int
        offset[1, cy_int, cx_int] = cy_out - cy_int
        mask[cy_int, cx_int] = 1.0

    return {
        "heatmap": heatmap[None, :, :],  # [1,H,W]
        "wh": wh,  # [2,H,W]
        "offset": offset,  # [2,H,W]
        "mask": mask[None, :, :],  # [1,H,W]
    }

