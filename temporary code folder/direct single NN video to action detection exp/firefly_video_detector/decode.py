from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float
    track_dx: float | None = None  # prev_center - curr_center in input pixels
    track_dy: float | None = None  # prev_center - curr_center in input pixels


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes: [N,4] and [M,4] in (x1,y1,x2,y2)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(
        min=0
    )
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(
        min=0
    )

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    idxs = scores.argsort(descending=True)
    keep: list[int] = []
    while idxs.numel() > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[idxs[1:]]).squeeze(0)
        idxs = idxs[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def decode_centernet(
    pred: dict[str, torch.Tensor],
    *,
    downsample: int,
    input_hw: tuple[int, int],
    topk: int = 50,
    score_thresh: float = 0.3,
    nms_iou: float = 0.5,
) -> list[Detection]:
    """
    Decode a single-image CenterNet-style prediction to xywh detections.

    pred tensors expected in shapes:
      - heatmap: [1,1,H',W']
      - wh:      [1,2,H',W']
      - offset:  [1,2,H',W']
    """
    in_h, in_w = input_hw
    heat = torch.sigmoid(pred["heatmap"])[0, 0]  # [H',W']
    wh = F.relu(pred["wh"])[0]  # [2,H',W']
    off = pred["offset"][0]  # [2,H',W']
    trk = pred.get("tracking")
    trk_map = trk[0] if trk is not None else None  # [2,H',W'] or None

    h_out, w_out = heat.shape
    k = min(int(topk), int(h_out * w_out))
    scores, inds = torch.topk(heat.view(-1), k=k)

    xs = (inds % w_out).float()
    ys = (inds // w_out).float()

    off_x = off[0].view(-1)[inds]
    off_y = off[1].view(-1)[inds]
    ws = wh[0].view(-1)[inds]
    hs = wh[1].view(-1)[inds]
    if trk_map is not None:
        trk_x = trk_map[0].view(-1)[inds] * float(downsample)
        trk_y = trk_map[1].view(-1)[inds] * float(downsample)
    else:
        trk_x = trk_y = None

    cx = (xs + off_x) * downsample
    cy = (ys + off_y) * downsample

    x1 = (cx - ws * 0.5).clamp(min=0, max=float(in_w - 1))
    y1 = (cy - hs * 0.5).clamp(min=0, max=float(in_h - 1))
    x2 = (cx + ws * 0.5).clamp(min=0, max=float(in_w - 1))
    y2 = (cy + hs * 0.5).clamp(min=0, max=float(in_h - 1))

    # Filter by score threshold
    keep = scores >= float(score_thresh)
    scores = scores[keep]
    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    if trk_x is not None and trk_y is not None:
        trk_x = trk_x[keep]
        trk_y = trk_y[keep]

    if scores.numel() == 0:
        return []

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    keep_idx = nms(boxes, scores, float(nms_iou))
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    if trk_x is not None and trk_y is not None:
        trk_x = trk_x[keep_idx]
        trk_y = trk_y[keep_idx]

    dets: list[Detection] = []
    for j, (b, s) in enumerate(zip(boxes, scores)):
        x1f, y1f, x2f, y2f = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        dx = float(trk_x[j]) if trk_x is not None else None
        dy = float(trk_y[j]) if trk_y is not None else None
        dets.append(
            Detection(
                x=x1f,
                y=y1f,
                w=max(1.0, x2f - x1f),
                h=max(1.0, y2f - y1f),
                score=float(s),
                track_dx=dx,
                track_dy=dy,
            )
        )
    return dets
