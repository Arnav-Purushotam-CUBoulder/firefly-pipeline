from __future__ import annotations

import torch
import torch.nn.functional as F


def centernet_focal_loss(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    CornerNet/CenterNet focal loss on a heatmap.

    pred_logits: [B,1,H,W] (logits)
    gt:         [B,1,H,W] (0..1 gaussian with peak 1 at centers)
    """
    eps = 1e-6
    pred = torch.sigmoid(pred_logits)

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = -torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = -torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    loss = (pos_loss.sum() + neg_loss.sum()) / (num_pos + 1.0)
    return loss


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: [B,C,H,W]
    mask:        [B,1,H,W] where 1 indicates positive centers
    """
    mask = mask.expand_as(pred)
    loss = F.l1_loss(pred, target, reduction="none") * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def compute_losses(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    wh_weight: float = 0.1,
    off_weight: float = 1.0,
    track_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    hm_loss = centernet_focal_loss(pred["heatmap"], target["heatmap"])
    wh_loss = masked_l1_loss(F.relu(pred["wh"]), target["wh"], target["mask"])
    off_loss = masked_l1_loss(pred["offset"], target["offset"], target["mask"])
    track_loss = masked_l1_loss(pred["tracking"], target["tracking"], target["tracking_mask"])
    total = hm_loss + wh_weight * wh_loss + off_weight * off_loss + track_weight * track_loss
    return {
        "total": total,
        "heatmap": hm_loss,
        "wh": wh_loss,
        "offset": off_loss,
        "tracking": track_loss,
    }
