from __future__ import annotations

import torch
import torch.nn as nn


def _make_group_norm(channels: int, *, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNReLU3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        k: tuple[int, int, int],
        s: tuple[int, int, int],
        p: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.norm = _make_group_norm(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvGNReLU2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int, s: int, p: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.norm = _make_group_norm(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class VideoEncoder3D(nn.Module):
    """
    Lightweight 3D CNN encoder.

    - Keeps temporal resolution (stride 1 in time).
    - Output spatial stride is 4 (H/4, W/4).
    """

    def __init__(self, *, in_channels: int = 3, base_channels: int = 32, out_channels: int = 128):
        super().__init__()
        c1 = int(base_channels)
        c2 = int(base_channels) * 2
        c3 = int(out_channels)

        self.stem = ConvGNReLU3d(
            in_channels,
            c1,
            k=(3, 7, 7),
            s=(1, 2, 2),
            p=(1, 3, 3),
        )  # /2 spatial
        self.block1 = ConvGNReLU3d(c1, c1, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1))
        self.down = ConvGNReLU3d(c1, c2, k=(3, 3, 3), s=(1, 2, 2), p=(1, 1, 1))  # /4 spatial
        self.block2 = ConvGNReLU3d(c2, c2, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1))
        self.proj = ConvGNReLU3d(c2, c3, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1))

    def forward(self, clip_b3thw: torch.Tensor) -> torch.Tensor:
        x = self.stem(clip_b3thw)
        x = self.block1(x)
        x = self.down(x)
        x = self.block2(x)
        x = self.proj(x)
        return x


class FireflyVideoCenterNet(nn.Module):
    """
    Spatiotemporal single-class detector with a 3D CNN backbone.

    Input:  clip tensor [B, 3, T, H, W] in 0..1
    Output: dict with CenterNet-style heads on the *center* frame.
    """

    def __init__(self, *, base_channels: int = 32, feat_channels: int = 128):
        super().__init__()
        self.encoder3d = VideoEncoder3D(
            in_channels=3, base_channels=int(base_channels), out_channels=int(feat_channels)
        )

        self.head = nn.Sequential(
            ConvGNReLU2d(int(feat_channels), int(feat_channels), k=3, s=1, p=1),
            ConvGNReLU2d(int(feat_channels), int(feat_channels), k=3, s=1, p=1),
        )

        self.heatmap = nn.Conv2d(int(feat_channels), 1, kernel_size=1)
        self.wh = nn.Conv2d(int(feat_channels), 2, kernel_size=1)
        self.offset = nn.Conv2d(int(feat_channels), 2, kernel_size=1)
        self.tracking = nn.Conv2d(int(feat_channels), 2, kernel_size=1)

        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, clip_b3thw: torch.Tensor) -> dict[str, torch.Tensor]:
        feats_bcthw = self.encoder3d(clip_b3thw)  # [B,C,T,H',W']
        t = feats_bcthw.shape[2]
        center = feats_bcthw[:, :, t // 2]  # [B,C,H',W']
        x = self.head(center)

        return {
            "heatmap": self.heatmap(x),
            "wh": self.wh(x),
            "offset": self.offset(x),
            "tracking": self.tracking(x),
        }
