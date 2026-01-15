from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int, s: int, p: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FrameEncoder(nn.Module):
    """
    2D CNN encoder run per-frame.

    Output stride is 4 (H/4, W/4).
    """

    def __init__(self, *, in_channels: int = 3, base_channels: int = 32, out_channels: int = 128):
        super().__init__()
        self.stem = ConvBNReLU(in_channels, base_channels, k=7, s=2, p=3)  # /2
        self.block1 = ConvBNReLU(base_channels, base_channels, k=3, s=1, p=1)
        self.block2 = ConvBNReLU(base_channels, base_channels * 2, k=3, s=2, p=1)  # /4
        self.block3 = ConvBNReLU(base_channels * 2, out_channels, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell (2D conv gates).

    State is (h, c) where both are feature maps [B,C,H,W].
    """

    def __init__(self, channels: int, *, kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(channels * 2, channels * 4, kernel_size=kernel_size, padding=p)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h_prev = torch.zeros_like(x)
            c_prev = torch.zeros_like(x)
        else:
            h_prev, c_prev = state

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class TemporalFusionBiLSTM(nn.Module):
    """
    Bidirectional ConvLSTM over the clip.

    Input feats: [B,T,C,H,W] -> output fused feature map for the center frame: [B,C,H,W]
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fwd = ConvLSTMCell(channels)
        self.bwd = ConvLSTMCell(channels)
        self.out = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = feats_btchw.shape
        center_idx = t // 2

        state_f: tuple[torch.Tensor, torch.Tensor] | None = None
        f_states: list[torch.Tensor] = []  # hidden states
        for i in range(t):
            state_f = self.fwd(feats_btchw[:, i], state_f)
            f_states.append(state_f[0])

        state_b: tuple[torch.Tensor, torch.Tensor] | None = None
        b_states: list[torch.Tensor] = [
            torch.empty(0, device=feats_btchw.device) for _ in range(t)
        ]  # placeholder
        for i in reversed(range(t)):
            state_b = self.bwd(feats_btchw[:, i], state_b)
            b_states[i] = state_b[0]

        fused = torch.cat([f_states[center_idx], b_states[center_idx]], dim=1)
        return self.out(fused)


class FireflyVideoCenterNet(nn.Module):
    """
    Spatiotemporal single-class detector.

    Input:  clip tensor [B, 3, T, H, W] in 0..1
    Output: dict with CenterNet-style heads on the *center* frame.
    """

    def __init__(
        self,
        *,
        base_channels: int = 32,
        feat_channels: int = 128,
    ):
        super().__init__()
        self.encoder = FrameEncoder(
            in_channels=3, base_channels=base_channels, out_channels=feat_channels
        )
        self.temporal = TemporalFusionBiLSTM(feat_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(feat_channels, 1, kernel_size=1)
        self.wh = nn.Conv2d(feat_channels, 2, kernel_size=1)
        self.offset = nn.Conv2d(feat_channels, 2, kernel_size=1)
        self.tracking = nn.Conv2d(feat_channels, 2, kernel_size=1)

        # Initialize heatmap bias so initial predictions are low-confidence (reduces early FPs)
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, clip_b3thw: torch.Tensor) -> dict[str, torch.Tensor]:
        b, c, t, h, w = clip_b3thw.shape
        x = clip_b3thw.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,3,H,W]
        x = x.view(b * t, c, h, w)  # [BT,3,H,W]

        feat = self.encoder(x)  # [BT,C,H',W']
        _, c2, h2, w2 = feat.shape
        feat = feat.view(b, t, c2, h2, w2)  # [B,T,C,H',W']

        center_raw = feat[:, t // 2]  # [B,C,H',W']
        center_temp = self.temporal(feat)  # [B,C,H',W']
        fused = self.fuse(torch.cat([center_raw, center_temp], dim=1))
        fused = self.head(fused)

        return {
            "heatmap": self.heatmap(fused),
            "wh": self.wh(fused),
            "offset": self.offset(fused),
            "tracking": self.tracking(fused),
        }
