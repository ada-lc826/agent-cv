"""Search-space modules in plain PyTorch (framework-independent)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    """Squeeze-and-Excitation (Hu et al., CVPR 2018)."""

    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(8, channels // ratio)
        self.fc = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(F.adaptive_avg_pool2d(x, 1))


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020) with adaptive kernel size."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)        # B,1,C
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)      # B,C,1,1
        return x * y


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""

    def __init__(self, channels: int, ratio: int = 16, kernel: int = 7):
        super().__init__()
        hidden = max(8, channels // ratio)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


class CoordAtt(nn.Module):
    """Coordinate Attention (Hou et al., CVPR 2021)."""

    def __init__(self, channels: int, ratio: int = 32):
        super().__init__()
        hidden = max(8, channels // ratio)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        xh = x.mean(3, keepdim=True)                         # n,c,h,1
        xw = x.mean(2, keepdim=True).permute(0, 1, 3, 2)     # n,c,w,1
        y = self.act(self.bn(self.conv1(torch.cat([xh, xw], 2))))
        yh, yw = torch.split(y, [h, w], 2)
        ah = torch.sigmoid(self.conv_h(yh))
        aw = torch.sigmoid(self.conv_w(yw.permute(0, 1, 3, 2)))
        return x * ah * aw


ATTENTION = {"se": SE, "eca": ECA, "cbam": CBAM, "coordatt": CoordAtt}


def build_attention(name: str, channels: int) -> nn.Module:
    return ATTENTION[name](channels)


def replace_conv3x3(conv: nn.Conv2d, kind: str) -> nn.Module:
    """Build a drop-in replacement for a 3x3 conv with the same in/out channels and stride."""
    cin, cout, stride = conv.in_channels, conv.out_channels, conv.stride
    if kind == "dilated":
        return nn.Conv2d(cin, cout, 3, stride=stride, padding=2, dilation=2, bias=False)
    if kind == "depthwise":
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False))
    raise ValueError(kind)


class WeightedFusion(nn.Module):
    """Learnable softmax-weighted sum of globally pooled multi-stage features (BiFPN-style weights)."""

    def __init__(self, in_channels: list, out_channels: int):
        super().__init__()
        self.proj = nn.ModuleList(nn.Identity() if c == out_channels else nn.Linear(c, out_channels)
                                  for c in in_channels)
        self.w = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, feats):
        pooled = [p(F.adaptive_avg_pool2d(f, 1).flatten(1)) for p, f in zip(self.proj, feats)]
        w = torch.softmax(self.w, 0)
        return sum(wi * x for wi, x in zip(w, pooled))
