"""UNet-baseline без Transolver: 17 кадров -> field + heatmap."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 17, base: int = 32,
                 use_heatmap: bool = True, use_regression: bool = False,
                 use_field: bool = True):
        super().__init__()
        self.use_heatmap = use_heatmap
        self.use_regression = use_regression
        self.use_field = use_field

        self.enc1 = _conv_block(in_channels, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.bottleneck = _conv_block(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)

        self.pool = nn.MaxPool2d(2)

        if use_field:
            self.field_head = nn.Conv2d(base, 1, 1)
        if use_heatmap:
            self.heat_head = nn.Conv2d(base, 1, 1)
        if use_regression:
            self.coord_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(base * 64, 128),
                nn.GELU(),
                nn.Linear(128, 2),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))

        # после upsample подгоняем размер под skip-связь (нечетные H/W)
        d3 = self.up3(bn)
        d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False) if d3.shape[-2:] != e3.shape[-2:] else d3
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False) if d2.shape[-2:] != e2.shape[-2:] else d2
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False) if d1.shape[-2:] != e1.shape[-2:] else d1
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out: dict[str, torch.Tensor] = {}
        if self.use_field:
            out["field"] = self.field_head(d1).squeeze(1)
        if self.use_heatmap:
            heat_logits = self.heat_head(d1)
            b, _, h, w = heat_logits.shape
            heat = torch.softmax(heat_logits.view(b, h * w), dim=-1).view(b, h, w)
            out["heatmap"] = heat
            xs = torch.linspace(0.0, w - 1.0, w, device=x.device).view(1, 1, w).expand(b, h, w)
            ys = torch.linspace(0.0, h - 1.0, h, device=x.device).view(1, h, 1).expand(b, h, w)
            out["coords_softargmax"] = torch.stack([(heat * xs).sum(dim=(-1, -2)),
                                                    (heat * ys).sum(dim=(-1, -2))], dim=-1)
        if self.use_regression:
            out["coords_regression"] = self.coord_head(d1)
        return out
