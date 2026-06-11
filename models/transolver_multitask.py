"""Transolver-backbone (поле t=0) + CNN-головы: heatmap и/или регрессор (x, y)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from TransolverPDE.model.Transolver_Structured_Mesh_2D import Model as TransolverBackbone


class CNNDecoder(nn.Module):
    def __init__(self, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, hidden, 3, padding=1), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.to_logits = nn.Conv2d(hidden, 1, 1)

    def forward(self, field_2d: torch.Tensor) -> torch.Tensor:
        return self.to_logits(self.body(field_2d))


def spatial_softmax(logits: torch.Tensor) -> torch.Tensor:
    # (B, 1, H, W) -> (B, H, W), сумма по сетке = 1
    b, _, h, w = logits.shape
    probs = torch.softmax(logits.view(b, h * w), dim=-1)
    return probs.view(b, h, w)


def soft_argmax_xy(heatmap: torch.Tensor) -> torch.Tensor:
    # матожидание (x, y) по heatmap: (B, H, W) -> (B, 2)
    b, h, w = heatmap.shape
    device = heatmap.device
    xs = torch.linspace(0.0, w - 1.0, w, device=device)
    ys = torch.linspace(0.0, h - 1.0, h, device=device)
    grid_x = xs.view(1, 1, w).expand(b, h, w)
    grid_y = ys.view(1, h, 1).expand(b, h, w)
    x = (heatmap * grid_x).sum(dim=(-1, -2))
    y = (heatmap * grid_y).sum(dim=(-1, -2))
    return torch.stack([x, y], dim=-1)


class TransolverMultiTask(nn.Module):
    def __init__(self,
                 h: int,
                 w: int,
                 t_in: int = 17,
                 extra_in_channels: int = 0,
                 n_layers: int = 3,
                 n_hidden: int = 64,
                 n_head: int = 4,
                 slice_num: int = 64,
                 ref: int = 8,
                 backbone_weights: str | None = None,
                 freeze_backbone: bool = False,
                 use_heatmap: bool = True,
                 use_regression: bool = False,
                 decoder_hidden: int = 64,
                 decoder_layers: int = 3):
        super().__init__()
        self.h = h
        self.w = w
        self.t_in = t_in
        self.extra_in_channels = extra_in_channels
        self.fun_dim = t_in + extra_in_channels
        self.use_heatmap = use_heatmap
        self.use_regression = use_regression

        self.backbone = TransolverBackbone(
            space_dim=2,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            Time_Input=False,
            mlp_ratio=1,
            fun_dim=self.fun_dim,
            out_dim=1,
            slice_num=slice_num,
            ref=ref,
            unified_pos=1,
            H=h,
            W=w,
        )
        # грузим веса только если число входных каналов совпадает
        if backbone_weights is not None and extra_in_channels == 0:
            state = torch.load(backbone_weights, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"backbone load: missing={len(missing)} unexpected={len(unexpected)}")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.decoder = CNNDecoder(hidden=decoder_hidden, n_layers=decoder_layers)

        if use_regression:
            self.regressor = nn.Sequential(
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(decoder_hidden * 8 * 8, 128),
                nn.GELU(),
                nn.Linear(128, 2),
            )

    def forward(self, pos: torch.Tensor, fx: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(pos, fx)  # (B, H*W, 1)
        b = z.shape[0]
        field_2d = z.view(b, self.h, self.w).unsqueeze(1)  # (B, 1, H, W)

        out: dict[str, torch.Tensor] = {"field": field_2d.squeeze(1)}

        if not (self.use_heatmap or self.use_regression):
            return out

        feats = self.decoder.body(field_2d)
        logits = self.decoder.to_logits(feats)

        if self.use_heatmap:
            heat = spatial_softmax(logits)
            out["heatmap"] = heat
            out["coords_softargmax"] = soft_argmax_xy(heat)

        if self.use_regression:
            out["coords_regression"] = self.regressor(feats)

        return out
