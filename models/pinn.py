"""PINN: свёрточная сеть (поле C(t=0) + heatmap) + physics-loss адвекции-диффузии.

physics-informed часть - не в архитектуре, а в loss (advection_diffusion_residual):
он привязывает предсказанное C(t=0) к первому наблюдению C(t=1) через известный
ветер. Уравнение без источника:
    dC/dt + u*dC/dx + v*dC/dy - D*(d2C/dx2 + d2C/dy2) = 0
Считаем одношаговую невязку конечными разностями (единицы клетка/кадр, dt/dx
свёрнуты в diffusion и wind_scale - это мягкий регуляризатор, не точный симулятор).
Нужен ветер -> только sakhalin.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PINNSourceNet(nn.Module):
    def __init__(self, in_channels: int = 17, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.field_head = nn.Conv2d(hidden, 1, 1)
        self.heat_head = nn.Conv2d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, T_in, H, W) -> {'field': C(t=0), 'heatmap': карта вероятности}."""
        f = self.body(x)
        field = self.field_head(f).squeeze(1)              # (B, H, W)
        logits = self.heat_head(f)                          # (B, 1, H, W)
        b, _, h, w = logits.shape
        heat = torch.softmax(logits.view(b, h * w), dim=-1).view(b, h, w)
        return {"field": field, "heatmap": heat}


def advection_diffusion_residual(c0_pred: torch.Tensor,
                                 frames: torch.Tensor,
                                 wind: torch.Tensor,
                                 diffusion: float = 0.1,
                                 wind_scale: float = 1.0,
                                 dt: float = 1.0) -> torch.Tensor:
    """c0_pred: (B,H,W) - предсказанное C(t=0); frames: (B,T,H,W) - наблюдения t=1..;
    wind: (B,2,H,W) - U,V. Возвращает средний квадрат невязки (скаляр)."""
    c0 = c0_pred
    c1 = frames[:, 0]
    u = wind[:, 0] * wind_scale
    v = wind[:, 1] * wind_scale

    dcdt = (c1 - c0) / dt
    dcdx = (torch.roll(c0, -1, dims=-1) - torch.roll(c0, 1, dims=-1)) * 0.5
    dcdy = (torch.roll(c0, -1, dims=-2) - torch.roll(c0, 1, dims=-2)) * 0.5
    lap = (torch.roll(c0, -1, dims=-1) + torch.roll(c0, 1, dims=-1)
           + torch.roll(c0, -1, dims=-2) + torch.roll(c0, 1, dims=-2) - 4.0 * c0)

    residual = dcdt + u * dcdx + v * dcdy - diffusion * lap
    return (residual ** 2).mean()
