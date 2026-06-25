"""Лоссы: relative L2 для поля, KL для heatmap, MSE для координат."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b = pred.shape[0]
    diff = (pred.reshape(b, -1) - target.reshape(b, -1)).norm(p=2, dim=1)
    norm = target.reshape(b, -1).norm(p=2, dim=1).clamp(min=eps)
    return (diff / norm).mean()


def heatmap_kl(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # обе карты - распределения по сетке (сумма 1)
    b = pred.shape[0]
    p = pred.reshape(b, -1).clamp(min=eps)
    q = target.reshape(b, -1).clamp(min=eps)
    return (q * (q.log() - p.log())).sum(dim=1).mean()


def coord_mse(pred_xy: torch.Tensor, true_xy: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_xy.float(), true_xy.float())
