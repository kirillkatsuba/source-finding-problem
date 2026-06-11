"""Гауссовы heatmap-таргеты и argmax-хелперы. Ось 0 = H (y), ось 1 = W (x)."""
from __future__ import annotations

import numpy as np
import torch


def gaussian_heatmap(x: float, y: float, h: int, w: int,
                     sigma: float = 4.0, normalize: bool = True) -> np.ndarray:
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma ** 2))
    g = g.astype(np.float32)
    if normalize:
        s = g.sum()
        if s > 0:
            g /= s
    return g


def argmax_xy(heatmap: torch.Tensor | np.ndarray) -> tuple[int, int]:
    if isinstance(heatmap, torch.Tensor):
        flat = heatmap.reshape(-1).argmax().item()
        h, w = heatmap.shape[-2], heatmap.shape[-1]
    else:
        flat = int(np.argmax(heatmap))
        h, w = heatmap.shape
    y = flat // w
    x = flat % w
    return int(x), int(y)


def batched_argmax_xy(heatmap: torch.Tensor) -> torch.Tensor:
    b, h, w = heatmap.shape
    flat = heatmap.reshape(b, -1).argmax(dim=1)
    y = (flat // w).long()
    x = (flat % w).long()
    return torch.stack([x, y], dim=1)
