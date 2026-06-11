"""Метрика локализации: евклидова ошибка в клетках до argmax поля t=0."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


@dataclass
class LocalizationMetrics:
    mean_error: float
    median_error: float
    std_error: float
    mean_smooth_error: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_error_cells": self.mean_error,
            "median_error_cells": self.median_error,
            "std_error_cells": self.std_error,
            "mean_smooth_error_cells": self.mean_smooth_error,
            "n_samples": self.n,
        }


def error_in_cells(pred_xy: np.ndarray, true_xy: np.ndarray) -> np.ndarray:
    diff = pred_xy.astype(np.float32) - true_xy.astype(np.float32)
    return np.sqrt((diff ** 2).sum(axis=-1))


def field_to_xy(field: np.ndarray, sigma: float | None = None) -> tuple[int, int]:
    f = field
    if sigma is not None and sigma > 0:
        f = gaussian_filter(field, sigma=sigma)
    h, w = f.shape[-2], f.shape[-1]
    flat = int(np.argmax(f))
    y, x = divmod(flat, w)
    return int(x), int(y)


def batch_field_to_xy(fields: torch.Tensor | np.ndarray,
                      sigma: float | None = None) -> np.ndarray:
    if isinstance(fields, torch.Tensor):
        fields = fields.detach().cpu().numpy()
    out = np.zeros((fields.shape[0], 2), dtype=np.int64)
    for i in range(fields.shape[0]):
        out[i] = field_to_xy(fields[i], sigma=sigma)
    return out


def summarize(pred_xy: np.ndarray,
              true_xy: np.ndarray,
              pred_xy_smooth: np.ndarray | None = None) -> LocalizationMetrics:
    err = error_in_cells(pred_xy, true_xy)
    err_smooth = err if pred_xy_smooth is None else error_in_cells(pred_xy_smooth, true_xy)
    return LocalizationMetrics(
        mean_error=float(err.mean()),
        median_error=float(np.median(err)),
        std_error=float(err.std()),
        mean_smooth_error=float(err_smooth.mean()),
        n=int(len(err)),
    )
