"""Общий цикл обучения/оценки (multi-task loss). Без val: учим N эпох, оцениваем финальную модель на тесте."""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.heatmap import batched_argmax_xy
from tools.losses import coord_mse, heatmap_kl, relative_l2
from tools.metrics import batch_field_to_xy, error_in_cells, summarize

_LOSS_KEYS = ("field", "heatmap", "coord", "physics")


@dataclass
class LossWeights:
    field: float = 1.0
    heatmap: float = 1.0
    coord: float = 0.0
    physics: float = 0.0
    # гиперпараметры физического residual (используются при physics > 0)
    phys_diffusion: float = 0.1
    phys_wind_scale: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 8
    device: str = "cpu"
    smooth_sigma: float = 2.0
    loss_weights: LossWeights = field(default_factory=LossWeights)
    log_every: int = 1


def _move(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _losses_from_outputs(out: dict[str, torch.Tensor],
                        batch: dict,
                        w: LossWeights) -> tuple[torch.Tensor, dict[str, float]]:
    parts: dict[str, float] = {}
    total = torch.zeros((), device=out[next(iter(out))].device)

    if w.field > 0 and "field" in out:
        l_field = relative_l2(out["field"], batch["field_target"])
        total = total + w.field * l_field
        parts["field"] = float(l_field.detach())

    if w.heatmap > 0 and "heatmap" in out:
        l_heat = heatmap_kl(out["heatmap"], batch["heatmap"])
        total = total + w.heatmap * l_heat
        parts["heatmap"] = float(l_heat.detach())

    if w.coord > 0:
        if "coords_regression" in out:
            l_coord = coord_mse(out["coords_regression"], batch["coords"])
        elif "coords_softargmax" in out:
            l_coord = coord_mse(out["coords_softargmax"], batch["coords"])
        else:
            l_coord = None
        if l_coord is not None:
            total = total + w.coord * l_coord
            parts["coord"] = float(l_coord.detach())

    if w.physics > 0 and "field" in out and "wind" in batch:
        from models.pinn import advection_diffusion_residual
        l_phys = advection_diffusion_residual(
            out["field"], batch["field_input"], batch["wind"],
            diffusion=w.phys_diffusion, wind_scale=w.phys_wind_scale,
        )
        total = total + w.physics * l_phys
        parts["physics"] = float(l_phys.detach())

    return total, parts


def _predicted_xy(out: dict[str, torch.Tensor]) -> torch.Tensor:
    # приоритет головы по предсказанию: heatmap -> регрессор -> поле
    if "heatmap" in out:
        return batched_argmax_xy(out["heatmap"].detach())
    if "coords_regression" in out:
        return out["coords_regression"].detach().round().long()
    if "field" in out:
        b = out["field"].shape[0]
        flat = out["field"].detach().reshape(b, -1).argmax(dim=1)
        h, w = out["field"].shape[-2:]
        y = (flat // w).long()
        x = (flat % w).long()
        return torch.stack([x, y], dim=1)
    raise RuntimeError("model output has no usable prediction key")


def _smooth_source(out: dict[str, torch.Tensor]) -> torch.Tensor | None:
    # что сглаживать перед argmax; у регрессора карты нет -> None (smooth == raw)
    if "heatmap" in out:
        return out["heatmap"]
    if "coords_regression" in out:
        return None
    return out.get("field")


def fit(model: torch.nn.Module,
        train_loader: DataLoader,
        cfg: TrainConfig,
        forward_fn: Callable,
        out_dir: pathlib.Path,
        experiment=None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        ep_totals: dict[str, list[float]] = {k: [] for k in ("total", *_LOSS_KEYS)}
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False):
            batch = _move(batch, cfg.device)
            out = forward_fn(model, batch)
            total, parts = _losses_from_outputs(out, batch, cfg.loss_weights)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            ep_totals["total"].append(float(total.detach()))
            for k in _LOSS_KEYS:
                if k in parts:
                    ep_totals[k].append(parts[k])

        train_metrics = {f"train_{k}": float(np.mean(v)) if v else 0.0 for k, v in ep_totals.items()}
        epoch_log = {"epoch": epoch, **train_metrics}
        history.append(epoch_log)

        if experiment is not None:
            experiment.log_metrics(epoch_log, step=epoch)
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            print(f"ep {epoch:03d} | train {train_metrics['train_total']:.4f}")

    # без val лучшую модель не выбираем - сохраняем финальную (её и оцениваем на тесте)
    state = model.state_dict()
    torch.save(state, out_dir / "best.pth")
    torch.save(state, out_dir / "last.pth")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    return {"history": history}


def evaluate_and_dump(model: torch.nn.Module,
                     test_loader: DataLoader,
                     cfg: TrainConfig,
                     forward_fn: Callable,
                     out_dir: pathlib.Path,
                     experiment=None) -> dict:
    model.eval()
    rows = []
    all_pred_xy = []
    all_pred_xy_smooth = []
    all_true_xy = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            batch_dev = _move(batch, cfg.device)
            out = forward_fn(model, batch_dev)
            pred_xy = _predicted_xy(out).cpu().numpy()
            smooth_src = _smooth_source(out)
            if smooth_src is not None:
                pred_xy_smooth = batch_field_to_xy(smooth_src.cpu().numpy(), sigma=cfg.smooth_sigma)
            else:
                pred_xy_smooth = pred_xy
            true_xy = batch_dev["coords"].cpu().numpy()
            files = batch["file"]
            src_idx = batch["source_idx"]
            for j in range(len(true_xy)):
                rows.append({
                    "file": files[j],
                    "source_idx": int(src_idx[j]),
                    "pred_x": int(pred_xy[j, 0]),
                    "pred_y": int(pred_xy[j, 1]),
                    "smooth_pred_x": int(pred_xy_smooth[j, 0]),
                    "smooth_pred_y": int(pred_xy_smooth[j, 1]),
                    "true_x": int(true_xy[j, 0]),
                    "true_y": int(true_xy[j, 1]),
                    "error": float(np.linalg.norm(pred_xy[j] - true_xy[j])),
                    "smooth_error": float(np.linalg.norm(pred_xy_smooth[j] - true_xy[j])),
                })
            all_pred_xy.append(pred_xy)
            all_pred_xy_smooth.append(pred_xy_smooth)
            all_true_xy.append(true_xy)
    pred_xy = np.concatenate(all_pred_xy, axis=0)
    pred_xy_smooth = np.concatenate(all_pred_xy_smooth, axis=0)
    true_xy = np.concatenate(all_true_xy, axis=0)
    stats = summarize(pred_xy, true_xy, pred_xy_smooth)
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(asdict(stats), f, indent=2)
    if experiment is not None:
        experiment.log_metrics({
            "test_mean_error": stats.mean_error,
            "test_median_error": stats.median_error,
            "test_mean_smooth_error": stats.mean_smooth_error,
        })
    print(f"test: mean_err={stats.mean_error:.2f} median={stats.median_error:.2f} "
          f"smooth={stats.mean_smooth_error:.2f} (n={stats.n})")
    return asdict(stats)
