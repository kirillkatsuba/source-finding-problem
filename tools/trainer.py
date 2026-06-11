"""Общий цикл обучения/оценки для всех экспериментов (multi-task loss + чекпойнты)."""
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


@dataclass
class LossWeights:
    field: float = 1.0
    heatmap: float = 1.0
    coord: float = 0.0


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


@dataclass
class EpochStats:
    train_total: float = 0.0
    train_field: float = 0.0
    train_heat: float = 0.0
    train_coord: float = 0.0
    val_total: float = 0.0
    val_field: float = 0.0
    val_heat: float = 0.0
    val_coord: float = 0.0
    val_error_cells: float = 0.0


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

    return total, parts


def _predicted_xy(out: dict[str, torch.Tensor]) -> torch.Tensor:
    # приоритет: heatmap -> поле -> регрессор
    if "heatmap" in out:
        return batched_argmax_xy(out["heatmap"].detach())
    if "field" in out:
        b = out["field"].shape[0]
        flat = out["field"].detach().reshape(b, -1).argmax(dim=1)
        h, w = out["field"].shape[-2:]
        y = (flat // w).long()
        x = (flat % w).long()
        return torch.stack([x, y], dim=1)
    if "coords_regression" in out:
        return out["coords_regression"].round().long()
    raise RuntimeError("model output has no usable prediction key")


def _eval_loader(model: torch.nn.Module,
                 loader: DataLoader,
                 cfg: TrainConfig,
                 forward_fn: Callable) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {"total": [], "field": [], "heatmap": [], "coord": []}
    all_pred_xy = []
    all_true_xy = []
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, cfg.device)
            out = forward_fn(model, batch)
            total, parts = _losses_from_outputs(out, batch, cfg.loss_weights)
            totals["total"].append(float(total.detach()))
            for k in ("field", "heatmap", "coord"):
                if k in parts:
                    totals[k].append(parts[k])
            all_pred_xy.append(_predicted_xy(out).cpu().numpy())
            all_true_xy.append(batch["coords"].cpu().numpy())
    pred_xy = np.concatenate(all_pred_xy, axis=0)
    true_xy = np.concatenate(all_true_xy, axis=0)
    err = error_in_cells(pred_xy, true_xy)
    return {
        "total": float(np.mean(totals["total"])) if totals["total"] else 0.0,
        "field": float(np.mean(totals["field"])) if totals["field"] else 0.0,
        "heatmap": float(np.mean(totals["heatmap"])) if totals["heatmap"] else 0.0,
        "coord": float(np.mean(totals["coord"])) if totals["coord"] else 0.0,
        "error_cells": float(err.mean()),
        "median_error_cells": float(np.median(err)),
    }


def fit(model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
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
    best_err = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        ep_totals: dict[str, list[float]] = {"total": [], "field": [], "heatmap": [], "coord": []}
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False):
            batch = _move(batch, cfg.device)
            out = forward_fn(model, batch)
            total, parts = _losses_from_outputs(out, batch, cfg.loss_weights)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            ep_totals["total"].append(float(total.detach()))
            for k in ("field", "heatmap", "coord"):
                if k in parts:
                    ep_totals[k].append(parts[k])

        train_metrics = {f"train_{k}": float(np.mean(v)) if v else 0.0 for k, v in ep_totals.items()}
        val_metrics = {f"val_{k}": v for k, v in _eval_loader(model, val_loader, cfg, forward_fn).items()}
        epoch_log = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(epoch_log)

        if experiment is not None:
            experiment.log_metrics(epoch_log, step=epoch)

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            print(f"ep {epoch:03d} | train {train_metrics['train_total']:.4f} "
                  f"| val {val_metrics['val_total']:.4f} | val_err {val_metrics['val_error_cells']:.2f}")

        if val_metrics["val_error_cells"] < best_err:
            best_err = val_metrics["val_error_cells"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, out_dir / "best.pth")

    torch.save(model.state_dict(), out_dir / "last.pth")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    return {"best_val_error_cells": best_err, "history": history}


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
            field_for_smooth = out.get("heatmap", out.get("field")).cpu().numpy()
            pred_xy_smooth = batch_field_to_xy(field_for_smooth, sigma=cfg.smooth_sigma)
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
