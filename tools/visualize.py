"""Картинки best/median/worst предсказаний для эксперимента (best.pth + predictions.csv)."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dataset import SourceDataset, transolver_inputs, transolver_inputs_with_wind
from tools.device import pick_device
from tools.splits import classify_files, split_files


def _model_for(exp_dir: pathlib.Path, train_set: SourceDataset, cfg: dict, device) -> tuple[torch.nn.Module, Callable]:
    extra = 2 if cfg.get("include_wind") else 0
    exp_name = cfg["experiment"]
    if exp_name.startswith("exp_005"):
        from models.unet import UNet
        m = UNet(in_channels=train_set.T_in, base=cfg.get("base", 32),
                 use_field=True, use_heatmap=True, use_regression=False)
        forward = lambda model, batch: model(batch["field_input"])
    else:
        from models.transolver_multitask import TransolverMultiTask
        use_heat = exp_name not in {"exp_002_transolver"}
        use_reg = exp_name.startswith("exp_004")
        m = TransolverMultiTask(
            h=train_set.H, w=train_set.W, t_in=train_set.T_in,
            extra_in_channels=extra,
            use_heatmap=use_heat, use_regression=use_reg,
        )
        if cfg.get("include_wind"):
            forward = lambda model, batch: model(*transolver_inputs_with_wind(batch))
        else:
            forward = lambda model, batch: model(*transolver_inputs(batch))
    m = m.to(device)
    return m, forward


def _move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _draw_one(out_path: pathlib.Path,
              field_input: np.ndarray,
              field_target: np.ndarray,
              heatmap_pred: np.ndarray | None,
              field_pred: np.ndarray | None,
              true_xy: tuple[int, int],
              pred_xy: tuple[int, int],
              title: str) -> None:
    n_panels = 2 + int(heatmap_pred is not None) + int(field_pred is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    idx = 0

    axes[idx].imshow(field_input[-1], origin="lower", cmap="viridis")
    axes[idx].set_title("Наблюдение t=17 (последний кадр)")
    axes[idx].scatter(*true_xy, c="white", s=40, marker="o", edgecolors="black", label="истина")
    axes[idx].scatter(*pred_xy, c="red", s=40, marker="x", label="предсказание")
    axes[idx].legend(loc="upper right", fontsize=8)
    idx += 1

    if field_pred is not None:
        axes[idx].imshow(field_pred, origin="lower", cmap="plasma")
        axes[idx].set_title("Предсказанное поле t=0")
        axes[idx].scatter(*true_xy, c="white", s=40, marker="o", edgecolors="black")
        axes[idx].scatter(*pred_xy, c="red", s=40, marker="x")
        idx += 1

    if heatmap_pred is not None:
        axes[idx].imshow(heatmap_pred, origin="lower", cmap="hot")
        axes[idx].set_title("Предсказанная heatmap")
        axes[idx].scatter(*true_xy, c="white", s=40, marker="o", edgecolors="black")
        axes[idx].scatter(*pred_xy, c="red", s=40, marker="x")
        idx += 1

    axes[idx].imshow(field_target, origin="lower", cmap="viridis")
    axes[idx].set_title("Истина t=0")
    axes[idx].scatter(*true_xy, c="white", s=40, marker="o", edgecolors="black")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def visualize(exp_dir: pathlib.Path,
              n_each: int = 3,
              device: str | None = None) -> None:
    cfg_path = exp_dir / "config.json"
    weights_path = exp_dir / "best.pth"
    pred_path = exp_dir / "predictions.csv"
    if not (cfg_path.exists() and weights_path.exists() and pred_path.exists()):
        raise FileNotFoundError(f"missing files in {exp_dir} (need config.json, best.pth, predictions.csv)")

    with open(cfg_path) as f:
        cfg = json.load(f)

    dev = torch.device(device) if device else pick_device()

    groups = classify_files()
    split = split_files(groups[cfg["dataset"]], seed=cfg.get("seed", 42))
    train_set = SourceDataset(split.train, dataset_kind=cfg["dataset"],
                              include_wind=cfg.get("include_wind", False), quiet=True)
    test_set = SourceDataset(split.test, dataset_kind=cfg["dataset"],
                             include_wind=cfg.get("include_wind", False), quiet=True)

    model, forward = _model_for(exp_dir, train_set, cfg, dev)
    state = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state, strict=False)
    model.eval()

    import pandas as pd
    preds_df = pd.read_csv(pred_path).reset_index().rename(columns={"index": "row"})
    sorted_df = preds_df.sort_values("error").reset_index(drop=True)
    n = len(sorted_df)
    picks = {
        "best":   sorted_df.head(n_each)["row"].tolist(),
        "median": sorted_df.iloc[n // 2 - n_each // 2: n // 2 + (n_each - n_each // 2)]["row"].tolist(),
        "worst":  sorted_df.tail(n_each)["row"].tolist(),
    }

    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    samples = list(loader)

    viz_dir = exp_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for tag, rows in picks.items():
            for j, row in enumerate(rows):
                batch = _move(samples[row], dev)
                out = forward(model, batch)
                field_pred = out.get("field")
                if field_pred is not None:
                    field_pred = field_pred[0].cpu().numpy()
                heat = out.get("heatmap")
                if heat is not None:
                    heat = heat[0].cpu().numpy()
                pred_x = preds_df.iloc[row]["pred_x"]
                pred_y = preds_df.iloc[row]["pred_y"]
                true_x = preds_df.iloc[row]["true_x"]
                true_y = preds_df.iloc[row]["true_y"]
                err = preds_df.iloc[row]["error"]
                file_label = preds_df.iloc[row]["file"]
                src_label = preds_df.iloc[row]["source_idx"]
                _draw_one(
                    viz_dir / f"{tag}_{j}.png",
                    field_input=batch["field_input"][0].cpu().numpy(),
                    field_target=batch["field_target"][0].cpu().numpy(),
                    heatmap_pred=heat,
                    field_pred=field_pred,
                    true_xy=(int(true_x), int(true_y)),
                    pred_xy=(int(pred_x), int(pred_y)),
                    title=f"{tag} | err={err:.2f} cells | file={file_label} src={src_label}",
                )
    print(f"wrote {sum(len(v) for v in picks.values())} images to {viz_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp_dir")
    p.add_argument("--n-each", type=int, default=3)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    visualize(pathlib.Path(args.exp_dir), n_each=args.n_each, device=args.device)


if __name__ == "__main__":
    main()
