"""Bar-chart и CDF ошибок по экспериментам -> experiments/plots/."""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_ROOT = pathlib.Path(__file__).resolve().parent
PLOTS_DIR = EXP_ROOT / "plots"

DISPLAY_NAMES = {
    "exp_000_trivial": "Trivial (argmax t=17)",
    "exp_001_baseline": "Physical (backward adv.)",
    "exp_002_transolver": "Transolver (field only)",
    "exp_003_transolver_heatmap_multitask": "Transolver + Heatmap",
    "exp_004_transolver_regressor_multitask": "Transolver + Regressor",
    "exp_005_unet_baseline": "UNet",
    "exp_006_transolver_with_wind": "Transolver + Wind",
    "exp_007_pinn": "PINN (adv-diff loss)",
}


def _experiment_label(p: pathlib.Path) -> tuple[str, str | None]:
    for part in p.parents:
        if part.name.startswith("exp_"):
            name = part.name
            if "__" in name:
                base, suffix = name.split("__", 1)
                return base, suffix
            return name, None
    return p.parent.name, None


def _display(label_tuple: tuple[str, str | None]) -> str:
    base, suffix = label_tuple
    name = DISPLAY_NAMES.get(base, base)
    return f"{name} [{suffix}]" if suffix else name


def _dataset_of(metrics_path: pathlib.Path) -> str | None:
    cfg = metrics_path.parent / "config.json"
    if cfg.exists():
        with open(cfg) as f:
            return json.load(f).get("dataset")
    name = metrics_path.parent.name
    if name == "sakhalin_unified":
        return "sakhalin"
    if name in {"nsk", "sakhalin"}:
        return name
    return None


def collect() -> dict[str, list[tuple[str, dict, pd.DataFrame]]]:
    out: dict[str, list[tuple[str, dict, pd.DataFrame]]] = {}
    for mp in EXP_ROOT.glob("exp_*/**/metrics.json"):
        ds = _dataset_of(mp)
        if ds is None:
            continue
        with open(mp) as f:
            metrics = json.load(f)
        pred_path = mp.parent / "predictions.csv"
        preds = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
        out.setdefault(ds, []).append((_experiment_label(mp), metrics, preds))
    return out


def bar_plot(dataset: str, items: list[tuple[tuple[str, str | None], dict, pd.DataFrame]]) -> None:
    items = sorted(items, key=lambda it: it[1].get("mean_error", float("inf")))
    labels = [_display(name) for name, _, _ in items]
    means = [it[1]["mean_error"] for it in items]
    stds = [it[1].get("std_error", 0.0) for it in items]
    smooth = [it[1].get("mean_smooth_error", it[1]["mean_error"]) for it in items]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(items)), 5))
    x = np.arange(len(items))
    w = 0.4
    ax.bar(x - w / 2, means, w, yerr=stds, capsize=4, label="raw")
    ax.bar(x + w / 2, smooth, w, label="+ Gaussian smoothing")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean source localization error (cells)")
    ax.set_title(f"Dataset: {dataset}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"bar_{dataset}.png", dpi=150)
    plt.close(fig)


def cdf_plot(dataset: str, items: list[tuple[tuple[str, str | None], dict, pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, _, preds in items:
        if preds.empty or "error" not in preds.columns:
            continue
        errs = np.sort(preds["error"].to_numpy())
        if len(errs) == 0:
            continue
        ys = np.arange(1, len(errs) + 1) / len(errs)
        ax.plot(errs, ys, label=_display(name))
    ax.set_xlabel("Source localization error (cells)")
    ax.set_ylabel("Cumulative fraction of test samples")
    ax.set_title(f"Error CDF - {dataset}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"cdf_{dataset}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    by_ds = collect()
    if not by_ds:
        print("no metrics yet")
        return
    for ds, items in by_ds.items():
        bar_plot(ds, items)
        cdf_plot(ds, items)
        print(f"wrote bar_{ds}.png and cdf_{ds}.png")


if __name__ == "__main__":
    main()
