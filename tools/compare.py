"""Сравнение предсказаний всех методов на тесте (одна картинка на датасет).

Берёт predictions.csv каждого эксперимента, мёржит по (file, source_idx) и на
поле t=0 нескольких тестовых сэмплов рисует истинный источник + предсказание
каждого метода. Cross-dataset не мешаем: отдельно nsk и sakhalin.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dataset import _load_nsk_samples, _load_sakhalin_samples
from tools.splits import POLLUTION_DIR

EXP_ROOT = ROOT / "experiments"
PLOTS_DIR = EXP_ROOT / "plots"

DISPLAY = {
    "exp_000_trivial": "trivial",
    "exp_001_baseline": "physical",
    "exp_002_transolver": "field-only",
    "exp_003_transolver_heatmap_multitask": "heatmap",
    "exp_004_transolver_regressor_multitask": "regressor",
    "exp_005_unet_baseline": "unet",
    "exp_006_transolver_with_wind": "wind",
    "exp_007_pinn": "pinn",
}


def _dataset_of(pred_path: pathlib.Path) -> str | None:
    cfg = pred_path.parent / "config.json"
    if cfg.exists():
        with open(cfg) as f:
            return json.load(f).get("dataset")
    name = pred_path.parent.name
    if name == "sakhalin_unified":
        return "sakhalin"
    if name in ("nsk", "sakhalin"):
        return name
    return None


def _label(pred_path: pathlib.Path) -> str:
    for part in pred_path.parents:
        if part.name.startswith("exp_"):
            base = part.name.split("__")[0]
            suffix = part.name.split("__", 1)[1] if "__" in part.name else None
            name = DISPLAY.get(base, base)
            return f"{name}[{suffix}]" if suffix else name
    return pred_path.parent.name


def collect(dataset: str) -> dict[str, pd.DataFrame]:
    methods: dict[str, pd.DataFrame] = {}
    for pc in sorted(EXP_ROOT.glob("exp_*/**/predictions.csv")):
        if _dataset_of(pc) != dataset:
            continue
        methods[_label(pc)] = pd.read_csv(pc).set_index(["file", "source_idx"])
    return methods


def _fields_for_file(file_name: str, dataset: str) -> dict[int, np.ndarray]:
    path = POLLUTION_DIR / file_name
    loader = _load_nsk_samples if dataset == "nsk" else _load_sakhalin_samples
    return {s.source_idx: s.field_target for s in loader(path)}


def compare(dataset: str, n: int = 6) -> None:
    methods = collect(dataset)
    if not methods:
        print(f"{dataset}: no predictions")
        return
    labels = list(methods)
    samples = sorted(set().union(*[set(df.index) for df in methods.values()]))

    def mean_err(idx) -> float:
        es = [df.loc[idx, "error"] for df in methods.values() if idx in df.index]
        return float(np.mean(es)) if es else math.inf

    # показываем сэмплы по всему диапазону сложности (от лёгких к трудным)
    samples.sort(key=mean_err)
    n = min(n, len(samples))
    pick = [samples[i] for i in np.linspace(0, len(samples) - 1, n).round().astype(int)]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    cmap = {lab: colors[i % 10] for i, lab in enumerate(labels)}

    field_cache: dict[str, dict[int, np.ndarray]] = {}
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (file, src) in zip(axes, pick):
        if file not in field_cache:
            field_cache[file] = _fields_for_file(file, dataset)
        ax.imshow(field_cache[file][src], origin="lower", cmap="viridis")
        ref = next(df for df in methods.values() if (file, src) in df.index)
        tx, ty = int(ref.loc[(file, src), "true_x"]), int(ref.loc[(file, src), "true_y"])
        ax.scatter(tx, ty, c="white", s=90, marker="o", edgecolors="black", zorder=5)
        for lab, df in methods.items():
            if (file, src) in df.index:
                px = int(df.loc[(file, src), "pred_x"])
                py = int(df.loc[(file, src), "pred_y"])
                ax.scatter(px, py, color=cmap[lab], s=70, marker="x", linewidths=2, zorder=4)
        short = file.removeprefix("processed_flxout_").removesuffix(".nc")
        ax.set_title(f"{short} src={src} (avg {mean_err((file, src)):.1f})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(pick):]:
        ax.axis("off")

    handles = [Line2D([0], [0], marker="o", color="white", markeredgecolor="black",
                      linestyle="", label="true")]
    handles += [Line2D([0], [0], marker="x", color=cmap[l], linestyle="", label=l) for l in labels]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.suptitle(f"Predictions on test set - {dataset}")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"compare_{dataset}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["nsk", "sakhalin", "both"], default="both")
    p.add_argument("--n", type=int, default=6, help="сколько тестовых сэмплов показать")
    args = p.parse_args()
    kinds = ["nsk", "sakhalin"] if args.dataset == "both" else [args.dataset]
    for k in kinds:
        compare(k, n=args.n)


if __name__ == "__main__":
    main()
