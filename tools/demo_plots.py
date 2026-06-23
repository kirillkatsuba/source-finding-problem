"""Демо-визуализации по методам: лёгкий/типичный/трудный пример на тесте.
Для каждого метода читает его predictions.csv, рисует 3 примера (10/50/90 перцентиль
ошибки): поле t=0 + истинный источник + предсказание. Кладёт demo_*.png в experiments/plots/.
Запуск локально (предсказания должны быть на месте): python tools/demo_plots.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dataset import _load_nsk_samples, _load_sakhalin_samples
from tools.splits import POLLUTION_DIR

EXP = ROOT / "experiments"
OUT = EXP / "plots"

# demo-имя -> (каталог с predictions.csv относительно experiments/, датасет)
METHODS = {
    "trivial":    ("exp_000_trivial/nsk",                     "nsk"),
    "field_only": ("exp_002_transolver",                      "nsk"),
    "heatmap":    ("exp_003_transolver_heatmap_multitask",    "nsk"),
    "regressor":  ("exp_004_transolver_regressor_multitask",  "nsk"),
    "unet":       ("exp_005_unet_baseline",                   "nsk"),
    "physical":   ("exp_001_baseline/sakhalin_unified",       "sakhalin"),
    "nowind":     ("exp_006_transolver_with_wind__no_wind",   "sakhalin"),
    "wind":       ("exp_006_transolver_with_wind__with_wind", "sakhalin"),
    "pinn":       ("exp_007_pinn",                            "sakhalin"),
    "pinn_nopde": ("exp_007_pinn__nopde",                     "sakhalin"),
}

_cache: dict = {}


def _fields(fname: str, kind: str) -> dict:
    key = (fname, kind)
    if key not in _cache:
        loader = _load_nsk_samples if kind == "nsk" else _load_sakhalin_samples
        _cache[key] = {s.source_idx: s.field_target for s in loader(POLLUTION_DIR / fname)}
    return _cache[key]


def make_demo(name: str, rel: str, kind: str) -> None:
    csv = EXP / rel / "predictions.csv"
    if not csv.exists():
        print(f"skip {name}: no {csv}")
        return
    df = pd.read_csv(csv).sort_values("error").reset_index(drop=True)
    n = len(df)
    picks = [("хороший", int(0.1 * (n - 1))),
             ("средний", int(0.5 * (n - 1))),
             ("плохой", int(0.9 * (n - 1)))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, (lab, i) in zip(axes, picks):
        r = df.iloc[i]
        f = _fields(r["file"], kind)[int(r["source_idx"])]
        ax.imshow(f, origin="lower", cmap="viridis", norm=PowerNorm(0.4))
        ax.scatter(r["true_x"], r["true_y"], c="white", s=130, marker="o",
                   edgecolors="black", lw=1.5, zorder=5)
        ax.scatter(r["pred_x"], r["pred_y"], c="red", s=140, marker="X",
                   edgecolors="white", lw=1.2, zorder=6)
        ax.set_title(f"{lab}: ошибка {r['error']:.1f} кл", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    h = [Line2D([0], [0], marker="o", color="white", mec="black", ls="", label="истинный источник"),
         Line2D([0], [0], marker="X", color="red", mec="white", ls="", label="предсказание")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"demo_{name}.png", dpi=130)
    plt.close(fig)
    print(f"wrote demo_{name}.png")


def main() -> None:
    for name, (rel, kind) in METHODS.items():
        make_demo(name, rel, kind)


if __name__ == "__main__":
    main()
