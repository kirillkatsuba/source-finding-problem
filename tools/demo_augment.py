"""Демо аугментации-сдвига: один сэмпл -> оригинал + N случайных сдвигов настоящей
apply_augmentation (только translate). Видно, что поле t=0, целевая heatmap и источник
смещаются согласованно (источник попадает в произвольную точку плоскости).
Кладёт demo_augment.png в experiments/plots/. Запуск: python tools/demo_augment.py
"""
from __future__ import annotations

import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.augmentations import AugConfig, apply_augmentation
from tools.dataset import _load_nsk_samples
from tools.heatmap import gaussian_heatmap
from tools.splits import classify_files

OUT = ROOT / "experiments" / "plots"
SIGMA = 4.0
MAX_SHIFT = 48
N_SHIFTS = 4


def _noticks(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def _src(ax, x: int, y: int) -> None:
    ax.scatter(x, y, c="white", s=120, marker="o", edgecolors="black", lw=1.5, zorder=5)
    _noticks(ax)


def main() -> None:
    nsk = classify_files(ROOT / "data" / "pollution")["nsk"]
    if not nsk:
        print("no nsk files (нужны data/pollution)")
        return
    samples = _load_nsk_samples(nsk[0])
    s = max(samples, key=lambda s: float(s.field_target.sum()))  # самый заметный плюм
    h, w = s.field_target.shape
    x0, y0 = int(s.coords[0]), int(s.coords[1])

    base = {
        "field_input": s.field_input,
        "field_target": s.field_target,
        "heatmap": gaussian_heatmap(x0, y0, h, w, sigma=SIGMA),
        "coords": s.coords.copy(),
    }
    cfg = AugConfig(flip_h=False, flip_v=False, rot90=False, translate=True, max_shift=MAX_SHIFT)

    # оригинал + N сдвигов (нулевые сдвиги пропускаем, чтобы демо было наглядным)
    cols = [("оригинал", base)]
    seed = 0
    while len(cols) <= N_SHIFTS and seed < 500:
        aug = apply_augmentation(base, cfg, random.Random(seed))
        dx, dy = int(aug["coords"][0]) - x0, int(aug["coords"][1]) - y0
        seed += 1
        if dx or dy:
            cols.append((f"сдвиг ({dx:+d}, {dy:+d})", aug))

    ncol = len(cols)
    fig, axes = plt.subplots(2, ncol, figsize=(3 * ncol, 6.2))
    for j, (title, d) in enumerate(cols):
        cx, cy = int(d["coords"][0]), int(d["coords"][1])
        axes[0, j].imshow(d["field_target"], origin="lower", cmap="viridis", norm=PowerNorm(0.4))
        _src(axes[0, j], cx, cy)
        axes[0, j].set_title(title, fontsize=11)
        axes[1, j].imshow(d["heatmap"], origin="lower", cmap="magma", norm=PowerNorm(0.4))
        _noticks(axes[1, j])
    axes[0, 0].set_ylabel("поле при $t{=}0$", fontsize=12)
    axes[1, 0].set_ylabel("heatmap", fontsize=12)

    leg = [Line2D([0], [0], marker="o", color="white", mec="black", ls="", label="источник")]
    fig.legend(handles=leg, loc="lower center", fontsize=11)
    fig.suptitle("Случайный сдвиг источника", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "demo_augment.png", dpi=130)
    plt.close(fig)
    print(f"wrote demo_augment.png ({ncol - 1} сдвигов, источник {x0},{y0})")


if __name__ == "__main__":
    main()
