"""Демо аугментаций: один сэмпл -> оригинал + повороты 90/180/270 + случайный сдвиг.
Поле t=0, целевая heatmap и источник преобразуются согласованно теми же функциями
(_rot90 / apply_augmentation), что и при обучении.

Берём Новосибирск: у него аккуратный гладкий плюм (у Сахалина поле при t=0 -
разреженные частицы). Поворот рисуем напрямую через _rot90 (он работает и на
несквадратной сетке); для ровного ряда каждую панель дополняем до квадрата только
при отрисовке. Кладёт demo_augment.png в experiments/plots/. Запуск: python tools/demo_augment.py
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

from tools.augmentations import AugConfig, _rot90, apply_augmentation
from tools.dataset import _load_nsk_samples
from tools.heatmap import gaussian_heatmap
from tools.splits import classify_files

OUT = ROOT / "experiments" / "plots"
SIGMA = 4.0
MAX_SHIFT = 48
ROT_KS = (1, 2, 3)  # 90/180/270 deg против часовой (как np.rot90)


def _noticks(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def _src(ax, x: int, y: int) -> None:
    ax.scatter(x, y, c="white", s=120, marker="o", edgecolors="black", lw=1.5, zorder=5)
    _noticks(ax)


def _pad_square(field: np.ndarray, x: int, y: int) -> tuple[np.ndarray, int, int]:
    # центрируем поле в квадрате max(H,W) -> все панели одного размера несмотря на поворот
    h, w = field.shape
    m = max(h, w)
    oy, ox = (m - h) // 2, (m - w) // 2
    out = np.zeros((m, m), dtype=field.dtype)
    out[oy:oy + h, ox:ox + w] = field
    return out, x + ox, y + oy


def _rotated(base: dict, k: int) -> dict:
    # тот же _rot90, что и в обучающей аугментации: крутим поле/heatmap и пересчитываем источник
    x, y = int(base["coords"][0]), int(base["coords"][1])
    h, w = base["field_target"].shape
    field, nx, ny, _, _ = _rot90(base["field_target"], x, y, h, w, k)
    heat, _, _, _, _ = _rot90(base["heatmap"], x, y, h, w, k)
    return {"field_target": field, "heatmap": heat, "coords": np.array([nx, ny])}


def _shifted(base: dict, max_shift: int) -> dict | None:
    # самый крупный сдвиг по сидам через реальную apply_augmentation (только translate)
    cfg = AugConfig(flip_h=False, flip_v=False, rot90=False, translate=True, max_shift=max_shift)
    x0, y0 = int(base["coords"][0]), int(base["coords"][1])
    best, best_mag = None, 0
    for seed in range(200):
        aug = apply_augmentation(base, cfg, random.Random(seed))
        mag = abs(int(aug["coords"][0]) - x0) + abs(int(aug["coords"][1]) - y0)
        if mag > best_mag:
            best, best_mag = aug, mag
    return best


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

    cols = [("оригинал", base)]
    for k in ROT_KS:
        cols.append((f"поворот ${90 * k}^\\circ$", _rotated(base, k)))
    shifted = _shifted(base, MAX_SHIFT)
    if shifted is not None:
        dx, dy = int(shifted["coords"][0]) - x0, int(shifted["coords"][1]) - y0
        cols.append((f"сдвиг ({dx:+d}, {dy:+d})", shifted))

    ncol = len(cols)
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol, 6.4))
    for j, (title, d) in enumerate(cols):
        field, cx, cy = _pad_square(d["field_target"], int(d["coords"][0]), int(d["coords"][1]))
        heat, _, _ = _pad_square(d["heatmap"], 0, 0)
        axes[0, j].imshow(field, origin="lower", cmap="viridis", norm=PowerNorm(0.4))
        _src(axes[0, j], cx, cy)
        axes[0, j].set_title(title, fontsize=11)
        axes[1, j].imshow(heat, origin="lower", cmap="magma", norm=PowerNorm(0.4))
        _noticks(axes[1, j])
    axes[0, 0].set_ylabel("поле при $t{=}0$", fontsize=12)
    axes[1, 0].set_ylabel("heatmap", fontsize=12)

    leg = [Line2D([0], [0], marker="o", color="white", mec="black", ls="", label="источник")]
    fig.legend(handles=leg, loc="lower center", fontsize=11)
    fig.suptitle("Случайный поворот и сдвиг источника", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "demo_augment.png", dpi=130)
    plt.close(fig)
    print(f"wrote demo_augment.png (повороты {[90 * k for k in ROT_KS]} + сдвиг, источник {x0},{y0})")


if __name__ == "__main__":
    main()
