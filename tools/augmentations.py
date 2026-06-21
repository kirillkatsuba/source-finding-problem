"""Label-aware аугментации: при отражении/повороте полей правим coords, heatmap и ветер."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class AugConfig:
    flip_h: bool = True       # отражение по W (x)
    flip_v: bool = True       # отражение по H (y)
    rot90: bool = False       # поворот 90/180/270 deg, только для квадратной сетки
    translate: bool = False   # случайный сдвиг xy (источник -> произвольная точка)
    max_shift: int = 48       # верхний предел сдвига; фактический ограничен bbox струи

    @property
    def any_enabled(self) -> bool:
        return self.flip_h or self.flip_v or self.rot90 or self.translate


def _flip_h(field: np.ndarray, x: int, w: int) -> tuple[np.ndarray, int]:
    return field[..., :, ::-1].copy(), w - 1 - x


def _flip_v(field: np.ndarray, y: int, h: int) -> tuple[np.ndarray, int]:
    return field[..., ::-1, :].copy(), h - 1 - y


def _rot90(field: np.ndarray, x: int, y: int, h: int, w: int, k: int) -> tuple[np.ndarray, int, int, int, int]:
    k %= 4
    if k == 0:
        return field, x, y, h, w
    rot = np.rot90(field, k=k, axes=(-2, -1)).copy()
    # пересчет (x, y) под np.rot90 (CCW); при k нечетном H и W меняются местами
    if k == 1:
        new_x, new_y, new_h, new_w = y, w - 1 - x, w, h
    elif k == 2:
        new_x, new_y, new_h, new_w = w - 1 - x, h - 1 - y, h, w
    else:  # k == 3
        new_x, new_y, new_h, new_w = h - 1 - y, x, w, h
    return rot, new_x, new_y, new_h, new_w


def _translate(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    # сдвиг по (dy: ось H, dx: ось W) с заполнением нулями, без заворота
    out = np.zeros_like(arr)
    h, w = arr.shape[-2], arr.shape[-1]
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    gy, gye = max(0, -dy), min(h, h - dy)
    gx, gxe = max(0, -dx), min(w, w - dx)
    out[..., ys:ye, xs:xe] = arr[..., gy:gye, gx:gxe]
    return out


def apply_augmentation(sample: dict,
                       cfg: AugConfig,
                       rng: random.Random) -> dict:
    field_input = sample["field_input"]
    field_target = sample["field_target"]
    heatmap = sample["heatmap"]
    x, y = int(sample["coords"][0]), int(sample["coords"][1])
    h, w = field_target.shape
    wind = sample.get("wind")

    if cfg.flip_h and rng.random() < 0.5:
        field_input, _ = _flip_h(field_input, x, w)
        field_target, _ = _flip_h(field_target, x, w)
        heatmap, _ = _flip_h(heatmap, x, w)
        x = w - 1 - x
        if wind is not None:
            wind, _ = _flip_h(wind, 0, w)
            wind = wind.copy()
            wind[0] = -wind[0]   # U меняет знак

    if cfg.flip_v and rng.random() < 0.5:
        field_input, _ = _flip_v(field_input, y, h)
        field_target, _ = _flip_v(field_target, y, h)
        heatmap, _ = _flip_v(heatmap, y, h)
        y = h - 1 - y
        if wind is not None:
            wind, _ = _flip_v(wind, 0, h)
            wind = wind.copy()
            wind[1] = -wind[1]   # V меняет знак

    if cfg.rot90 and h == w:
        k = rng.randint(0, 3)
        if k != 0:
            field_input, x_i, y_i, h_i, w_i = _rot90(field_input, x, y, h, w, k)
            field_target, _, _, _, _ = _rot90(field_target, x, y, h, w, k)
            heatmap, _, _, _, _ = _rot90(heatmap, x, y, h, w, k)
            if wind is not None:
                # поворот вектора (U, V): k=1 -> (-V,U), k=2 -> (-U,-V), k=3 -> (V,-U)
                rot_wind, _, _, _, _ = _rot90(wind, 0, 0, h, w, k)
                u_old, v_old = rot_wind[0].copy(), rot_wind[1].copy()
                if k == 1:
                    rot_wind[0], rot_wind[1] = -v_old, u_old
                elif k == 2:
                    rot_wind[0], rot_wind[1] = -u_old, -v_old
                else:
                    rot_wind[0], rot_wind[1] = v_old, -u_old
                wind = rot_wind
            x, y = x_i, y_i

    if cfg.translate:
        h, w = field_target.shape
        sig = field_input.max(axis=0)
        thr = float(sig.max()) * 1e-3
        ys, xs = np.where((sig > thr) | (field_target > thr))
        if len(xs):
            m = cfg.max_shift
            dx = rng.randint(max(-int(xs.min()), -m), min(w - 1 - int(xs.max()), m))
            dy = rng.randint(max(-int(ys.min()), -m), min(h - 1 - int(ys.max()), m))
            if dx or dy:
                field_input = _translate(field_input, dx, dy)
                field_target = _translate(field_target, dx, dy)
                heatmap = _translate(heatmap, dx, dy)
                x += dx
                y += dy
                if wind is not None:
                    wind = _translate(wind, dx, dy)

    out = {
        "field_input": field_input,
        "field_target": field_target,
        "heatmap": heatmap,
        "coords": np.array([x, y], dtype=np.int64),
    }
    if wind is not None:
        out["wind"] = wind
    return out
