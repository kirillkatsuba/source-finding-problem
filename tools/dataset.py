"""Dataset: один сэмпл = один выброс (release) из .nc.

field_input (T_in,H,W) при t=1..t=17, field_target (H,W) при t=0,
coords (x,y) = истинный источник из attrs (точка выброса), heatmap (H,W), wind опц. (Sakhalin).
"""
from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from tqdm import tqdm

from tools.augmentations import AugConfig, apply_augmentation
from tools.heatmap import gaussian_heatmap
from tools.splits import WIND_DIR, wind_path_for


@dataclass
class NormStats:
    input_mean: float
    input_std: float
    target_mean: float
    target_std: float
    wind_scale: float | None = None   # общий std ветра (один на оба канала -> сохраняет направление)


@dataclass
class Sample:
    field_input: np.ndarray    # (T_in, H, W)
    field_target: np.ndarray   # (H, W)
    coords: np.ndarray         # (2,) int (x, y)
    wind: np.ndarray | None    # (2, H, W) | None
    file: str
    source_idx: int


def true_source_xy(conc: xr.DataArray, release_idx: int) -> tuple[int, int]:
    """Истинный источник из attrs (точка выброса) -> клетка сетки. Сетка Regular
    Latit/Longit, поэтому линейный перевод lon/lat. argmax поля брать нельзя: это пик
    струи, снесённый ветром от источника (расхождение до десятков клеток)."""
    a = conc.attrs
    olon, olat = float(a["OUTGRID_LONG"]), float(a["OUTGRID_LAT"])
    dx, dy = float(a["DX"]), float(a["DY"])
    h, w = int(conc.shape[-2]), int(conc.shape[-1])
    lat = 0.5 * (float(np.atleast_1d(a["MIN_LATS"])[release_idx]) + float(np.atleast_1d(a["MAX_LATS"])[release_idx]))
    lon = 0.5 * (float(np.atleast_1d(a["MIN_LONGS"])[release_idx]) + float(np.atleast_1d(a["MAX_LONGS"])[release_idx]))
    x = min(max(int(round((lon - olon) / dx)), 0), w - 1)
    y = min(max(int(round((lat - olat) / dy)), 0), h - 1)
    return x, y


def _load_nsk_samples(path: pathlib.Path) -> list[Sample]:
    samples: list[Sample] = []
    with xr.open_dataset(path) as ds:
        conc_da = ds["CONC"]
        conc = conc_da.values  # (18, 1, 10, 251, 201)
        n_releases = conc.shape[2]
        for s in range(n_releases):
            cube = conc[:, 0, s, :, :].astype(np.float32, copy=False)  # (18, H, W)
            field_target = cube[0]
            field_input = cube[1:]
            x, y = true_source_xy(conc_da, s)
            samples.append(Sample(
                field_input=field_input,
                field_target=field_target,
                coords=np.array([x, y], dtype=np.int64),
                wind=None,
                file=path.name,
                source_idx=s,
            ))
    return samples


def _load_sakhalin_samples(path: pathlib.Path,
                           bottom_top: int = 0,
                           wind_dir: pathlib.Path = WIND_DIR) -> list[Sample]:
    samples: list[Sample] = []
    wind_path = wind_path_for(path, wind_dir=wind_dir)
    with xr.open_dataset(path) as ds:
        conc_da = ds["CONC"]
        conc = conc_da.isel(bottom_top=bottom_top).values  # (18, 1, 19, H, W)
        n_releases = conc.shape[2]
        src_xy = [true_source_xy(conc_da, s) for s in range(n_releases)]
    wind_arr: np.ndarray | None = None
    if wind_path is not None:
        with xr.open_dataset(wind_path) as wds:
            u = wds["U10"].isel(Time=0).values.astype(np.float32)
            v = wds["V10"].isel(Time=0).values.astype(np.float32)
        wind_arr = np.stack([u, v], axis=0)  # (2, H, W)

    for s in range(n_releases):
        cube = conc[:, 0, s, :, :].astype(np.float32, copy=False)  # (18, H, W)
        field_target = cube[0]
        field_input = cube[1:]
        x, y = src_xy[s]
        samples.append(Sample(
            field_input=field_input,
            field_target=field_target,
            coords=np.array([x, y], dtype=np.int64),
            wind=wind_arr,
            file=path.name,
            source_idx=s,
        ))
    return samples


class SourceDataset(Dataset):
    """In-memory датасет одного подмножества (nsk или sakhalin)."""

    def __init__(self,
                 files: list[pathlib.Path],
                 dataset_kind: str,
                 heatmap_sigma: float = 4.0,
                 include_wind: bool = False,
                 quiet: bool = False,
                 source_indices: list[int] | None = None,
                 augment: AugConfig | None = None,
                 norm_stats: NormStats | None = None,
                 seed: int = 0):
        if dataset_kind not in {"nsk", "sakhalin"}:
            raise ValueError(f"unknown dataset_kind={dataset_kind!r}")
        self.dataset_kind = dataset_kind
        self.heatmap_sigma = heatmap_sigma
        self.include_wind = include_wind and dataset_kind == "sakhalin"
        self.augment = augment
        self.norm_stats = norm_stats
        self._rng = random.Random(seed)

        loader = _load_nsk_samples if dataset_kind == "nsk" else _load_sakhalin_samples
        self.samples: list[Sample] = []
        iterator = files if quiet else tqdm(files, desc=f"load {dataset_kind}")
        for p in iterator:
            self.samples.extend(loader(p))

        if source_indices is not None:
            keep = set(source_indices)
            self.samples = [s for s in self.samples if s.source_idx in keep]

        if not self.samples:
            raise RuntimeError(f"no samples loaded for dataset_kind={dataset_kind}")

        first = self.samples[0]
        self.H, self.W = first.field_target.shape
        self.T_in = first.field_input.shape[0]

        x = np.linspace(0.0, 1.0, self.W, dtype=np.float32)
        y = np.linspace(0.0, 1.0, self.H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        self.pos = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (H*W, 2) в [0, 1]

    def compute_norm_stats(self) -> NormStats:
        all_in = np.concatenate([s.field_input.reshape(-1) for s in self.samples])
        all_tg = np.concatenate([s.field_target.reshape(-1) for s in self.samples])
        wind_scale = None
        winds = [s.wind for s in self.samples if s.wind is not None]
        if winds:
            all_w = np.concatenate([w.reshape(-1) for w in winds])
            wind_scale = float(all_w.std() + 1e-8)
        return NormStats(
            input_mean=float(all_in.mean()),
            input_std=float(all_in.std() + 1e-8),
            target_mean=float(all_tg.mean()),
            target_std=float(all_tg.std() + 1e-8),
            wind_scale=wind_scale,
        )

    def set_norm_stats(self, stats: NormStats) -> None:
        self.norm_stats = stats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        s = self.samples[idx]
        x_src, y_src = int(s.coords[0]), int(s.coords[1])
        hm = gaussian_heatmap(x_src, y_src, self.H, self.W, sigma=self.heatmap_sigma)

        field_input = s.field_input
        field_target = s.field_target
        wind = s.wind
        coords = s.coords

        if self.augment is not None and self.augment.any_enabled:
            aug_in = {
                "field_input": field_input,
                "field_target": field_target,
                "heatmap": hm,
                "coords": coords,
            }
            if wind is not None:
                aug_in["wind"] = wind
            aug = apply_augmentation(aug_in, self.augment, self._rng)
            field_input = aug["field_input"]
            field_target = aug["field_target"]
            hm = aug["heatmap"]
            coords = aug["coords"]
            wind = aug.get("wind")

        if self.norm_stats is not None:
            field_input = (field_input - self.norm_stats.input_mean) / self.norm_stats.input_std
            field_target = (field_target - self.norm_stats.target_mean) / self.norm_stats.target_std
            # ветер делим на общий std (один на U/V) - масштаб ~ как у концентрации,
            # направление сохраняется, знак после аугментации не трогаем
            if wind is not None and self.norm_stats.wind_scale:
                wind = wind / self.norm_stats.wind_scale

        item: dict[str, torch.Tensor | str | int] = {
            "field_input": torch.from_numpy(field_input.astype(np.float32, copy=False)),
            "field_target": torch.from_numpy(field_target.astype(np.float32, copy=False)),
            "coords": torch.from_numpy(coords),
            "heatmap": torch.from_numpy(hm),
            "pos": torch.from_numpy(self.pos),
            "file": s.file,
            "source_idx": s.source_idx,
        }
        if self.include_wind and wind is not None:
            item["wind"] = torch.from_numpy(wind.astype(np.float32, copy=False))  # (2, H, W)
        return item


def transolver_inputs(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    # (B, T_in, H, W) -> pos (B, H*W, 2), fx (B, H*W, T_in)
    field = batch["field_input"]
    b, t_in, h, w = field.shape
    fx = field.permute(0, 2, 3, 1).reshape(b, h * w, t_in).contiguous()
    pos = batch["pos"]
    return pos, fx


def transolver_inputs_with_wind(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    # как transolver_inputs, но fx -> (B, H*W, T_in + 2) с каналами ветра
    pos, fx = transolver_inputs(batch)
    wind = batch["wind"]
    b, _, h, w = wind.shape
    wind_flat = wind.permute(0, 2, 3, 1).reshape(b, h * w, 2)
    fx = torch.cat([fx, wind_flat], dim=-1).contiguous()
    return pos, fx
