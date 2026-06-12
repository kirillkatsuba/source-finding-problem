"""Train/test split на уровне файлов: источники из одного файла не текут между сплитами.

Данных мало, поэтому val не выделяем - только train/test (по умолчанию 80/20).
"""
from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass

import xarray as xr

POLLUTION_DIR = pathlib.Path("data/pollution")
WIND_DIR = pathlib.Path("data/wind")

NSK_SHAPE = (18, 1, 10, 251, 201)
SAKHALIN_SHAPE = (18, 1, 19, 3, 300, 300)


@dataclass
class Split:
    train: list[pathlib.Path]
    test: list[pathlib.Path]

    def summary(self) -> str:
        return f"train={len(self.train)} test={len(self.test)}"


def _file_shape(path: pathlib.Path) -> tuple[int, ...]:
    with xr.open_dataset(path) as ds:
        return tuple(ds["CONC"].shape)


def classify_files(pollution_dir: pathlib.Path = POLLUTION_DIR,
                   wind_dir: pathlib.Path = WIND_DIR) -> dict[str, list[pathlib.Path]]:
    # skip скрытые файлы (макосные ._* из tar тоже матчат glob, но это не nc)
    pol_files = sorted(p for p in pollution_dir.glob("*.nc") if not p.name.startswith("."))
    groups: dict[str, list[pathlib.Path]] = {"nsk": [], "sakhalin": []}
    for p in pol_files:
        shape = _file_shape(p)
        if shape == NSK_SHAPE:
            groups["nsk"].append(p)
        elif shape == SAKHALIN_SHAPE:
            groups["sakhalin"].append(p)
    return groups


def split_files(files: list[pathlib.Path],
                train_frac: float = 0.8,
                seed: int = 42) -> Split:
    files = sorted(files)
    rng = random.Random(seed)
    indices = list(range(len(files)))
    rng.shuffle(indices)
    n_train = int(round(train_frac * len(files)))
    return Split(
        train=[files[i] for i in indices[:n_train]],
        test=[files[i] for i in indices[n_train:]],
    )


def wind_path_for(pol_path: pathlib.Path,
                  wind_dir: pathlib.Path = WIND_DIR) -> pathlib.Path | None:
    w = wind_dir / f"wind_for_{pol_path.name}"
    return w if w.exists() else None
