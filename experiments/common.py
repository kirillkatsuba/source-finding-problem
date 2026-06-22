"""Общая настройка эксперимента: seed, устройство, train/test даталоадеры, Comet."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.augmentations import AugConfig
from tools.dataset import NormStats, SourceDataset
from tools.device import pick_device
from tools.splits import Split, classify_files, n_sources_for, split_files, split_sources


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ExperimentContext:
    args: argparse.Namespace
    device: torch.device
    split: Split
    train_loader: DataLoader
    test_loader: DataLoader
    train_set: SourceDataset
    test_set: SourceDataset
    out_dir: pathlib.Path
    experiment: object | None


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=["nsk", "sakhalin"], default="nsk")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--heatmap-sigma", type=float, default=4.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--include-wind", action="store_true")
    parser.add_argument("--wind-per-frame", action="store_true")
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--rot90", action="store_true")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--max-shift", type=int, default=48)
    parser.add_argument("--source-split", action="store_true")
    parser.add_argument("--normalize", action="store_true")


def setup_experiment(parser: argparse.ArgumentParser, default_name: str,
                     unified_norm: bool = False) -> ExperimentContext:
    # unified_norm: target нормируем теми же статистиками, что и input
    # (нужно PINN, чтобы C(t=0) и C(t=1) были на одной шкале для physics-loss)
    args = parser.parse_args()
    if args.smoke:
        args.epochs = max(1, min(args.epochs, 2))
        args.batch_size = min(args.batch_size, 2)

    set_seed(args.seed)
    device = pick_device(args.device)

    groups = classify_files()
    files = groups[args.dataset]
    if not files:
        raise RuntimeError(f"no files for dataset={args.dataset}")

    # source-split: те же файлы, но непересекающиеся источники; иначе обычный file-split
    if args.source_split:
        train_src, test_src = split_sources(n_sources_for(args.dataset), seed=args.seed)
        train_files = test_files = files
    else:
        fsplit = split_files(files, seed=args.seed)
        train_files, test_files = fsplit.train, fsplit.test
        train_src = test_src = None
    split = Split(train=train_files, test=test_files)

    common_ds_kwargs = dict(
        dataset_kind=args.dataset,
        heatmap_sigma=args.heatmap_sigma,
        include_wind=args.include_wind,
        wind_per_frame=args.wind_per_frame,
    )
    aug_on = args.augment or args.rot90 or args.translate
    aug_cfg = AugConfig(flip_h=args.augment, flip_v=args.augment, rot90=args.rot90,
                        translate=args.translate, max_shift=args.max_shift)
    train_set = SourceDataset(train_files, **common_ds_kwargs, source_indices=train_src,
                              augment=aug_cfg if aug_on else None, seed=args.seed)
    test_set = SourceDataset(test_files, **common_ds_kwargs, source_indices=test_src)
    print(f"split[{'source' if args.source_split else 'file'}]: "
          f"train={len(train_set)} test={len(test_set)} samples")
    if args.normalize:
        stats = train_set.compute_norm_stats()
        if unified_norm:
            stats = NormStats(stats.input_mean, stats.input_std,
                              stats.input_mean, stats.input_std, stats.wind_scale)
        train_set.set_norm_stats(stats)
        test_set.set_norm_stats(stats)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin)

    out_name = f"{default_name}__{args.out_suffix}" if args.out_suffix else default_name
    out_dir = pathlib.Path(__file__).resolve().parent / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(args), "experiment": default_name, "out_name": out_name}, f, indent=2)

    experiment = None
    if args.comet:
        try:
            from comet_ml import Experiment as _Exp
            experiment = _Exp()
            experiment.set_name(args.name or default_name)
            experiment.log_parameters(vars(args))
        except Exception as e:
            print(f"comet disabled: {e}")

    return ExperimentContext(
        args=args, device=device, split=split,
        train_loader=train_loader, test_loader=test_loader,
        train_set=train_set, test_set=test_set,
        out_dir=out_dir, experiment=experiment,
    )
