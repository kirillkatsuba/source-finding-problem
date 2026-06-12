"""exp_007: physics-informed сеть с advection-diffusion residual в loss (sakhalin).

Loss = w_field * relative_L2(field) + w_heat * KL(heatmap) + w_phys * R_pde,
где R_pde - невязка адвекции-диффузии (нужен ветер -> только sakhalin).
Источник = argmax heatmap. unified_norm: C(t=0) и C(t=1) на одной шкале для R_pde.
"""
from __future__ import annotations

import argparse
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import add_common_args, setup_experiment
from models.pinn import PINNSourceNet
from tools.trainer import LossWeights, TrainConfig, evaluate_and_dump, fit


def forward_fn(model, batch):
    return model(batch["field_input"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--w-field", type=float, default=1.0)
    parser.add_argument("--w-heatmap", type=float, default=1.0)
    parser.add_argument("--w-physics", type=float, default=0.1)
    parser.add_argument("--phys-diffusion", type=float, default=0.1)
    parser.add_argument("--phys-wind-scale", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=64)
    # physics-loss требует ветер в батче -> sakhalin + include_wind
    parser.set_defaults(dataset="sakhalin", include_wind=True)
    ctx = setup_experiment(parser, default_name="exp_007_pinn", unified_norm=True)

    if ctx.args.dataset != "sakhalin":
        raise RuntimeError("exp_007 requires --dataset sakhalin (physics loss needs wind)")

    model = PINNSourceNet(in_channels=ctx.train_set.T_in, hidden=ctx.args.hidden).to(ctx.device)

    cfg = TrainConfig(
        epochs=ctx.args.epochs,
        lr=ctx.args.lr,
        weight_decay=ctx.args.weight_decay,
        batch_size=ctx.args.batch_size,
        device=str(ctx.device),
        smooth_sigma=ctx.args.smooth_sigma,
        loss_weights=LossWeights(
            field=ctx.args.w_field, heatmap=ctx.args.w_heatmap, coord=0.0,
            physics=ctx.args.w_physics,
            phys_diffusion=ctx.args.phys_diffusion,
            phys_wind_scale=ctx.args.phys_wind_scale,
        ),
    )
    fit(model, ctx.train_loader, cfg, forward_fn,
        ctx.out_dir, experiment=ctx.experiment)

    if not ctx.args.no_test:
        evaluate_and_dump(model, ctx.test_loader, cfg, forward_fn, ctx.out_dir, experiment=ctx.experiment)

    if ctx.experiment is not None:
        ctx.experiment.end()


if __name__ == "__main__":
    main()
