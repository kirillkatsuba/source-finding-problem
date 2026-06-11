"""exp_005: UNet вместо Transolver (контроль ценности backbone)."""
from __future__ import annotations

import argparse
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import add_common_args, setup_experiment
from models.unet import UNet
from tools.trainer import LossWeights, TrainConfig, evaluate_and_dump, fit


def forward_fn(model, batch):
    return model(batch["field_input"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--w-field", type=float, default=1.0)
    parser.add_argument("--w-heatmap", type=float, default=1.0)
    ctx = setup_experiment(parser, default_name="exp_005_unet_baseline")

    model = UNet(
        in_channels=ctx.train_set.T_in,
        base=ctx.args.base,
        use_field=True,
        use_heatmap=True,
        use_regression=False,
    ).to(ctx.device)

    cfg = TrainConfig(
        epochs=ctx.args.epochs,
        lr=ctx.args.lr,
        weight_decay=ctx.args.weight_decay,
        batch_size=ctx.args.batch_size,
        device=str(ctx.device),
        smooth_sigma=ctx.args.smooth_sigma,
        loss_weights=LossWeights(field=ctx.args.w_field, heatmap=ctx.args.w_heatmap, coord=0.0),
    )
    fit(model, ctx.train_loader, ctx.val_loader, cfg, forward_fn,
        ctx.out_dir, experiment=ctx.experiment)

    if not ctx.args.no_test:
        evaluate_and_dump(model, ctx.test_loader, cfg, forward_fn, ctx.out_dir, experiment=ctx.experiment)

    if ctx.experiment is not None:
        ctx.experiment.end()


if __name__ == "__main__":
    main()
