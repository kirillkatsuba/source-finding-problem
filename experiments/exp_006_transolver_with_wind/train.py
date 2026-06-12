"""exp_006: Transolver + heatmap + ветер (U10/V10) на sakhalin; абляция с ветром / без."""
from __future__ import annotations

import argparse
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import add_common_args, setup_experiment
from models.transolver_multitask import TransolverMultiTask
from tools.dataset import transolver_inputs_with_wind, transolver_inputs
from tools.trainer import LossWeights, TrainConfig, evaluate_and_dump, fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--backbone-weights", type=str, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--w-field", type=float, default=1.0)
    parser.add_argument("--w-heatmap", type=float, default=1.0)
    # ветер включается флагом --include-wind; без него no-wind (для честной абляции)
    parser.set_defaults(dataset="sakhalin")
    ctx = setup_experiment(parser, default_name="exp_006_transolver_with_wind")

    if ctx.args.dataset != "sakhalin":
        raise RuntimeError("exp_006 requires --dataset sakhalin (only files with wind)")

    extra = 2 if ctx.args.include_wind else 0
    forward = (lambda model, batch: model(*transolver_inputs_with_wind(batch))) if ctx.args.include_wind \
        else (lambda model, batch: model(*transolver_inputs(batch)))

    model = TransolverMultiTask(
        h=ctx.train_set.H,
        w=ctx.train_set.W,
        t_in=ctx.train_set.T_in,
        extra_in_channels=extra,
        backbone_weights=ctx.args.backbone_weights,
        freeze_backbone=ctx.args.freeze_backbone,
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
    fit(model, ctx.train_loader, cfg, forward,
        ctx.out_dir, experiment=ctx.experiment)

    if not ctx.args.no_test:
        evaluate_and_dump(model, ctx.test_loader, cfg, forward, ctx.out_dir, experiment=ctx.experiment)

    if ctx.experiment is not None:
        ctx.experiment.end()


if __name__ == "__main__":
    main()
