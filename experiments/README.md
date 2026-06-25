# Experiments

This folder contains all experiments comparing different approaches to
the air-pollution source localization problem.

## The big picture

The simulations in `data/pollution/` form **two physically separate
datasets** (different geography, different grid sizes, different number
of sources per file). We never mix them.

| dataset    | files | grid    | sources / file | wind paired |
|------------|-------|---------|----------------|-------------|
| `nsk`      |   49  | 251×201 | 10             | no          |
| `sakhalin` |   15  | 300×300 | 19 × 3 heights | yes (U10,V10) |

Both datasets are split into train/val/test on the **file level** with
`seed=42` (see `tools/splits.py`). Every experiment uses the same
split, so the numbers in the comparison table are directly comparable.

Metric: Euclidean distance in grid cells between the predicted source
location and the ground-truth argmax of the t=0 concentration field.

## Experiments

| #     | folder                                     | description                                                                                              | dataset |
|-------|--------------------------------------------|----------------------------------------------------------------------------------------------------------|---------|
| 000   | `exp_000_trivial/`                         | Sanity baseline: argmax of the latest observed frame (t=17).                                             | both    |
| 001   | `exp_001_baseline/`                        | Physical baseline — backward-in-time advection of the observed concentration.                            | sakhalin|
| 002   | `exp_002_transolver/`                      | Transolver only, predicts t=0 field from t=1..t=17, source = argmax(field).                              | nsk     |
| 003   | `exp_003_transolver_heatmap_multitask/`    | **Main contribution.** Transolver backbone + CNN decoder, multi-task loss field + heatmap.               | nsk     |
| 004   | `exp_004_transolver_regressor_multitask/`  | Same backbone, (x, y) regressor head instead of heatmap. Tests the heatmap-vs-regression hypothesis.     | nsk     |
| 005   | `exp_005_unet_baseline/`                   | No Transolver — pure UNet on the 17-frame input. Control for the value of the Transolver backbone.      | nsk     |
| 006   | `exp_006_transolver_with_wind/`            | Transolver + heatmap with two extra wind channels (U10, V10).                                            | sakhalin|

## How to run

All ML experiments share the same CLI in `experiments/common.py`. Key flags:

```
--dataset {nsk,sakhalin}        which physical dataset to use
--epochs                         number of training epochs (default 100)
--batch-size                     default 4
--lr                             default 1e-3
--device {cuda,mps,cpu}          auto-detected if omitted
--include-wind                   only for sakhalin
--backbone-weights PATH          warm-start Transolver from a checkpoint (exp_002 weights file)
--freeze-backbone                freeze the Transolver during training
--augment                        flip H/V augmentations (label-aware: also flips coords + heatmap)
--rot90                          90/180/270 rotation (square grids only — sakhalin)
--normalize                      z-score normalize inputs+target using TRAIN stats
--out-suffix STR                 append "__STR" to output folder (for ablations on the same exp)
--comet                          enable Comet ML logging
--smoke                          tiny smoke run (1-2 epochs, batch 2)
```

### Smoke test (a Mac, before paying for a VM)

```bash
poetry run python experiments/exp_003_transolver_heatmap_multitask/train.py \
    --smoke --epochs 1 --batch-size 2 --device cpu --augment --normalize
```

Confirms the pipeline runs end-to-end. Each smoke run takes ~2 minutes on CPU.

### One-shot full training on a CUDA VM

```bash
EPOCHS=100 BS=8 LR=1e-3 bash scripts/run_all.sh --comet
```

Runs every experiment in sequence (trivial, physical, Transolver, multi-task,
regressor, UNet, wind/no-wind ablation) with sensible defaults, then rebuilds
RESULTS.md and the comparison plots. ~5–8 hours on a single A100, much less if
you bump batch size.

### Real training on a CUDA VM (vast.ai)

```bash
# main multi-task experiment
poetry run python experiments/exp_003_transolver_heatmap_multitask/train.py \
    --dataset nsk --epochs 100 --batch-size 8 --lr 1e-3 --comet

# regressor variant (for the heatmap-vs-regression ablation)
poetry run python experiments/exp_004_transolver_regressor_multitask/train.py \
    --dataset nsk --epochs 100 --batch-size 8 --lr 1e-3 --comet

# UNet control
poetry run python experiments/exp_005_unet_baseline/train.py \
    --dataset nsk --epochs 100 --batch-size 8 --lr 1e-3 --base 32 --comet

# Transolver alone (reference for the multi-task delta)
poetry run python experiments/exp_002_transolver/train_transolver.py \
    --dataset nsk --epochs 100 --batch-size 8 --lr 1e-3 --comet

# Sakhalin: wind ablation
poetry run python experiments/exp_006_transolver_with_wind/train.py \
    --epochs 100 --batch-size 4 --lr 1e-3 --include-wind --comet
poetry run python experiments/exp_006_transolver_with_wind/train.py \
    --epochs 100 --batch-size 4 --lr 1e-3 --comet --name transolver_sakhalin_no_wind
```

### Lower-bound baselines (cheap — run locally)

```bash
poetry run python experiments/exp_000_trivial/run.py --dataset both
poetry run python experiments/exp_001_baseline/run_unified.py
```

### Build the comparison table

After any run finishes (it writes `metrics.json` into its folder), regenerate
the comparison table and plots:

```bash
poetry run python experiments/aggregate.py
poetry run python experiments/plots.py
```

Outputs:
- `experiments/RESULTS.md` — table grouped by dataset, ready to drop into the thesis.
- `experiments/RESULTS.csv` — same data for further plotting.
- `experiments/plots/bar_<dataset>.png` — bar chart per dataset.
- `experiments/plots/cdf_<dataset>.png` — error CDFs.

For qualitative best/median/worst examples (drop into the thesis):

```bash
poetry run python tools/visualize.py experiments/exp_003_transolver_heatmap_multitask --n-each 3
```

Writes annotated PNGs to `<exp_dir>/viz/`.

## Output structure (per experiment)

```
exp_XXX_*/
├── config.json         # CLI args, dataset, seed
├── history.json        # per-epoch losses + val metrics
├── metrics.json        # final test metrics (mean_error_cells, …)
├── predictions.csv     # per-sample (file, source_idx, pred, true, error)
├── best.pth            # best checkpoint by val error
└── last.pth            # last checkpoint
```
