# Comparison of source-localization experiments

Each row is one method on one dataset. Metric is Euclidean error in grid cells
(lower is better). All experiments share the same per-dataset test split (seed=42).

## Dataset: nsk

| method | mean err | median | std | smooth mean | n |
| --- | --- | --- | --- | --- | --- |
| UNet baseline | 1.62 | 1.41 | 1.17 | 1.49 | 100 |
| Transolver + Heatmap (multi-task) | 3.55 | 2.83 | 4.20 | 2.08 | 100 |
| Transolver (field only) | 16.25 | 13.15 | 12.24 | 21.04 | 100 |
| Transolver + Regressor (multi-task) | 20.32 | 17.59 | 12.52 | 20.32 | 100 |
| Trivial baseline (argmax t=17) | 22.35 | 17.13 | 21.91 | 29.85 | 100 |

## Dataset: sakhalin

| method | mean err | median | std | smooth mean | n |
| --- | --- | --- | --- | --- | --- |
| PINN (advection-diffusion loss) | 2.56 | 2.24 | 1.49 | 1.64 | 57 |
| Transolver + Heatmap + Wind [no_wind] | 3.51 | 2.00 | 7.39 | 2.45 | 57 |
| Transolver + Heatmap + Wind [with_wind] | 3.78 | 3.16 | 5.54 | 1.48 | 57 |
| Trivial baseline (argmax t=17) | 6.60 | 2.24 | 14.91 | 7.89 | 57 |
| Physical baseline (backward advection) | 20.45 | 13.60 | 21.18 | 17.48 | 57 |
