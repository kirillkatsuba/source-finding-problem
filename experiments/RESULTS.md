# Comparison of source-localization experiments

Each row is one method on one dataset. Metric is Euclidean error in grid cells
(lower is better). All experiments share the same per-dataset test split (seed=42).

## Dataset: nsk

| method | mean err | median | std | smooth mean | n |
| --- | --- | --- | --- | --- | --- |
| UNet baseline | 1.36 | 1.00 | 0.93 | 1.34 | 100 |
| Transolver + Heatmap (multi-task) | 4.99 | 2.24 | 10.30 | 3.77 | 100 |
| Transolver (field only) | 15.51 | 11.44 | 12.41 | 20.61 | 100 |
| Transolver + Regressor (multi-task) | 15.89 | 14.82 | 8.14 | 15.89 | 100 |
| Trivial baseline (argmax t=17) | 22.35 | 17.13 | 21.91 | 29.85 | 100 |

## Dataset: sakhalin

| method | mean err | median | std | smooth mean | n |
| --- | --- | --- | --- | --- | --- |
| Transolver + Heatmap + Wind [with_wind] | 2.71 | 2.24 | 1.52 | 2.03 | 57 |
| PINN (advection-diffusion loss) | 3.98 | 2.24 | 13.66 | 1.56 | 57 |
| Transolver + Heatmap + Wind [no_wind] | 4.76 | 1.41 | 14.00 | 3.43 | 57 |
| Trivial baseline (argmax t=17) | 6.60 | 2.24 | 14.91 | 7.89 | 57 |
| Physical baseline (backward advection) | 20.45 | 13.60 | 21.18 | 17.48 | 57 |
