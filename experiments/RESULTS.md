# Comparison of source-localization experiments

Each row is one method on one dataset. Metric is Euclidean error in grid cells
(lower is better). All experiments share the same per-dataset test split (seed=42).

## Dataset: nsk

| method | mean err | median | std | smooth mean | n |
| --- | --- | --- | --- | --- | --- |
| Transolver (field only) | 3.93 | 2.24 | 4.68 | 7.19 | 100 |
| UNet baseline | 4.13 | 1.71 | 6.08 | 4.13 | 100 |
| Transolver + Heatmap (multi-task) | 4.64 | 3.61 | 3.74 | 3.55 | 100 |
| Transolver + Regressor (multi-task) | 12.60 | 11.29 | 8.13 | 12.60 | 100 |
| Trivial baseline (argmax t=17) | 17.63 | 12.67 | 17.09 | 22.64 | 100 |

## Dataset: sakhalin

| method | mean err | median | std | smooth mean | n |
| --- | --- | --- | --- | --- | --- |
| Transolver + Heatmap + Wind [with_wind] | 3.38 | 1.41 | 4.46 | 2.87 | 57 |
| Transolver + Heatmap + Wind [no_wind] | 3.83 | 2.24 | 5.30 | 2.89 | 57 |
| PINN (advection-diffusion loss) | 5.32 | 2.24 | 10.92 | 4.81 | 57 |
| Trivial baseline (argmax t=17) | 8.93 | 2.83 | 16.38 | 10.10 | 57 |
| Physical baseline (backward advection) | 19.19 | 14.32 | 16.49 | 15.74 | 57 |
