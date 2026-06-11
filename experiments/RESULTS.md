# Comparison of source-localization experiments

Each row is one method on one dataset. Metric is Euclidean error in grid cells
(lower is better). All experiments share the same per-dataset test split (seed=42).

## Dataset: nsk

| method                              |   mean err |   median |   std |   smooth mean |   n |
|:------------------------------------|-----------:|---------:|------:|--------------:|----:|
| UNet baseline                       |       3.16 |     2.24 |  2.67 |          2.86 |  50 |
| Transolver (field only)             |       3.32 |     1.41 |  4.30 |          8.06 |  50 |
| Transolver + Heatmap (multi-task)   |       4.24 |     3.16 |  3.20 |          3.52 |  50 |
| Transolver + Regressor (multi-task) |       5.04 |     2.24 |  6.86 |          8.86 |  50 |
| Trivial baseline (argmax t=17)      |      14.62 |    12.58 | 12.10 |         20.28 |  50 |

## Dataset: sakhalin

| method                                  |   mean err |   median |   std |   smooth mean |   n |
|:----------------------------------------|-----------:|---------:|------:|--------------:|----:|
| Transolver + Heatmap + Wind [with_wind] |       5.37 |     3.61 |  5.78 |          5.23 |  19 |
| Transolver + Heatmap + Wind [no_wind]   |       6.24 |     3.16 |  9.43 |          4.93 |  19 |
| Trivial baseline (argmax t=17)          |       8.36 |     3.16 | 14.28 |          8.16 |  19 |
| Physical baseline (backward advection)  |      30.33 |    26.31 | 21.61 |         22.40 |  19 |
