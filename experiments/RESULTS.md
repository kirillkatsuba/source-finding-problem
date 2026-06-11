# Comparison of source-localization experiments

Each row is one method on one dataset. Metric is Euclidean error in grid cells
(lower is better). All experiments share the same per-dataset test split (seed=42).

## Dataset: nsk

| method                            |   mean err |   median |   std |   smooth mean |   n |
|:----------------------------------|-----------:|---------:|------:|--------------:|----:|
| Transolver + Heatmap (multi-task) |       6.29 |     4.12 |  5.15 |          5.57 |  50 |
| Trivial baseline (argmax t=17)    |      14.62 |    12.58 | 12.10 |         20.28 |  50 |

## Dataset: sakhalin

| method                                 |   mean err |   median |   std |   smooth mean |   n |
|:---------------------------------------|-----------:|---------:|------:|--------------:|----:|
| Trivial baseline (argmax t=17)         |       8.36 |     3.16 | 14.28 |          8.16 |  19 |
| Physical baseline (backward advection) |      30.33 |    26.31 | 21.61 |         22.40 |  19 |
