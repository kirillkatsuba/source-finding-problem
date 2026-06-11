"""Собрать metrics.json со всех экспериментов в RESULTS.md / RESULTS.csv."""
from __future__ import annotations

import json
import pathlib

import pandas as pd


EXP_ROOT = pathlib.Path(__file__).resolve().parent

DISPLAY_NAMES = {
    "exp_000_trivial": "Trivial baseline (argmax t=17)",
    "exp_001_baseline": "Physical baseline (backward advection)",
    "exp_002_transolver": "Transolver (field only)",
    "exp_003_transolver_heatmap_multitask": "Transolver + Heatmap (multi-task)",
    "exp_004_transolver_regressor_multitask": "Transolver + Regressor (multi-task)",
    "exp_005_unet_baseline": "UNet baseline",
    "exp_006_transolver_with_wind": "Transolver + Heatmap + Wind",
}


def _infer_dataset(metrics_path: pathlib.Path) -> str | None:
    cfg = metrics_path.parent / "config.json"
    if cfg.exists():
        with open(cfg) as f:
            return json.load(f).get("dataset")
    name = metrics_path.parent.name
    if name == "sakhalin_unified":
        return "sakhalin"
    if name in {"nsk", "sakhalin"}:
        return name
    return None


def _experiment_label(metrics_path: pathlib.Path) -> tuple[str, str | None]:
    # exp_006_..._with_wind__with_wind -> (exp_006_..._with_wind, with_wind)
    for part in metrics_path.parents:
        if part.name.startswith("exp_"):
            name = part.name
            if "__" in name:
                base, suffix = name.split("__", 1)
                return base, suffix
            return name, None
    return metrics_path.parent.name, None


def collect() -> pd.DataFrame:
    rows = []
    for metrics_path in EXP_ROOT.glob("exp_*/**/metrics.json"):
        with open(metrics_path) as f:
            m = json.load(f)
        base, suffix = _experiment_label(metrics_path)
        dataset = _infer_dataset(metrics_path) or "unknown"
        display = DISPLAY_NAMES.get(base, base)
        if suffix:
            display = f"{display} [{suffix}]"
        rows.append({
            "experiment": base if suffix is None else f"{base}__{suffix}",
            "display": display,
            "dataset": dataset,
            "mean_error_cells": m.get("mean_error"),
            "median_error_cells": m.get("median_error"),
            "std_error_cells": m.get("std_error"),
            "mean_smooth_error_cells": m.get("mean_smooth_error"),
            "n_samples": m.get("n"),
            "metrics_path": str(metrics_path.relative_to(EXP_ROOT.parent)),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "mean_error_cells"]).reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no metrics.json found yet - run some experiments first_\n"
    chunks = []
    for ds in sorted(df["dataset"].unique()):
        chunks.append(f"## Dataset: {ds}\n")
        sub = df[df["dataset"] == ds][[
            "display", "mean_error_cells", "median_error_cells",
            "std_error_cells", "mean_smooth_error_cells", "n_samples",
        ]].rename(columns={
            "display": "method",
            "mean_error_cells": "mean err",
            "median_error_cells": "median",
            "std_error_cells": "std",
            "mean_smooth_error_cells": "smooth mean",
            "n_samples": "n",
        })
        chunks.append(sub.to_markdown(index=False, floatfmt=".2f") + "\n")
    return "\n".join(chunks)


def main() -> None:
    df = collect()
    md = to_markdown(df)
    print(md)
    (EXP_ROOT / "RESULTS.md").write_text(
        "# Comparison of source-localization experiments\n\n"
        "Each row is one method on one dataset. Metric is Euclidean error in grid cells\n"
        "(lower is better). All experiments share the same per-dataset test split (seed=42).\n\n"
        + md
    )
    if not df.empty:
        df.to_csv(EXP_ROOT / "RESULTS.csv", index=False)


if __name__ == "__main__":
    main()
