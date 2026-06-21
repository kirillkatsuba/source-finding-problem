"""exp_001 (unified): физический baseline (обратная адвекция) на тест-сплите sakhalin, все 19 источников."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_001_baseline.run_baseline import (
    compute_grid_spacing,
    parse_time,
    solve_backwards,
)
from tools.dataset import true_source_xy
from tools.metrics import summarize
from tools.splits import classify_files, split_files, wind_path_for


def _argmax_xy(field: np.ndarray) -> tuple[int, int]:
    h, w = field.shape
    flat = int(np.argmax(field))
    y, x = divmod(flat, w)
    return int(x), int(y)


def predict_one(pol_path: pathlib.Path,
                wind_path: pathlib.Path,
                release_idx: int,
                level_idx: int = 0,
                smooth_sigma: float = 2.0) -> dict | None:
    with xr.open_dataset(pol_path) as ds_pol, xr.open_dataset(wind_path) as ds_wind:
        conc = ds_pol["CONC"]
        true_x, true_y = true_source_xy(conc, release_idx)

        dx, dy = compute_grid_spacing(ds_pol)
        t_target = parse_time(ds_pol["Time"][0].values)
        t_known = parse_time(ds_pol["Time"][1].values)
        if t_target is None or t_known is None:
            return None
        time_gap = (t_known - t_target).total_seconds()

        C_known = conc.isel(
            Time=1, releases=release_idx, bottom_top=level_idx, species=0,
        ).values
        U_field = ds_wind["U10"].isel(Time=0).values
        V_field = ds_wind["V10"].isel(Time=0).values

        min_y = min(C_known.shape[0], U_field.shape[0])
        min_x = min(C_known.shape[1], U_field.shape[1])
        C_known = C_known[:min_y, :min_x]
        U_field = U_field[:min_y, :min_x]
        V_field = V_field[:min_y, :min_x]

        # число субшагов по CFL
        max_vel = np.max(np.sqrt(U_field ** 2 + V_field ** 2)) + 1e-6
        dt_cfl = 0.5 * min(dx, dy) / max_vel
        steps = int(np.ceil(time_gap / dt_cfl))
        dt_sub = time_gap / steps

        C_rec = C_known.copy()
        for _ in range(steps):
            C_rec = solve_backwards(C_rec, U_field, V_field, dx, dy, dt_sub)

        C_smooth = gaussian_filter(C_rec, sigma=smooth_sigma)
        pred_x, pred_y = _argmax_xy(C_rec)
        smooth_x, smooth_y = _argmax_xy(C_smooth)
    return {
        "file": pol_path.name,
        "source_idx": release_idx,
        "pred_x": pred_x, "pred_y": pred_y,
        "smooth_pred_x": smooth_x, "smooth_pred_y": smooth_y,
        "true_x": true_x, "true_y": true_y,
        "error": float(np.linalg.norm([pred_x - true_x, pred_y - true_y])),
        "smooth_error": float(np.linalg.norm([smooth_x - true_x, smooth_y - true_y])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smooth-sigma", type=float, default=2.0)
    args = p.parse_args()

    out_dir = pathlib.Path(__file__).resolve().parent / "sakhalin_unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = classify_files()
    split = split_files(groups["sakhalin"], seed=args.seed)

    rows: list[dict] = []
    for pol_path in tqdm(split.test, desc="files"):
        wind_path = wind_path_for(pol_path)
        if wind_path is None:
            continue
        with xr.open_dataset(pol_path) as ds:
            n_releases = ds["CONC"].shape[2]
        for r in range(n_releases):
            res = predict_one(pol_path, wind_path, r, smooth_sigma=args.smooth_sigma)
            if res is not None:
                rows.append(res)

    if not rows:
        print("no predictions")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)
    pred_xy = df[["pred_x", "pred_y"]].to_numpy()
    pred_xy_smooth = df[["smooth_pred_x", "smooth_pred_y"]].to_numpy()
    true_xy = df[["true_x", "true_y"]].to_numpy()
    stats = summarize(pred_xy, true_xy, pred_xy_smooth)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(asdict(stats), f, indent=2)
    print(f"physical baseline (sakhalin): mean_err={stats.mean_error:.2f} "
          f"median={stats.median_error:.2f} smooth={stats.mean_smooth_error:.2f} (n={stats.n})")


if __name__ == "__main__":
    main()
