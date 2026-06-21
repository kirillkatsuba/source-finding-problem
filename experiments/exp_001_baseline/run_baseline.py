import os
import sys
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dataset import true_source_xy

PLOTS_DIR = Path("experiments/exp_001_baseline/plots")
PREDICTIONS_CSV_PATH = Path("experiments/exp_001_baseline/predictions.csv")
POLLUTION_DIR = Path("data/pollution")
WIND_DIR = Path("data/wind")


class Prediction(TypedDict):
    file: str
    pred_x: int
    pred_y: int
    smooth_pred_x: int
    smooth_pred_y: int
    true_x: int
    true_y: int
    error: float
    smooth_error: float
    C_known: np.ndarray
    C_reconstructed: np.ndarray
    C_real_0: np.ndarray


def parse_time(t_raw: Any) -> pd.Timestamp | None:
    """Parse model time values into pandas datetime."""
    if np.issubdtype(np.asarray(t_raw).dtype, np.datetime64):
        return pd.to_datetime(t_raw)

    try:
        t_str = np.asarray(t_raw).item()
        if isinstance(t_str, bytes):
            t_str = t_str.decode("utf-8")
        return pd.to_datetime(t_str, format="%Y%m%d_%H%M%S")
    except Exception:
        return None


def get_true_source_coordinates(nc_path: Path | str, release_idx: int = 0) -> tuple[int, int]:
    """Истинный источник из attrs (точка выброса), не argmax поля."""
    with xr.open_dataset(nc_path) as ds:
        return true_source_xy(ds["CONC"], release_idx)


def solve_backwards(
    C: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
) -> np.ndarray:
    """Move concentration backward in time using reversed wind and upwind advection."""
    u_rev = -U
    v_rev = -V

    dC_dx_bwd = (C - np.roll(C, 1, axis=1)) / dx
    dC_dx_fwd = (np.roll(C, -1, axis=1) - C) / dx
    adv_x = (np.maximum(u_rev, 0) * dC_dx_bwd) + (np.minimum(u_rev, 0) * dC_dx_fwd)

    dC_dy_bwd = (C - np.roll(C, 1, axis=0)) / dy
    dC_dy_fwd = (np.roll(C, -1, axis=0) - C) / dy
    adv_y = (np.maximum(v_rev, 0) * dC_dy_bwd) + (np.minimum(v_rev, 0) * dC_dy_fwd)

    change = -(adv_x + adv_y)
    C_new = C + dt * change

    C_new[0, :] = 0
    C_new[-1, :] = 0
    C_new[:, 0] = 0
    C_new[:, -1] = 0
    C_new[C_new < 0] = 0
    return C_new


def compute_grid_spacing(ds_pol: xr.Dataset) -> tuple[float, float]:
    """Estimate grid spacing in meters from latitude/longitude coordinates."""
    lat = ds_pol["south_north"].values
    lon = ds_pol["west_east"].values
    mean_lat = np.mean(lat)
    dy = np.abs(lat[1] - lat[0]) * 111320.0
    dx = np.abs(lon[1] - lon[0]) * 40075000.0 * np.cos(np.deg2rad(mean_lat)) / 360.0
    return float(dx), float(dy)


def build_wind_path(pol_file_name: str) -> Path:
    """Build expected wind file path for a given pollution file name."""
    return WIND_DIR / f"wind_for_{pol_file_name}"


def find_source_location(
    pol_file: Path | str,
    wind_file: Path | str,
    mean_error: list[float],
    mean_smooth_error: list[float],
    release_idx: int = 0,
    level_idx: int = 0,
) -> Prediction:
    with xr.open_dataset(pol_file) as ds_pol, xr.open_dataset(wind_file) as ds_wind:
        real_x, real_y = get_true_source_coordinates(pol_file, release_idx=release_idx)

        idx_known = 1
        idx_target = 0

        dx, dy = compute_grid_spacing(ds_pol)

        t_target = parse_time(ds_pol["Time"][idx_target].values)
        t_known = parse_time(ds_pol["Time"][idx_known].values)
        if t_target is None or t_known is None:
            raise ValueError(f"Failed to parse time values in {Path(pol_file).name}")
        time_gap = (t_known - t_target).total_seconds()

        C_known = ds_pol["CONC"].isel(
            Time=idx_known,
            releases=release_idx,
            bottom_top=level_idx,
            species=0,
        ).values

        U_field = ds_wind["U10"].isel(Time=idx_target).values
        V_field = ds_wind["V10"].isel(Time=idx_target).values

        min_y = min(C_known.shape[0], U_field.shape[0])
        min_x = min(C_known.shape[1], U_field.shape[1])
        C_known = C_known[:min_y, :min_x]
        U_field = U_field[:min_y, :min_x]
        V_field = V_field[:min_y, :min_x]

        max_vel = np.max(np.sqrt(U_field ** 2 + V_field ** 2)) + 1e-6
        dt_cfl = 0.5 * min(dx, dy) / max_vel
        steps = int(np.ceil(time_gap / dt_cfl))
        dt_sub = time_gap / steps

        C_reconstructed = C_known.copy()
        for _ in range(steps):
            C_reconstructed = solve_backwards(C_reconstructed, U_field, V_field, dx, dy, dt_sub)

        C_smoothed = gaussian_filter(C_reconstructed, sigma=2.0)

        y_src, x_src = np.unravel_index(np.argmax(C_reconstructed), C_reconstructed.shape)
        y_smooth, x_smooth = np.unravel_index(np.argmax(C_smoothed), C_smoothed.shape)

        error_dist = np.sqrt((x_src - real_x) ** 2 + (y_src - real_y) ** 2)
        error_dist_smooth = np.sqrt((x_smooth - real_x) ** 2 + (y_smooth - real_y) ** 2)
        mean_error.append(error_dist)
        mean_smooth_error.append(error_dist_smooth)

        C_real_0 = ds_pol["CONC"].isel(
            Time=idx_target,
            releases=release_idx,
            bottom_top=level_idx,
            species=0,
        ).values
        C_real_0 = C_real_0[:min_y, :min_x]

        prediction: Prediction = {
            "file": Path(pol_file).name,
            "pred_x": int(x_src),
            "pred_y": int(y_src),
            "smooth_pred_x": int(x_smooth),
            "smooth_pred_y": int(y_smooth),
            "true_x": int(real_x),
            "true_y": int(real_y),
            "error": float(error_dist),
            "smooth_error": float(error_dist_smooth),
            "C_known": C_known,
            "C_reconstructed": C_reconstructed,
            "C_real_0": C_real_0,
        }

        return prediction


def save_prediction_plot(prediction: Prediction, category: str) -> None:
    """Save one prediction plot with a category-specific file name."""
    x_src = prediction["pred_x"]
    y_src = prediction["pred_y"]
    x_smooth = prediction["smooth_pred_x"]
    y_smooth = prediction["smooth_pred_y"]
    real_x = prediction["true_x"]
    real_y = prediction["true_y"]
    error_dist = prediction["error"]
    error_dist_smooth = prediction["smooth_error"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(prediction["C_known"], origin="lower", cmap="viridis")
    axes[0].set_title("Input frame (t=1)")

    axes[1].imshow(prediction["C_reconstructed"], origin="lower", cmap="plasma")
    axes[1].scatter(x_src, y_src, c="red", marker="o", s=30, label="Prediction")
    axes[1].scatter(x_smooth, y_smooth, c="yellow", marker="o", s=30, label="Smoothed prediction")
    axes[1].scatter(real_x, real_y, c="white", marker="o", s=30, label="True source")
    axes[1].set_title(
        "Reconstruction\n"
        f"Error: {error_dist:.2f} cells, Smoothed error: {error_dist_smooth:.2f} cells"
    )
    axes[1].legend()

    axes[2].imshow(prediction["C_real_0"], origin="lower", cmap="viridis")
    axes[2].set_title("Ground truth at t=0")

    stem = Path(prediction["file"]).stem
    plot_path = PLOTS_DIR / f"{category}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    mean_error: list[float] = []
    mean_smooth_error: list[float] = []
    predictions: list[Prediction] = []

    for pol_file_name in tqdm(os.listdir(POLLUTION_DIR)):
        if not pol_file_name.endswith(".nc"):
            continue

        pol_path = POLLUTION_DIR / pol_file_name
        wind_path = build_wind_path(pol_file_name)

        if not wind_path.exists():
            continue

        prediction = find_source_location(
            pol_path,
            wind_path,
            mean_error,
            mean_smooth_error,
        )
        predictions.append(prediction)

    if mean_error:
        print(f"Mean error across all files: {np.mean(mean_error):.4f} cells")
        print(f"Mean smoothed error across all files: {np.mean(mean_smooth_error):.4f} cells")

        errors = np.array([p["error"] for p in predictions], dtype=float)
        smooth_errors = np.array([p["smooth_error"] for p in predictions], dtype=float)
        improvements = errors - smooth_errors

        selected_cases = [
            ("min_error_raw", int(np.argmin(errors))),
            ("min_error_smooth", int(np.argmin(smooth_errors))),
            ("max_error_raw", int(np.argmax(errors))),
            ("max_error_smooth", int(np.argmax(smooth_errors))),
            ("max_smoothing_improvement", int(np.argmax(improvements))),
        ]

        for category, idx in selected_cases:
            save_prediction_plot(predictions[idx], category)
    else:
        print("No valid predictions were produced.")

    pred_df = pd.DataFrame(
        [
            {
                "file": p["file"],
                "pred_x": p["pred_x"],
                "pred_y": p["pred_y"],
                "smooth_pred_x": p["smooth_pred_x"],
                "smooth_pred_y": p["smooth_pred_y"],
                "true_x": p["true_x"],
                "true_y": p["true_y"],
                "error": p["error"],
                "smooth_error": p["smooth_error"],
            }
            for p in predictions
        ]
    )
    pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
