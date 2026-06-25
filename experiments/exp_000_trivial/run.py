"""exp_000: trivial baseline - argmax последнего наблюдаемого кадра (t=17)."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dataset import SourceDataset
from tools.metrics import batch_field_to_xy, summarize
from tools.splits import classify_files, n_sources_for, split_files, split_sources


def run(dataset_kind: str, seed: int = 42, smooth_sigma: float = 2.0,
        source_split: bool = False) -> None:
    out_dir = pathlib.Path(__file__).resolve().parent / dataset_kind
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = classify_files()
    if source_split:
        _, test_src = split_sources(n_sources_for(dataset_kind), seed=seed)
        test_set = SourceDataset(groups[dataset_kind], dataset_kind=dataset_kind,
                                 source_indices=test_src, quiet=False)
    else:
        split = split_files(groups[dataset_kind], seed=seed)
        test_set = SourceDataset(split.test, dataset_kind=dataset_kind, quiet=False)

    pred_xy = np.zeros((len(test_set), 2), dtype=np.int64)
    pred_xy_smooth = np.zeros_like(pred_xy)
    true_xy = np.zeros_like(pred_xy)
    rows = []
    for i in range(len(test_set)):
        s = test_set.samples[i]
        last_frame = s.field_input[-1]
        pred_xy[i] = batch_field_to_xy(last_frame[None], sigma=None)[0]
        pred_xy_smooth[i] = batch_field_to_xy(last_frame[None], sigma=smooth_sigma)[0]
        true_xy[i] = s.coords
        rows.append({
            "file": s.file,
            "source_idx": s.source_idx,
            "pred_x": int(pred_xy[i, 0]), "pred_y": int(pred_xy[i, 1]),
            "smooth_pred_x": int(pred_xy_smooth[i, 0]), "smooth_pred_y": int(pred_xy_smooth[i, 1]),
            "true_x": int(true_xy[i, 0]), "true_y": int(true_xy[i, 1]),
            "error": float(np.linalg.norm(pred_xy[i] - true_xy[i])),
            "smooth_error": float(np.linalg.norm(pred_xy_smooth[i] - true_xy[i])),
        })
    stats = summarize(pred_xy, true_xy, pred_xy_smooth)
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(asdict(stats), f, indent=2)
    print(f"{dataset_kind}: mean_err={stats.mean_error:.2f} median={stats.median_error:.2f} (n={stats.n})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["nsk", "sakhalin", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smooth-sigma", type=float, default=2.0)
    p.add_argument("--source-split", action="store_true")
    args = p.parse_args()
    kinds = ["nsk", "sakhalin"] if args.dataset == "both" else [args.dataset]
    for k in kinds:
        run(k, seed=args.seed, smooth_sigma=args.smooth_sigma, source_split=args.source_split)


if __name__ == "__main__":
    main()
