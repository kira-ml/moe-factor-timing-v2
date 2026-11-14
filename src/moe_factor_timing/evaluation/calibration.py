"""Calibration diagnostics for probabilistic forecasts.

The PIT is computed from the same predictive distribution used by the
decision layer. A well-calibrated model has uniformly distributed PIT values;
the KS test is reported as a diagnostic, not as proof of decision value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kstest


def calibration_by_model(model_results: dict) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for name, result in model_results.items():
        if result.pit is None or result.nll is None:
            continue
        pit = result.pit.to_numpy(dtype=float)
        flat = pit.reshape(-1)
        pooled = kstest(flat, "uniform")
        per_asset_p = [kstest(pit[:, j], "uniform").pvalue for j in range(pit.shape[1])]
        rows[name] = {
            "n_obs": int(len(result.pit)),
            "mean_nll": float(result.nll.mean()),
            "mean_pit": float(np.mean(flat)),
            "pit_below_5pct": float(np.mean(flat < 0.05)),
            "pit_above_95pct": float(np.mean(flat > 0.95)),
            "ks_pooled_stat": float(pooled.statistic),
            "ks_pooled_p": float(pooled.pvalue),
            "ks_asset_min_p": float(np.min(per_asset_p)),
            "ks_assets_passing_5pct": int(np.sum(np.array(per_asset_p) >= 0.05)),
            "n_assets": int(pit.shape[1]),
        }
    frame = pd.DataFrame(rows).T
    frame.index.name = "model"
    return frame.astype(float)


def calibration_by_asset(model_results: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for name, result in model_results.items():
        if result.pit is None:
            continue
        for asset in result.pit.columns:
            test = kstest(result.pit[asset].to_numpy(dtype=float), "uniform")
            rows.append(
                {
                    "model": name,
                    "asset": asset,
                    "ks_stat": float(test.statistic),
                    "ks_p": float(test.pvalue),
                    "mean_pit": float(result.pit[asset].mean()),
                }
            )
    return pd.DataFrame(rows)


def pit_histogram(model_results: dict, bins: int = 10) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, result in model_results.items():
        if result.pit is None:
            continue
        values = result.pit.to_numpy(dtype=float).reshape(-1)
        counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
        histograms[name] = (counts, edges)
    return histograms
