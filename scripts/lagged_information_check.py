"""Information-set lag check for the MoE paper.

The reference uses X_t (returns through month t) to predict r_{t+1}. This
script shifts the feature matrix one month so the decision at t uses
information through t-1, then refits the same walk-forward MoE. It reports
rank and Pearson IC for the full, pre-holdout, and holdout windows.

Usage:
    python scripts/lagged_information_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from moe_factor_timing.models.moe import MixtureOfExpertsModel

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "reference"
OUT = ROOT / "reports" / "lagged_information_check.json"
HOLDOUT = pd.Timestamp("2006-01-31")
MIN_TRAIN = 120


def _hac_t(x: np.ndarray, lags: int = 6) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    mu = x.mean()
    e = x - mu
    gamma0 = float((e * e).mean())
    var = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float((e[lag:] * e[:-lag]).mean())
        var += 2.0 * weight * gamma
    se = np.sqrt(max(var, 1e-18) / n)
    return float(mu / se) if se > 0 else float("nan")


def _ic_rows(forecast: np.ndarray, realized: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    rank = []
    linear = []
    for i in range(len(dates)):
        rho = spearmanr(forecast[i], realized[i]).statistic
        pear = pearsonr(forecast[i], realized[i]).statistic
        rank.append(float(rho) if np.isfinite(rho) else np.nan)
        linear.append(float(pear) if np.isfinite(pear) else np.nan)
    rank = np.asarray(rank)
    linear = np.asarray(linear)

    def block(mask: np.ndarray) -> dict:
        return {
            "n_months": int(mask.sum()),
            "rank_ic": float(np.nanmean(rank[mask])),
            "rank_hac_t": _hac_t(rank[mask]),
            "pearson_ic": float(np.nanmean(linear[mask])),
            "pearson_hac_t": _hac_t(linear[mask]),
        }

    return {
        "full": block(np.ones(len(dates), dtype=bool)),
        "pre_holdout": block(dates < HOLDOUT),
        "holdout": block(dates >= HOLDOUT),
    }


def main() -> int:
    X = pd.read_csv(RUN / "features" / "X.csv", index_col=0, parse_dates=True)
    y = pd.read_csv(RUN / "features" / "y.csv", index_col=0, parse_dates=True)
    X_lag = X.shift(1).iloc[1:]
    y = y.loc[X_lag.index]
    dates = pd.DatetimeIndex(X_lag.index)

    forecasts = []
    realized = []
    test_dates = []
    for t in range(MIN_TRAIN, len(X_lag)):
        model = MixtureOfExpertsModel(
            n_experts=3,
            n_restarts=1,
            max_iter=25,
            ridge_alpha=1.0,
            gating_l2=0.01,
            covariance_shrinkage=0.2,
            gating_steps=6,
            gating_lr=0.5,
            seed=42,
        )
        model.fit(X_lag.iloc[:t].to_numpy(float), y.iloc[:t].to_numpy(float))
        forecast = model.predict(X_lag.iloc[t].to_numpy(float))
        forecasts.append(np.asarray(forecast.mean, dtype=float))
        realized.append(y.iloc[t].to_numpy(float))
        test_dates.append(dates[t])
        if (t - MIN_TRAIN + 1) % 100 == 0:
            print(f"{t - MIN_TRAIN + 1}/{len(X_lag) - MIN_TRAIN}")

    payload = {
        "description": "Walk-forward MoE with features shifted one month (information through t-1).",
        "n_decisions": len(test_dates),
        "first_decision": str(test_dates[0].date()),
        "last_decision": str(test_dates[-1].date()),
        "holdout_start": str(HOLDOUT.date()),
        "ic": _ic_rows(np.asarray(forecasts), np.asarray(realized), pd.DatetimeIndex(test_dates)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
