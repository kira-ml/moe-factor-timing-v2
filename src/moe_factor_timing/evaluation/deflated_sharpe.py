"""Probabilistic and deflated Sharpe ratios.

The Probabilistic Sharpe Ratio (PSR) is the probability that the true Sharpe
exceeds a benchmark, accounting for skewness, kurtosis, and sample length. The
Deflated Sharpe Ratio (DSR) additionally accounts for the number of trials and
the dispersion of trial Sharpe ratios.

Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
Journal of Portfolio Management, 40(5), 94-107.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def _monthly_sharpe(annualised_sharpe: float) -> float:
    return annualised_sharpe / np.sqrt(12.0)


def probabilistic_sharpe_ratio(
    annualised_sharpe: float,
    n_months: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    """PSR: probability that the true Sharpe exceeds ``benchmark_sharpe``."""
    if n_months <= 2 or not np.isfinite(annualised_sharpe):
        return float("nan")
    sr = _monthly_sharpe(annualised_sharpe)
    sr0 = _monthly_sharpe(benchmark_sharpe)
    denominator = np.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2, 1e-12))
    z = (sr - sr0) * np.sqrt(n_months - 1) / denominator
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Expected maximum of ``n_trials`` independent Sharpe estimates under the null."""
    if n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(trial_sharpe_std * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    annualised_sharpe: float,
    n_months: int,
    n_trials: int,
    trial_sharpe_std: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> dict[str, float]:
    sr0 = expected_max_sharpe(n_trials, trial_sharpe_std)
    psr = probabilistic_sharpe_ratio(
        annualised_sharpe, n_months, skew=skew, kurt=kurt, benchmark_sharpe=0.0
    )
    dsr = probabilistic_sharpe_ratio(
        annualised_sharpe, n_months, skew=skew, kurt=kurt, benchmark_sharpe=sr0
    )
    return {
        "psr": psr,
        "dsr": dsr,
        "dsr_p_value": float(1.0 - dsr) if np.isfinite(dsr) else float("nan"),
        "expected_max_sharpe": float(sr0),
        "n_trials": int(n_trials),
        "trial_sharpe_std": float(trial_sharpe_std),
    }


def deflated_sharpe_table(metrics: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    """Compute PSR/DSR for every model using the cross-section of trial Sharpes.

    ``trial_sharpe_std`` is estimated from the dispersion of model Sharpes, with
    a floor of the null standard error ``1/sqrt(years)``. This is conservative:
    it treats the model set as the trial set and does not assume independence.
    """
    if metrics.empty:
        return pd.DataFrame()
    sharpes = metrics["sharpe"].dropna().to_numpy(dtype=float)
    n_months = int(metrics["n_obs"].max())
    years = max(n_months / 12.0, 1.0)
    floor = 1.0 / np.sqrt(years)
    trial_std = float(max(np.std(sharpes, ddof=1) if len(sharpes) > 1 else 0.0, floor))
    rows: dict[str, dict] = {}
    for name, row in metrics.iterrows():
        rows[name] = deflated_sharpe_ratio(
            annualised_sharpe=float(row["sharpe"]),
            n_months=int(row["n_obs"]),
            n_trials=n_trials,
            trial_sharpe_std=trial_std,
            skew=float(row.get("skew", 0.0)),
            kurt=float(row.get("excess_kurtosis", 0.0)) + 3.0,
        )
    frame = pd.DataFrame(rows).T
    frame.index.name = "model"
    return frame
