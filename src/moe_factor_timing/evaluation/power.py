"""Statistical power for Sharpe-ratio comparisons.

A 49-month out-of-sample window cannot distinguish a Sharpe of 1.3 from 1.1
with meaningful confidence. Reporting the minimum detectable effect is part of
the result, not an appendix.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

TRADING_MONTHS = 12


def sharpe_standard_error(
    sharpe: float, n_months: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """Lo (2002) standard error of an annualised Sharpe ratio."""
    if n_months <= 1:
        return float("nan")
    years = n_months / TRADING_MONTHS
    variance = (
        1.0
        + 0.5 * sharpe**2
        - skew * sharpe / np.sqrt(TRADING_MONTHS)
        + ((kurt - 3.0) / 4.0) * sharpe**2 / TRADING_MONTHS
    )
    return float(np.sqrt(max(variance, 0.0) / years))


def minimum_detectable_sharpe(n_months: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Approximate MDE for a two-sided test of Sharpe against zero.

    Under the null the annualised Sharpe estimator has standard error
    ``1/sqrt(years)``. This is a conservative reference for comparing two
    strategies, because the paired difference has a smaller variance when the
    strategies are correlated.
    """
    if n_months <= 0:
        return float("nan")
    years = n_months / TRADING_MONTHS
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_power = norm.ppf(power)
    return float((z_alpha + z_power) / np.sqrt(years))


def power_summary(
    n_months_full: int,
    n_months_holdout: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float]:
    return {
        "n_months_full": int(n_months_full),
        "n_years_full": float(n_months_full / TRADING_MONTHS),
        "mde_full": minimum_detectable_sharpe(n_months_full, alpha, power),
        "n_months_holdout": int(n_months_holdout),
        "n_years_holdout": float(n_months_holdout / TRADING_MONTHS),
        "mde_holdout": minimum_detectable_sharpe(n_months_holdout, alpha, power),
        "alpha": float(alpha),
        "power": float(power),
    }
