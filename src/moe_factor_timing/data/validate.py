"""Contract checks for the market panel.

The checks are deliberately conservative: a failed check stops the run before
any model sees the data. Warnings are recorded but do not silently pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .schema import MarketData, ValidationCheck, ValidationReport


def validate_market_data(
    market: MarketData, config: Config, min_rows: int = 240
) -> ValidationReport:
    factors = market.factors
    risk_free = market.risk_free
    checks: list[ValidationCheck] = []

    checks.append(
        ValidationCheck(
            name="columns_present",
            passed=set(config.data.factors).issubset(factors.columns)
            and config.data.risk_free == risk_free.name,
            detail=f"factors={list(factors.columns)} risk_free={risk_free.name}",
        )
    )
    checks.append(
        ValidationCheck(
            name="datetime_index",
            passed=isinstance(factors.index, pd.DatetimeIndex),
            detail=f"index_type={type(factors.index).__name__}",
        )
    )
    checks.append(
        ValidationCheck(
            name="monotonic_unique",
            passed=factors.index.is_monotonic_increasing and factors.index.is_unique,
            detail=f"monotonic={factors.index.is_monotonic_increasing} unique={factors.index.is_unique}",
        )
    )

    periods = factors.index.to_period("M")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    checks.append(
        ValidationCheck(
            name="continuous_monthly",
            passed=len(periods) == len(expected) and (periods == expected).all(),
            detail=f"observed={len(periods)} expected={len(expected)}",
        )
    )
    month_ends = factors.index + pd.offsets.MonthEnd(0) == factors.index
    checks.append(
        ValidationCheck(
            name="month_end_dates",
            passed=bool(month_ends.all()),
            detail="all observation dates are month-end",
        )
    )
    checks.append(
        ValidationCheck(
            name="minimum_history",
            passed=len(factors) >= min_rows,
            detail=f"rows={len(factors)} minimum={min_rows}",
        )
    )
    checks.append(
        ValidationCheck(
            name="no_missing_values",
            passed=bool(factors.notna().all().all() and risk_free.notna().all()),
            detail=f"factor_nan={int(factors.isna().sum().sum())} rf_nan={int(risk_free.isna().sum())}",
        )
    )
    finite = np.isfinite(factors.to_numpy()).all() and np.isfinite(risk_free.to_numpy()).all()
    checks.append(ValidationCheck(name="finite_values", passed=bool(finite), detail=""))
    checks.append(
        ValidationCheck(
            name="risk_free_plausible",
            passed=bool((risk_free.abs() < 0.10).all()),
            detail=f"min={risk_free.min():.4f} max={risk_free.max():.4f}",
        )
    )
    non_degenerate = (factors.std(ddof=1) > 1e-8).all()
    checks.append(
        ValidationCheck(
            name="non_degenerate_factors",
            passed=bool(non_degenerate),
            detail="all factor return series have non-zero variance",
        )
    )
    extremes = (factors.abs() > 1.0).sum().sum()
    checks.append(
        ValidationCheck(
            name="no_implausible_extremes",
            passed=bool(extremes == 0),
            detail=f"observations with |monthly return| > 100%: {int(extremes)}",
        )
    )
    aligned = factors.index.equals(risk_free.index)
    checks.append(
        ValidationCheck(
            name="risk_free_aligned",
            passed=bool(aligned),
            detail="risk-free series shares the factor index",
        )
    )

    return ValidationReport(checks=tuple(checks))
