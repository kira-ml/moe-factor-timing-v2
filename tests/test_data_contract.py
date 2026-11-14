from __future__ import annotations

import pandas as pd

from moe_factor_timing.data.schema import MarketData
from moe_factor_timing.data.validate import validate_market_data


def test_reference_panel_passes_all_checks(market, default_config):
    report = validate_market_data(market, default_config)
    assert report.passed, report.to_dict()
    assert market.factors.shape == (759, 6)
    assert str(market.start.date()) == "1963-07-31"
    assert str(market.end.date()) == "2026-09-30"
    assert market.risk_free.index.equals(market.factors.index)


def test_nan_fails(market, default_config):
    factors = market.factors.copy()
    factors.iloc[10, 0] = float("nan")
    broken = MarketData(factors=factors, risk_free=market.risk_free, metadata={})
    report = validate_market_data(broken, default_config)
    assert not report.passed
    assert any(check.name == "no_missing_values" and not check.passed for check in report.checks)


def test_missing_month_fails(market, default_config):
    factors = market.factors.drop(market.factors.index[10])
    broken = MarketData(
        factors=factors, risk_free=market.risk_free.drop(market.risk_free.index[10]), metadata={}
    )
    report = validate_market_data(broken, default_config)
    assert not report.passed
    assert any(check.name == "continuous_monthly" and not check.passed for check in report.checks)


def test_duplicate_date_fails(market, default_config):
    factors = pd.concat([market.factors, market.factors.iloc[[10]]]).sort_index()
    risk_free = pd.concat([market.risk_free, market.risk_free.iloc[[10]]]).sort_index()
    broken = MarketData(factors=factors, risk_free=risk_free, metadata={})
    report = validate_market_data(broken, default_config)
    assert not report.passed
    assert any(check.name == "monotonic_unique" and not check.passed for check in report.checks)


def test_short_history_fails(market, default_config):
    factors = market.factors.iloc[:100]
    risk_free = market.risk_free.iloc[:100]
    broken = MarketData(factors=factors, risk_free=risk_free, metadata={})
    report = validate_market_data(broken, default_config)
    assert not report.passed
    assert any(check.name == "minimum_history" and not check.passed for check in report.checks)
