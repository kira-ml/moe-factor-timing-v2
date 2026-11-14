from __future__ import annotations

from dataclasses import replace

import numpy as np
from conftest import synthetic_market

from moe_factor_timing.backtest.engine import _drift_weights, run_backtest
from moe_factor_timing.config import ModelConfig, SplitConfig
from moe_factor_timing.features.build import build_features
from moe_factor_timing.splits.walk_forward import make_walk_forward_splits


def test_drift_weights_are_wealth_weights():
    weights = np.array([0.5, 0.5])
    drifted = _drift_weights(weights, np.array([0.10, 0.0]), risk_free=0.0)
    assert drifted[0] > drifted[1]
    assert np.isclose(drifted.sum(), 1.0)


def test_drift_weights_with_zero_returns_are_unchanged():
    weights = np.array([0.6, 0.4])
    drifted = _drift_weights(weights, np.zeros(2), risk_free=0.01)
    np.testing.assert_allclose(drifted, weights)


def test_backtest_accounting_identity(default_config):
    market = synthetic_market(n_months=120, n_assets=4, seed=3)
    config = replace(
        default_config,
        data=replace(
            default_config.data,
            factors=("F0", "F1", "F2", "F3"),
            start=None,
            end=None,
        ),
        split=SplitConfig(min_train=36, test_step=1, holdout_start="2005-01-31"),
        models=(
            ModelConfig(name="one_over_n", kind="allocator", method="equal"),
            ModelConfig(name="cash", kind="allocator", method="cash"),
        ),
    )
    features = build_features(market, config)
    plan = make_walk_forward_splits(features.decision_dates, features.target_dates, config)
    suite = run_backtest(market, features, plan, config)

    result = suite.models["one_over_n"]
    w_prev_target = np.zeros(4)
    for i, split in enumerate(plan.splits):
        if i == 0:
            drifted = np.zeros(4)
        else:
            drifted = _drift_weights(
                w_prev_target,
                market.factors.loc[split.decision_date].to_numpy(dtype=float),
                float(market.risk_free.loc[split.decision_date]),
            )
        w = np.ones(4) / 4
        traded = float(np.abs(w - drifted).sum())
        cost = 10.0 / 10_000.0 * traded
        gross = float(w @ features.y.iloc[split.test_idx].to_numpy(dtype=float))
        net_total = float(market.risk_free.loc[split.target_date]) + gross - cost
        np.testing.assert_allclose(result.weights.iloc[i].to_numpy(), w)
        np.testing.assert_allclose(result.traded_notional.iloc[i], traded)
        np.testing.assert_allclose(result.costs.iloc[i], cost)
        np.testing.assert_allclose(result.net_total.iloc[i], net_total)
        w_prev_target = w

    assert np.isclose(result.traded_notional.iloc[0], 1.0)
    assert np.isclose(result.turnover.iloc[0], 0.5)
    assert (result.traded_notional.iloc[1:] > 0).any()
    assert result.fallback_count == 0

    cash = suite.models["cash"]
    np.testing.assert_allclose(cash.weights.to_numpy(), 0.0)
    np.testing.assert_allclose(cash.traded_notional.to_numpy(), 0.0)
    np.testing.assert_allclose(cash.costs.to_numpy(), 0.0)
    target_dates = [split.target_date for split in plan.splits]
    np.testing.assert_allclose(
        cash.net_total.to_numpy(),
        market.risk_free.reindex(target_dates).to_numpy(),
    )
