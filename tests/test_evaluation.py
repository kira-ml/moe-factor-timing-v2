from __future__ import annotations

import numpy as np
import pandas as pd

from moe_factor_timing.evaluation.bootstrap import sharpe_difference
from moe_factor_timing.evaluation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from moe_factor_timing.evaluation.metrics import (
    annualized_return,
    cost_sensitivity_frame,
    maximum_drawdown,
    sharpe_ratio,
)
from moe_factor_timing.evaluation.power import minimum_detectable_sharpe, power_summary


def test_annualized_return_compounds():
    returns = pd.Series([0.01] * 12)
    assert np.isclose(annualized_return(returns), (1.01**12) - 1.0)


def test_max_drawdown_known_path():
    path = pd.Series([0.10, -0.20, 0.05])
    assert np.isclose(maximum_drawdown(path), -0.20)


def test_constant_returns_have_no_sharpe():
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 12)))


def test_bootstrap_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(4)
    model = pd.Series(rng.normal(0.006, 0.03, size=240))
    benchmark = pd.Series(rng.normal(0.004, 0.03, size=240))
    first = sharpe_difference(model, benchmark, n_repetitions=200, block_length=6, seed=9)
    second = sharpe_difference(model, benchmark, n_repetitions=200, block_length=6, seed=9)
    assert first == second
    assert first["ci_low"] <= first["diff"] <= first["ci_high"]
    assert 0.0 <= first["p_value"] <= 1.0


def test_psr_and_dsr_bounds():
    assert 0.0 <= probabilistic_sharpe_ratio(1.0, 240, skew=0.0, kurt=3.0) <= 1.0
    assert probabilistic_sharpe_ratio(2.0, 240) > probabilistic_sharpe_ratio(0.1, 240)
    assert expected_max_sharpe(10, 0.2) > expected_max_sharpe(2, 0.2)
    result = deflated_sharpe_ratio(1.0, 240, n_trials=10, trial_sharpe_std=0.2)
    assert 0.0 <= result["dsr"] <= 1.0
    assert result["dsr"] <= result["psr"] + 1e-12


def test_power_decreases_with_sample_size():
    assert minimum_detectable_sharpe(600) < minimum_detectable_sharpe(120)
    summary = power_summary(600, 240)
    assert summary["n_years_full"] == 50.0
    assert summary["mde_holdout"] > summary["mde_full"]


def test_cost_sensitivity_monotonic():
    rng = np.random.default_rng(1)
    n = 120
    dummy = type(
        "Dummy",
        (),
        {
            "gross_excess": pd.Series(rng.normal(0.005, 0.02, n)),
            "traded_notional": pd.Series(np.full(n, 0.25)),
            "risk_free": pd.Series(np.full(n, 0.002)),
        },
    )()
    frame = cost_sensitivity_frame({"m": dummy}, cost_levels_bps=(0.0, 10.0, 20.0))
    sharpes = frame.set_index("cost_bps")["sharpe"]
    assert sharpes.loc[0.0] > sharpes.loc[10.0] > sharpes.loc[20.0]
