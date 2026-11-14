from __future__ import annotations

import numpy as np

from moe_factor_timing.config import ConstraintConfig, CostConfig, DecisionConfig, SolverConfig
from moe_factor_timing.decision.mean_variance import solve
from moe_factor_timing.utils.weights import cap_and_normalise, enforce_turnover_cap


def _config(fully_invested=True, turnover_cap=None, max_weight=0.35):
    return DecisionConfig(
        utility="mean_variance",
        risk_aversion=2.0,
        constraints=ConstraintConfig(
            long_only=True,
            fully_invested=fully_invested,
            max_weight=max_weight,
            turnover_cap=turnover_cap,
        ),
        cost=CostConfig(one_way_bps=10.0, strategy_drag_bps_per_month=0.0),
        solver=SolverConfig(name="slsqp", max_iter=500, ftol=1e-10, fallback_max_rate=0.05),
    )


def _problem(n=6, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.005, 0.004, size=n)
    A = rng.normal(size=(n, n))
    cov = A @ A.T * 1e-4
    return mu, cov


def test_fully_invested_solution_is_feasible_and_deterministic():
    mu, cov = _problem()
    w_prev = np.ones(6) / 6
    first = solve(mu, cov, w_prev, _config())
    second = solve(mu, cov, w_prev, _config())
    assert first.status == "optimal"
    assert first.weights.shape == (6,)
    assert np.isclose(first.weights.sum(), 1.0)
    assert first.weights.min() >= -1e-12
    assert first.weights.max() <= 0.35 + 1e-6
    np.testing.assert_allclose(first.weights, second.weights, atol=1e-12)


def test_cash_allowed_solution_respects_sum():
    mu, cov = _problem()
    result = solve(mu, cov, np.zeros(6), _config(fully_invested=False))
    assert result.status == "optimal"
    assert result.weights.sum() <= 1.0 + 1e-9
    assert result.weights.min() >= -1e-12


def test_turnover_cap_is_respected():
    mu = np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0])
    cov = np.eye(6) * 1e-3
    w_prev = np.array([0.35, 0.35, 0.0, 0.30, 0.0, 0.0])
    result = solve(mu, cov, w_prev, _config(turnover_cap=0.20))
    traded = np.abs(result.weights - w_prev).sum()
    assert traded <= 0.20 + 1e-5


def test_nonfinite_input_falls_back():
    mu = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])
    cov = np.eye(6)
    result = solve(mu, cov, np.ones(6) / 6, _config())
    assert result.status.startswith("fallback")


def test_cap_and_normalise_waterfills():
    weights = cap_and_normalise(np.array([0.90, 0.05, 0.05]), max_weight=0.40)
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.40 + 1e-12


def test_cap_and_normalise_handles_zero_free_weights():
    weights = cap_and_normalise(np.array([1.0, 0.0, 0.0]), max_weight=0.35, target_sum=1.0)
    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.35 + 1e-12


def test_enforce_turnover_cap_moves_toward_previous():
    w_prev = np.array([0.5, 0.5, 0.0])
    target = np.array([0.0, 0.0, 1.0])
    adjusted = enforce_turnover_cap(target, w_prev, 0.20)
    assert np.abs(adjusted - w_prev).sum() <= 0.20 + 1e-12
    assert np.isclose(adjusted.sum(), 1.0)
