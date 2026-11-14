"""Constrained mean-variance allocation with turnover-aware costs.

The optimizer solves

    max_w  w'mu - (lambda/2) w'Sigma w - c * ||w - w_prev||_1 - drag * sum(w)
    s.t.   w >= 0, sum(w) = 1 (or <= 1), w_i <= max_weight,
           ||w - w_prev||_1 <= turnover_cap (optional)

The absolute-value term is represented with auxiliary variables so the problem
is a smooth convex quadratic program with linear constraints. SLSQP is used
because it is deterministic and dependency-light; failures are recorded and
never silently replaced by a zero return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..config import DecisionConfig
from ..utils.weights import cap_and_normalise, enforce_turnover_cap


@dataclass(frozen=True)
class DecisionResult:
    weights: np.ndarray
    status: str
    objective: float
    traded_notional: float
    message: str = ""


def _equal_weights(n: int, max_weight: float) -> np.ndarray:
    weights = np.full(n, 1.0 / n)
    if weights.max() > max_weight:
        weights = np.full(n, max_weight)
        weights = weights / weights.sum()
    return weights


def _feasible_start(w_prev: np.ndarray, config: DecisionConfig) -> np.ndarray:
    n = len(w_prev)
    constraints = config.constraints
    candidate = np.clip(w_prev, 0.0, constraints.max_weight)
    total = candidate.sum()
    target = 1.0 if constraints.fully_invested else min(1.0, total)
    if total <= 0:
        return _equal_weights(n, constraints.max_weight)
    candidate = candidate / total * target
    if candidate.max() > constraints.max_weight + 1e-12:
        return _equal_weights(n, constraints.max_weight)
    return candidate


def solve(
    mu: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray,
    config: DecisionConfig,
) -> DecisionResult:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float).reshape(-1)
    n = len(mu)
    if cov.shape != (n, n):
        raise ValueError(f"covariance shape {cov.shape} does not match mu length {n}")
    if w_prev.shape != (n,):
        raise ValueError("w_prev has the wrong length")
    cov = 0.5 * (cov + cov.T)

    constraints = config.constraints
    cost = config.cost.one_way_bps / 10_000.0
    drag = config.cost.strategy_drag_bps_per_month / 10_000.0
    lam = config.risk_aversion
    max_weight = constraints.max_weight

    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(cov)):
        return _fallback(w_prev, config, "fallback_nonfinite_input")

    w0 = _feasible_start(w_prev, config)
    t0 = np.abs(w0 - w_prev)
    x0 = np.concatenate([w0, t0])

    def objective(x: np.ndarray) -> float:
        w = x[:n]
        t = x[n:]
        return float(-(w @ mu - 0.5 * lam * w @ cov @ w - cost * t.sum() - drag * w.sum()))

    def gradient(x: np.ndarray) -> np.ndarray:
        w = x[:n]
        grad_w = -(mu - lam * cov @ w - drag * np.ones(n))
        grad_t = cost * np.ones(n)
        return np.concatenate([grad_w, grad_t])

    constraint_list: list[dict] = []
    if constraints.fully_invested:
        constraint_list.append(
            {
                "type": "eq",
                "fun": lambda x: float(x[:n].sum() - 1.0),
                "jac": lambda x: np.concatenate([np.ones(n), np.zeros(n)]),
            }
        )
    else:
        constraint_list.append(
            {
                "type": "ineq",
                "fun": lambda x: float(1.0 - x[:n].sum()),
                "jac": lambda x: np.concatenate([-np.ones(n), np.zeros(n)]),
            }
        )

    # w_i <= max_weight  ->  max_weight - w_i >= 0
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = -1.0
        constraint_list.append(
            {
                "type": "ineq",
                "fun": lambda x, i=i: float(max_weight - x[i]),
                "jac": lambda x, row=row: row,
            }
        )

    # t >= w - w_prev  ->  w_prev - w + t >= 0
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = -1.0
        row[n + i] = 1.0
        constraint_list.append(
            {
                "type": "ineq",
                "fun": lambda x, i=i: float(w_prev[i] - x[i] + x[n + i]),
                "jac": lambda x, row=row: row,
            }
        )
        # t >= w_prev - w  ->  w - w_prev + t >= 0
        row2 = np.zeros(2 * n)
        row2[i] = 1.0
        row2[n + i] = 1.0
        constraint_list.append(
            {
                "type": "ineq",
                "fun": lambda x, i=i: float(x[i] - w_prev[i] + x[n + i]),
                "jac": lambda x, row=row2: row,
            }
        )

    if constraints.turnover_cap is not None:
        cap_row = np.concatenate([np.zeros(n), -np.ones(n)])
        constraint_list.append(
            {
                "type": "ineq",
                "fun": lambda x: float(constraints.turnover_cap - x[n:].sum()),
                "jac": lambda x, row=cap_row: row,
            }
        )

    bounds = [(0.0, max_weight)] * n + [(0.0, None)] * n
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={
                "maxiter": config.solver.max_iter,
                "ftol": config.solver.ftol,
                "disp": False,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive path
        return _fallback(w_prev, config, f"fallback_exception:{type(exc).__name__}")

    if not result.success or result.x is None:
        return _fallback(w_prev, config, f"fallback_solver:{result.message}")

    raw = np.clip(result.x[:n], 0.0, max_weight)
    target_sum = 1.0 if constraints.fully_invested else min(1.0, float(raw.sum()))
    w = cap_and_normalise(raw, max_weight=max_weight, target_sum=target_sum)
    if constraints.turnover_cap is not None and w_prev.sum() > 0:
        w = enforce_turnover_cap(w, w_prev, constraints.turnover_cap)
    traded = float(np.abs(w - w_prev).sum())
    if traded > 1e-8 and np.any(w < -1e-8):
        return _fallback(w_prev, config, "fallback_negative_weight")
    return DecisionResult(
        weights=w,
        status="optimal",
        objective=float(-result.fun),
        traded_notional=traded,
        message=str(result.message),
    )


def _fallback(w_prev: np.ndarray, config: DecisionConfig, status: str) -> DecisionResult:
    weights = _feasible_start(w_prev, config)
    return DecisionResult(
        weights=weights,
        status=status,
        objective=float("nan"),
        traded_notional=float(np.abs(weights - w_prev).sum()),
        message="fallback",
    )
