"""Weight-space projections shared by allocators and the optimizer."""

from __future__ import annotations

import numpy as np


def cap_and_normalise(
    weights: np.ndarray, max_weight: float, target_sum: float = 1.0
) -> np.ndarray:
    """Project non-negative weights onto ``sum == target_sum`` with a cap.

    The projection water-fills any excess from capped assets into the
    uncapped ones. If the cap makes the target infeasible, the maximum
    feasible sum is returned. The result is deterministic.
    """
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if np.any(weights < -1e-12):
        raise ValueError("weights must be non-negative")
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0:
        return np.zeros_like(weights)
    weights = weights / weights.sum() * target_sum
    if max_weight >= target_sum:
        return weights

    capped = np.zeros_like(weights, dtype=bool)
    for _ in range(len(weights) + 1):
        over = weights > max_weight + 1e-12
        if not over.any():
            break
        capped |= over
        weights[capped] = max_weight
        free = ~capped
        remaining = target_sum - weights[capped].sum()
        if not free.any() or remaining <= 1e-12:
            break
        free_weights = weights[free]
        free_sum = free_weights.sum()
        if free_sum <= 1e-12:
            # All uncapped assets have zero raw weight. Spread the residual
            # equally rather than dividing by zero.
            weights[free] = remaining / free.sum()
            break
        weights[free] = free_weights / free_sum * remaining
    total = weights.sum()
    if total > 0:
        weights = weights / total * min(target_sum, max_weight * len(weights))
    else:
        weights = np.zeros_like(weights)
    return weights


def enforce_turnover_cap(weights: np.ndarray, w_prev: np.ndarray, cap: float | None) -> np.ndarray:
    """Move toward ``w_prev`` until the L1 trade is within ``cap``.

    A convex combination of two feasible long-only portfolios remains feasible
    with respect to the cap and the sum constraint.
    """
    if cap is None:
        return weights
    traded = float(np.abs(weights - w_prev).sum())
    if traded <= cap + 1e-12:
        return weights
    alpha = cap / traded
    return w_prev + alpha * (weights - w_prev)
