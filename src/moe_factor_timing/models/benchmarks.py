"""Transparent, deterministic allocation benchmarks.

These are deliberately simple. A complex model has to beat them, not merely
beat cash. All allocators see history only through the decision date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.weights import cap_and_normalise
from .base import Allocator


class _HistoryAllocator(Allocator):
    def _history_through(self, history: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
        if not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("history must have a DatetimeIndex")
        if history.index.max() > decision_date:
            raise ValueError("allocator received data after the decision date")
        window = history.loc[:decision_date]
        if window.empty:
            raise ValueError("allocator received an empty history")
        return window


class EqualWeightAllocator(_HistoryAllocator):
    name = "one_over_n"

    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def weights(self, history: pd.DataFrame, decision_date: pd.Timestamp) -> np.ndarray:
        self._history_through(history, decision_date)
        return np.full(self.n_assets, 1.0 / self.n_assets)


class CashAllocator(_HistoryAllocator):
    name = "cash"

    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def weights(self, history: pd.DataFrame, decision_date: pd.Timestamp) -> np.ndarray:
        self._history_through(history, decision_date)
        return np.zeros(self.n_assets)


class InverseVolatilityAllocator(_HistoryAllocator):
    name = "inverse_vol"

    def __init__(self, n_assets: int, window: int = 12, max_weight: float = 0.35):
        self.n_assets = n_assets
        self.window = window
        self.max_weight = max_weight

    def weights(self, history: pd.DataFrame, decision_date: pd.Timestamp) -> np.ndarray:
        window = self._history_through(history, decision_date).iloc[-self.window :]
        if len(window) < 2:
            return np.full(self.n_assets, 1.0 / self.n_assets)
        volatility = window.std(ddof=1).to_numpy()
        if not np.isfinite(volatility).all() or np.any(volatility <= 1e-12):
            return np.full(self.n_assets, 1.0 / self.n_assets)
        raw = 1.0 / volatility
        return cap_and_normalise(raw, max_weight=self.max_weight)


class MomentumAllocator(_HistoryAllocator):
    name = "momentum"

    def __init__(self, n_assets: int, window: int = 12, max_weight: float = 0.35):
        self.n_assets = n_assets
        self.window = window
        self.max_weight = max_weight

    def weights(self, history: pd.DataFrame, decision_date: pd.Timestamp) -> np.ndarray:
        window = self._history_through(history, decision_date).iloc[-self.window :]
        if window.empty:
            return np.zeros(self.n_assets)
        momentum = window.sum(axis=0).to_numpy()
        raw = np.clip(momentum, 0.0, None)
        if raw.sum() <= 0:
            return np.zeros(self.n_assets)
        n_positive = int(np.sum(raw > 0))
        target = min(1.0, n_positive * self.max_weight)
        return cap_and_normalise(raw, max_weight=self.max_weight, target_sum=target)
