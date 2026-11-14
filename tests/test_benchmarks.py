from __future__ import annotations

import numpy as np
import pandas as pd

from moe_factor_timing.models.benchmarks import MomentumAllocator


def test_momentum_with_one_positive_factor_holds_cash():
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    history = pd.DataFrame(
        {
            "A": np.full(12, 0.01),
            "B": np.full(12, -0.01),
            "C": np.full(12, -0.01),
        },
        index=dates,
    )
    allocator = MomentumAllocator(3, window=12, max_weight=0.35)
    weights = allocator.weights(history, dates[-1])
    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 0.35)
    assert weights[0] == 0.35
    assert weights[1] == 0.0 and weights[2] == 0.0
