"""Shared fixtures.

The reference panel is loaded once per session. Synthetic panels are small and
deterministic so unit tests do not depend on network access or wall-clock time.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from moe_factor_timing.config import load_config
from moe_factor_timing.data.loader import load_market_data
from moe_factor_timing.data.schema import MarketData
from moe_factor_timing.features.build import build_features


@pytest.fixture(scope="session")
def default_config():
    return load_config("configs/default.yaml")


@pytest.fixture(scope="session")
def smoke_config():
    return load_config("configs/smoke.yaml")


@pytest.fixture(scope="session")
def market(default_config):
    try:
        return load_market_data(default_config)
    except FileNotFoundError as exc:
        pytest.skip(f"reference panel not present locally: {exc}")


@pytest.fixture(scope="session")
def features(market, default_config):
    return build_features(market, default_config)


def synthetic_market(n_months: int = 240, n_assets: int = 4, seed: int = 0) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    assets = [f"F{i}" for i in range(n_assets)]
    base = rng.normal(0.002, 0.025, size=(n_months, n_assets))
    # Add a persistent regime shift so a mixture model has something to find.
    base[::48] += 0.015
    factors = pd.DataFrame(base, index=dates, columns=assets)
    risk_free = pd.Series(np.full(n_months, 0.002), index=dates, name="RF")
    return MarketData(
        factors=factors,
        risk_free=risk_free,
        metadata={"source": "synthetic", "loaded_rows": n_months},
    )


def small_config(config, tmp_path, models=None):
    """A fast config that keeps the real reference data but shrinks the run."""
    from moe_factor_timing.config import (
        BootstrapConfig,
        ModelConfig,
        OutputConfig,
        SplitConfig,
    )

    if models is None:
        models = (
            ModelConfig(name="one_over_n", kind="allocator", method="equal"),
            ModelConfig(
                name="ridge",
                kind="forecast",
                method="ridge",
                params={"alpha": 10.0, "shrinkage": 0.2},
            ),
        )
    evaluation = replace(
        config.evaluation,
        bootstrap=BootstrapConfig(n_repetitions=100, block_length=6, seed=42),
    )
    return replace(
        config,
        data=replace(config.data, start="2000-01-31"),
        split=SplitConfig(min_train=60, test_step=1, holdout_start="2015-01-31"),
        models=tuple(models),
        evaluation=evaluation,
        output=OutputConfig(
            runs_dir=str(tmp_path / "runs"),
            reports_dir=str(tmp_path / "reports"),
            save_plots=False,
        ),
    )
