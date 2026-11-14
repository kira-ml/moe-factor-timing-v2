from __future__ import annotations

import pandas as pd

from moe_factor_timing.data.schema import MarketData
from moe_factor_timing.features.build import build_features


def test_feature_shapes_and_target_definition(features, default_config):
    assert features.X.shape[1] == 30
    assert features.y.shape[1] == 6
    assert features.X.index.equals(features.y.index)
    assert len(features.X) > 700
    assert not features.X.isna().any().any()
    assert not features.y.isna().any().any()
    expected_targets = features.decision_dates + pd.offsets.MonthEnd(1)
    assert (features.target_dates == expected_targets).all()
    assert features.manifest["max_abs_feature_target_correlation"] < 0.8


def test_future_perturbation_does_not_change_features(market, default_config, features):
    cutoff = features.decision_dates[400]
    perturbed_factors = market.factors.copy()
    mask = perturbed_factors.index > cutoff
    perturbed_factors.loc[mask] = perturbed_factors.loc[mask] + 0.75
    perturbed_market = MarketData(
        factors=perturbed_factors,
        risk_free=market.risk_free,
        metadata={"source": "perturbed"},
    )
    perturbed = build_features(perturbed_market, default_config)
    common = features.X.index.intersection(perturbed.X.index)
    common = common[common <= cutoff]
    assert len(common) > 100
    pd.testing.assert_frame_equal(
        features.X.loc[common],
        perturbed.X.loc[common],
        check_exact=True,
    )


def test_truncation_does_not_change_features(market, default_config, features):
    cutoff = features.decision_dates[400]
    truncated_market = MarketData(
        factors=market.factors.loc[:cutoff],
        risk_free=market.risk_free.loc[:cutoff],
        metadata={"source": "truncated"},
    )
    truncated = build_features(truncated_market, default_config)
    common = features.X.index.intersection(truncated.X.index)
    pd.testing.assert_frame_equal(
        features.X.loc[common],
        truncated.X.loc[common],
        check_exact=True,
    )


def test_no_target_columns_leak_into_features(features):
    assert not any(name.startswith("y_") for name in features.feature_names)
    assert set(features.target_names) == {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"}


def test_feature_manifest_records_construction(features):
    assert features.manifest["n_features"] == 30
    assert "y_t" in features.manifest["target_definition"]
    assert features.manifest["scaling"].startswith("models standardise")
