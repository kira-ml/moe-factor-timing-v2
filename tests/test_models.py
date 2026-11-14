from __future__ import annotations

import numpy as np
import pytest

from moe_factor_timing.models.moe import MixtureOfExpertsModel
from moe_factor_timing.models.ridge import RidgeForecastModel
from moe_factor_timing.models.static_mv import StaticMeanVarianceModel


def _data(n=240, p=8, q=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    regime = (X[:, 0] > 0).astype(int)
    B0 = rng.normal(scale=0.1, size=(p, q))
    B1 = rng.normal(scale=0.1, size=(p, q))
    y = np.where(regime[:, None] == 0, X @ B0, X @ B1) + rng.normal(scale=0.05, size=(n, q))
    return X, y


def _assert_valid_forecast(forecast, q):
    assert forecast.mean.shape == (q,)
    assert forecast.cov.shape == (q, q)
    assert np.allclose(forecast.cov, forecast.cov.T)
    assert np.linalg.eigvalsh(forecast.cov).min() > 0
    assert np.isfinite(forecast.nll(np.zeros(q)))
    pit = forecast.pit(np.zeros(q))
    assert pit.shape == (q,)
    assert np.all((pit >= 0) & (pit <= 1))


def test_static_mean_variance_is_constant():
    X, y = _data()
    model = StaticMeanVarianceModel(shrinkage=0.2).fit(X[:180], y[:180])
    first = model.predict(X[180])
    second = model.predict(X[181])
    np.testing.assert_allclose(first.mean, second.mean)
    np.testing.assert_allclose(first.cov, second.cov)
    _assert_valid_forecast(first, y.shape[1])


def test_ridge_shapes_and_determinism():
    X, y = _data()
    a = RidgeForecastModel(alpha=10.0, shrinkage=0.2).fit(X[:180], y[:180])
    b = RidgeForecastModel(alpha=10.0, shrinkage=0.2).fit(X[:180], y[:180])
    fa = a.predict(X[180])
    fb = b.predict(X[180])
    _assert_valid_forecast(fa, y.shape[1])
    np.testing.assert_allclose(fa.mean, fb.mean)
    np.testing.assert_allclose(fa.cov, fb.cov)


def test_moe_shapes_components_and_determinism():
    X, y = _data()
    kwargs = dict(
        n_experts=2,
        n_restarts=1,
        max_iter=10,
        gating_steps=3,
        seed=7,
    )
    a = MixtureOfExpertsModel(**kwargs).fit(X[:180], y[:180])
    b = MixtureOfExpertsModel(**kwargs).fit(X[:180], y[:180])
    fa = a.predict(X[180])
    fb = b.predict(X[180])
    _assert_valid_forecast(fa, y.shape[1])
    assert fa.components is not None
    assert len(fa.components) == 2
    assert np.isclose(sum(component.weight for component in fa.components), 1.0)
    np.testing.assert_allclose(fa.mean, fb.mean)
    np.testing.assert_allclose(fa.cov, fb.cov)
    assert np.isclose(a.predict_proba(X[180]).sum(), 1.0)


def test_moe_rejects_too_few_observations():
    X, y = _data(n=12)
    with pytest.raises(ValueError, match="not enough"):
        MixtureOfExpertsModel(n_experts=10, n_restarts=1).fit(X, y)


def test_unfitted_models_raise():
    X, _ = _data()
    with pytest.raises(RuntimeError):
        StaticMeanVarianceModel().predict(X[0])
    with pytest.raises(RuntimeError):
        RidgeForecastModel().predict(X[0])
    with pytest.raises(RuntimeError):
        MixtureOfExpertsModel().predict(X[0])
