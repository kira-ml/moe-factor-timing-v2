"""Unconditional mean-variance benchmark.

This is the parsimony baseline for the forecast layer: if a conditional model
cannot beat a plug-in estimate of the historical mean and covariance, the
conditional machinery has not earned its complexity.
"""

from __future__ import annotations

import numpy as np

from .base import DistributionForecast, ForecastModel, shrink_covariance


class StaticMeanVarianceModel(ForecastModel):
    name = "static_mv"

    def __init__(self, shrinkage: float = 0.2):
        self.shrinkage = float(shrinkage)
        self.mean_: np.ndarray | None = None
        self.cov_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> StaticMeanVarianceModel:
        X, y = self._validate_fit_inputs(X, y)
        if len(y) < 2:
            raise ValueError("static mean-variance needs at least two observations")
        self.mean_ = y.mean(axis=0)
        covariance = np.cov(y, rowvar=False, ddof=1)
        self.cov_ = shrink_covariance(np.atleast_2d(covariance), self.shrinkage)
        return self

    def predict(self, x: np.ndarray) -> DistributionForecast:
        if self.mean_ is None or self.cov_ is None:
            raise RuntimeError("model is not fitted")
        return DistributionForecast(mean=self.mean_, cov=self.cov_, model_name=self.name)
