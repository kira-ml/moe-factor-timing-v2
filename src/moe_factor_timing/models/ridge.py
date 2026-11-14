"""Multivariate ridge forecast with a residual covariance.

Ridge is the regularised linear baseline. It is intentionally strong: with 30
features and 120+ training months, it can fit a stable conditional mean, and
its residual covariance gives the decision layer a complete predictive
distribution.
"""

from __future__ import annotations

import numpy as np

from .base import DistributionForecast, ForecastModel, shrink_covariance


class RidgeForecastModel(ForecastModel):
    name = "ridge"

    def __init__(self, alpha: float = 10.0, shrinkage: float = 0.2):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.shrinkage = float(shrinkage)
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: np.ndarray | None = None
        self.y_std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.cov_scaled_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeForecastModel:
        X, y = self._validate_fit_inputs(X, y)
        n, p = X.shape
        q = y.shape[1]
        if n < 3:
            raise ValueError("ridge needs at least three observations")

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = np.maximum(X.std(axis=0, ddof=0), 1e-12)
        self.y_mean_ = y.mean(axis=0)
        self.y_std_ = np.maximum(y.std(axis=0, ddof=0), 1e-12)

        Xs = (X - self.x_mean_) / self.x_std_
        Ys = (y - self.y_mean_) / self.y_std_
        design = np.column_stack([np.ones(n), Xs])

        penalty = self.alpha * np.eye(p + 1)
        penalty[0, 0] = 0.0
        gram = design.T @ design + penalty
        rhs = design.T @ Ys
        try:
            self.coef_ = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(gram, rhs, rcond=None)[0]

        residuals = Ys - design @ self.coef_
        dof = max(n - p - 1, 1)
        covariance = residuals.T @ residuals / dof
        self.cov_scaled_ = shrink_covariance(np.atleast_2d(covariance), self.shrinkage)
        if self.cov_scaled_.shape != (q, q):
            raise AssertionError("residual covariance has the wrong shape")
        return self

    def predict(self, x: np.ndarray) -> DistributionForecast:
        if self.coef_ is None or self.cov_scaled_ is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.x_mean_.shape[0]:  # type: ignore[union-attr]
            raise ValueError(f"expected {self.x_mean_.shape[0]} features, got {x.shape[0]}")  # type: ignore[union-attr]
        xs = (x - self.x_mean_) / self.x_std_
        design = np.concatenate([[1.0], xs])
        mean_scaled = design @ self.coef_
        mean = mean_scaled * self.y_std_ + self.y_mean_
        cov = self.cov_scaled_ * np.outer(self.y_std_, self.y_std_)
        return DistributionForecast(mean=mean, cov=cov, model_name=self.name)
