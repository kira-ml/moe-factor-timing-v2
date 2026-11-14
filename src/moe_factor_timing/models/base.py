"""Model contracts.

Every probabilistic model returns a :class:`DistributionForecast`, not just a
point estimate. The forecast object owns the log-density and marginal PIT
calculations so calibration is evaluated against the same distribution the
decision layer sees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm


def ensure_covariance(cov: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Symmetrise and clip eigenvalues so the matrix is safely positive definite."""
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    scale = max(float(np.median(np.diag(cov))), floor)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, floor * scale, None)
    repaired = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return 0.5 * (repaired + repaired.T)


def shrink_covariance(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """Shrink off-diagonal covariance toward a diagonal target."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    diagonal = np.diag(np.diag(cov))
    return ensure_covariance((1.0 - shrinkage) * cov + shrinkage * diagonal)


def gaussian_logpdf(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Multivariate Gaussian log-density via Cholesky, with a PSD repair."""
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = ensure_covariance(cov)
    diff = y - mean
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + np.eye(len(y)) * 1e-8
        chol = np.linalg.cholesky(cov)
    solved = np.linalg.solve(chol, diff)
    log_det = 2.0 * np.log(np.diag(chol)).sum()
    dim = len(y)
    return float(-0.5 * (dim * np.log(2.0 * np.pi) + log_det + solved @ solved))


@dataclass(frozen=True)
class Component:
    weight: float
    mean: np.ndarray
    cov: np.ndarray


@dataclass(frozen=True)
class DistributionForecast:
    """A predictive distribution over the next-month factor return vector."""

    mean: np.ndarray
    cov: np.ndarray
    model_name: str
    components: tuple[Component, ...] | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        cov = ensure_covariance(self.cov, floor=1e-12)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "cov", cov)
        if cov.shape != (len(mean), len(mean)):
            raise ValueError(f"covariance shape {cov.shape} does not match mean length {len(mean)}")
        if self.components is not None:
            total = sum(component.weight for component in self.components)
            if not np.isclose(total, 1.0, atol=1e-8):
                raise ValueError(f"mixture weights must sum to 1, got {total}")
            for component in self.components:
                if component.mean.shape != mean.shape:
                    raise ValueError("component mean has the wrong shape")
                if component.cov.shape != cov.shape:
                    raise ValueError("component covariance has the wrong shape")

    @property
    def n_assets(self) -> int:
        return int(len(self.mean))

    def nll(self, y: np.ndarray) -> float:
        """Negative log predictive density of a realised return vector."""
        y = np.asarray(y, dtype=float).reshape(-1)
        if self.components is None:
            return -gaussian_logpdf(y, self.mean, self.cov)
        log_terms = []
        for component in self.components:
            if component.weight <= 0.0:
                continue
            log_terms.append(
                np.log(component.weight) + gaussian_logpdf(y, component.mean, component.cov)
            )
        return float(-logsumexp(log_terms))

    def pit(self, y: np.ndarray) -> np.ndarray:
        """Marginal probability integral transform, one value per asset.

        For a Gaussian this is the usual Φ((y-μ)/σ). For a mixture it is the
        exact marginal mixture CDF, not a Gaussian approximation.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        if self.components is None:
            std = np.sqrt(np.diag(self.cov))
            return norm.cdf((y - self.mean) / std)
        result = np.zeros_like(y)
        for component in self.components:
            if component.weight <= 0.0:
                continue
            std = np.sqrt(np.diag(component.cov))
            result += component.weight * norm.cdf((y - component.mean) / std)
        return result


class ForecastModel(ABC):
    """A one-step-ahead probabilistic forecaster with no future access."""

    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> ForecastModel:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> DistributionForecast:
        raise NotImplementedError

    @staticmethod
    def _validate_fit_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 2:
            raise ValueError(f"y must be 2-D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y row mismatch: {X.shape[0]} vs {y.shape[0]}")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values")
        return X, y


class Allocator(ABC):
    """A direct allocation rule that maps observable history to weights."""

    name: str

    @abstractmethod
    def weights(self, history: pd.DataFrame, decision_date: pd.Timestamp) -> np.ndarray:
        raise NotImplementedError
