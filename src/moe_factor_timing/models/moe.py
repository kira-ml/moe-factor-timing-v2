"""Probabilistic mixture of linear experts.

Model
-----
For ``K`` experts and a ``p``-dimensional feature vector ``x``:

    pi_k(x) = softmax_k(W^T [1, x])
    p(y | x, k) = N(y | B_k^T [1, x], Sigma_k)
    p(y | x) = sum_k pi_k(x) p(y | x, k)

The model is trained by expectation-maximisation with weighted ridge experts,
a multinomial-logit gating network, and shrunk covariance estimates. Multiple
restarts are selected by the training log-likelihood only; the evaluation
window is never used for model selection.

This is deliberately a small, interpretable MoE. It is not a neural network,
and it does not claim to be one. The point is to test whether soft regime
membership adds decision value over a linear conditional mean.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_softmax, logsumexp, softmax

from .base import (
    Component,
    DistributionForecast,
    ForecastModel,
    ensure_covariance,
    shrink_covariance,
)


def _cholesky_info(cov: np.ndarray) -> tuple[np.ndarray, float]:
    cov = ensure_covariance(cov)
    chol = np.linalg.cholesky(cov)
    log_det = float(2.0 * np.log(np.diag(chol)).sum())
    return chol, log_det


def _gaussian_logpdf_rows(
    y: np.ndarray, mean: np.ndarray, chol: np.ndarray, log_det: float
) -> np.ndarray:
    """Vectorised multivariate Gaussian log-density for every row of ``y``."""
    diff = y - mean
    solved = np.linalg.solve(chol, diff.T)
    quadratic = np.sum(solved**2, axis=0)
    dim = y.shape[1]
    return -0.5 * (dim * np.log(2.0 * np.pi) + log_det + quadratic)


def _kmeans_labels(X: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 12) -> np.ndarray:
    """Small deterministic k-means used only for EM initialisation.

    sklearn's KMeans is avoided in the walk-forward loop because its joblib
    backend performs CPU discovery on every fit, which dominates runtime on
    Windows. This implementation is deliberately minimal.
    """
    n = X.shape[0]
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        distances = (
            np.sum(X**2, axis=1)[:, None]
            - 2.0 * X @ centers.T
            + np.sum(centers**2, axis=1)[None, :]
        )
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)
    return labels


class MixtureOfExpertsModel(ForecastModel):
    name = "moe"

    def __init__(
        self,
        n_experts: int = 3,
        n_restarts: int = 2,
        max_iter: int = 50,
        tol: float = 1e-6,
        ridge_alpha: float = 1.0,
        gating_l2: float = 0.01,
        covariance_shrinkage: float = 0.2,
        gating_steps: int = 12,
        gating_lr: float = 0.5,
        min_responsibility: float = 1e-4,
        seed: int = 42,
    ):
        if n_experts < 2:
            raise ValueError("n_experts must be >= 2")
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if ridge_alpha < 0 or gating_l2 < 0:
            raise ValueError("regularisation strengths must be non-negative")
        if not 0.0 <= covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        self.n_experts = int(n_experts)
        self.n_restarts = int(n_restarts)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ridge_alpha = float(ridge_alpha)
        self.gating_l2 = float(gating_l2)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.gating_steps = int(gating_steps)
        self.gating_lr = float(gating_lr)
        self.min_responsibility = float(min_responsibility)
        self.seed = int(seed)

        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: np.ndarray | None = None
        self.y_std_: np.ndarray | None = None
        self.gating_weights_: np.ndarray | None = None
        self.expert_coef_: list[np.ndarray] | None = None
        self.expert_cov_: list[np.ndarray] | None = None
        self.log_likelihood_: float | None = None
        self.n_iter_: int | None = None
        self.restart_scores_: list[float] = []
        self.n_features_in_: int | None = None
        self.n_targets_in_: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> MixtureOfExpertsModel:
        X, y = self._validate_fit_inputs(X, y)
        n, p = X.shape
        q = y.shape[1]
        if n < max(10, self.n_experts * 3):
            raise ValueError(f"not enough observations for {self.n_experts} experts: n={n}")

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = np.maximum(X.std(axis=0, ddof=0), 1e-12)
        self.y_mean_ = y.mean(axis=0)
        self.y_std_ = np.maximum(y.std(axis=0, ddof=0), 1e-12)
        self.n_features_in_ = p
        self.n_targets_in_ = q

        Xs = (X - self.x_mean_) / self.x_std_
        Ys = (y - self.y_mean_) / self.y_std_
        design = np.column_stack([np.ones(n), Xs])

        best: dict | None = None
        self.restart_scores_ = []
        for restart in range(self.n_restarts):
            params = self._initialise(design, Xs, Ys, restart)
            previous_ll = -np.inf
            current_ll = -np.inf
            iterations = 0
            for iteration in range(1, self.max_iter + 1):
                iterations = iteration
                responsibilities, current_ll = self._e_step(design, Ys, params)
                params = self._m_step(design, Ys, responsibilities, params)
                if np.isfinite(previous_ll) and abs(current_ll - previous_ll) <= self.tol * max(
                    1.0, abs(previous_ll)
                ):
                    break
                previous_ll = current_ll
            self.restart_scores_.append(float(current_ll))
            _, final_ll = self._e_step(design, Ys, params)
            if best is None or final_ll > best["log_likelihood"]:
                best = {
                    "params": params,
                    "log_likelihood": float(final_ll),
                    "iterations": iterations,
                }

        if best is None:
            raise RuntimeError("mixture-of-experts failed to fit")
        self.gating_weights_ = best["params"]["gating"]
        self.expert_coef_ = best["params"]["coef"]
        self.expert_cov_ = best["params"]["cov"]
        self.log_likelihood_ = best["log_likelihood"]
        self.n_iter_ = int(best["iterations"])
        self._canonicalise_components(design)
        return self

    def predict(self, x: np.ndarray) -> DistributionForecast:
        if self.gating_weights_ is None or self.expert_coef_ is None or self.expert_cov_ is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x.shape[0]}")
        xs = (x - self.x_mean_) / self.x_std_
        design = np.concatenate([[1.0], xs])

        weights = softmax(design @ self.gating_weights_)
        means_scaled = np.vstack([design @ coef for coef in self.expert_coef_])
        means = means_scaled * self.y_std_ + self.y_mean_
        covariances = [cov * np.outer(self.y_std_, self.y_std_) for cov in self.expert_cov_]

        mixture_mean = np.sum(weights[:, None] * means, axis=0)
        mixture_cov = np.zeros((self.n_targets_in_, self.n_targets_in_))
        for weight, mean, cov in zip(weights, means, covariances, strict=True):
            shift = mean - mixture_mean
            mixture_cov += weight * (cov + np.outer(shift, shift))
        mixture_cov = ensure_covariance(mixture_cov)

        components = tuple(
            Component(weight=float(weight), mean=mean.copy(), cov=ensure_covariance(cov))
            for weight, mean, cov in zip(weights, means, covariances, strict=True)
        )
        return DistributionForecast(
            mean=mixture_mean,
            cov=mixture_cov,
            model_name=self.name,
            components=components,
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.gating_weights_ is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=float).reshape(-1)
        xs = (x - self.x_mean_) / self.x_std_
        design = np.concatenate([[1.0], xs])
        return softmax(design @ self.gating_weights_)

    # ------------------------------------------------------------------
    # EM internals
    # ------------------------------------------------------------------
    def _initialise(self, design: np.ndarray, Xs: np.ndarray, Ys: np.ndarray, restart: int) -> dict:
        n = design.shape[0]
        rng = np.random.default_rng(self.seed + restart)
        labels = _kmeans_labels(Xs, self.n_experts, rng)
        responsibilities = np.zeros((n, self.n_experts))
        responsibilities[np.arange(n), labels] = 1.0
        responsibilities = responsibilities + 0.05
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

        gating = self._fit_gating(
            design, responsibilities, np.zeros((design.shape[1], self.n_experts))
        )
        coef, cov, chol, log_det = self._fit_experts(design, responsibilities, Ys)
        return {
            "gating": gating,
            "coef": coef,
            "cov": cov,
            "chol": chol,
            "log_det": log_det,
        }

    def _e_step(self, design: np.ndarray, Ys: np.ndarray, params: dict) -> tuple[np.ndarray, float]:
        log_pi = log_softmax(design @ params["gating"], axis=1)
        log_joint = np.empty((Ys.shape[0], self.n_experts))
        for k in range(self.n_experts):
            mean = design @ params["coef"][k]
            log_joint[:, k] = log_pi[:, k] + _gaussian_logpdf_rows(
                Ys, mean, params["chol"][k], params["log_det"][k]
            )
        log_evidence = logsumexp(log_joint, axis=1)
        responsibilities = np.exp(log_joint - log_evidence[:, None])
        responsibilities = np.clip(responsibilities, self.min_responsibility, None)
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
        return responsibilities, float(log_evidence.sum())

    def _m_step(
        self, design: np.ndarray, Ys: np.ndarray, responsibilities: np.ndarray, previous: dict
    ) -> dict:
        coef, cov, chol, log_det = self._fit_experts(design, responsibilities, Ys)
        gating = self._fit_gating(design, responsibilities, previous["gating"])
        return {
            "gating": gating,
            "coef": coef,
            "cov": cov,
            "chol": chol,
            "log_det": log_det,
        }

    def _fit_experts(
        self, design: np.ndarray, responsibilities: np.ndarray, Ys: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
        n, p1 = design.shape
        penalty = self.ridge_alpha * np.eye(p1)
        penalty[0, 0] = 0.0
        coefs: list[np.ndarray] = []
        covs: list[np.ndarray] = []
        chols: list[np.ndarray] = []
        log_dets: list[float] = []
        for k in range(self.n_experts):
            weights = responsibilities[:, k]
            total = weights.sum()
            if total < 1e-10:
                weights = np.ones(n) / n
                total = 1.0
            weighted_design = design * weights[:, None]
            gram = design.T @ weighted_design + penalty
            rhs = weighted_design.T @ Ys
            try:
                coef = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            residuals = Ys - design @ coef
            covariance = (residuals.T @ (residuals * weights[:, None])) / max(total, 1.0)
            covariance = shrink_covariance(np.atleast_2d(covariance), self.covariance_shrinkage)
            chol, log_det = _cholesky_info(covariance)
            coefs.append(coef)
            covs.append(covariance)
            chols.append(chol)
            log_dets.append(log_det)
        return coefs, covs, chols, log_dets

    def _fit_gating(
        self, design: np.ndarray, responsibilities: np.ndarray, initial: np.ndarray
    ) -> np.ndarray:
        """L2-regularised multinomial logit fitted by gradient descent.

        The M-step objective is convex in the gating weights, so a backtracking
        gradient step is sufficient and fully deterministic.
        """
        n = design.shape[0]
        weights = initial.copy()
        penalty = np.ones_like(weights)
        penalty[0, :] = 0.0

        def objective(w: np.ndarray) -> float:
            log_probs = log_softmax(design @ w, axis=1)
            return float(
                -np.sum(responsibilities * log_probs) / n
                + 0.5 * self.gating_l2 * np.sum((w * penalty) ** 2)
            )

        current = objective(weights)
        step = self.gating_lr
        for _ in range(self.gating_steps):
            probs = softmax(design @ weights, axis=1)
            gradient = (
                design.T @ (probs - responsibilities) / n + self.gating_l2 * weights * penalty
            )
            gradient_norm = float(np.sum(gradient * gradient))
            if gradient_norm < 1e-12:
                break
            accepted = False
            for _ in range(20):
                candidate = weights - step * gradient
                candidate_value = objective(candidate)
                if candidate_value <= current - 1e-4 * step * gradient_norm:
                    weights = candidate
                    current = candidate_value
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            step = min(step * 1.1, 1.0)
        return weights

    def _canonicalise_components(self, design: np.ndarray) -> None:
        """Order experts by their average predicted first-factor return.

        This does not change the likelihood; it makes diagnostics easier to
        read across walk-forward refits. Component labels remain descriptive,
        not causal regimes.
        """
        if self.expert_coef_ is None or self.expert_cov_ is None or self.gating_weights_ is None:
            return
        average_predictions = np.array(
            [float((design @ coef)[:, 0].mean()) for coef in self.expert_coef_]
        )
        order = np.argsort(average_predictions)
        self.expert_coef_ = [self.expert_coef_[i] for i in order]
        self.expert_cov_ = [self.expert_cov_[i] for i in order]
        self.gating_weights_ = self.gating_weights_[:, order]
