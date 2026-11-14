"""Model registry with strict parameter validation.

Unknown method names or misspelled parameters raise immediately. This prevents
a configuration typo from silently producing a different model than the one
described in the research protocol.
"""

from __future__ import annotations

from typing import Any

from ..config import ModelConfig
from .base import Allocator, ForecastModel
from .benchmarks import (
    CashAllocator,
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    MomentumAllocator,
)
from .moe import MixtureOfExpertsModel
from .ridge import RidgeForecastModel
from .static_mv import StaticMeanVarianceModel


def _check_params(name: str, params: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"model '{name}' has unknown parameters: {sorted(unknown)}")


def build_allocator(spec: ModelConfig, n_assets: int) -> Allocator:
    params = dict(spec.params)
    if spec.method == "equal":
        _check_params(spec.name, params, set())
        return EqualWeightAllocator(n_assets)
    if spec.method == "cash":
        _check_params(spec.name, params, set())
        return CashAllocator(n_assets)
    if spec.method == "inverse_vol":
        _check_params(spec.name, params, {"window", "max_weight"})
        return InverseVolatilityAllocator(
            n_assets,
            window=int(params.get("window", 12)),
            max_weight=float(params.get("max_weight", 0.35)),
        )
    if spec.method == "momentum":
        _check_params(spec.name, params, {"window", "max_weight"})
        return MomentumAllocator(
            n_assets,
            window=int(params.get("window", 12)),
            max_weight=float(params.get("max_weight", 0.35)),
        )
    raise ValueError(f"unknown allocator method: {spec.method}")


def build_forecast_model(spec: ModelConfig, seed: int) -> ForecastModel:
    params = dict(spec.params)
    if spec.method == "static_mv":
        _check_params(spec.name, params, {"shrinkage"})
        return StaticMeanVarianceModel(shrinkage=float(params.get("shrinkage", 0.2)))
    if spec.method == "ridge":
        _check_params(spec.name, params, {"alpha", "shrinkage"})
        return RidgeForecastModel(
            alpha=float(params.get("alpha", 10.0)),
            shrinkage=float(params.get("shrinkage", 0.2)),
        )
    if spec.method == "moe":
        _check_params(
            spec.name,
            params,
            {
                "n_experts",
                "n_restarts",
                "max_iter",
                "tol",
                "ridge_alpha",
                "gating_l2",
                "covariance_shrinkage",
                "gating_steps",
                "gating_lr",
                "min_responsibility",
            },
        )
        return MixtureOfExpertsModel(
            n_experts=int(params.get("n_experts", 3)),
            n_restarts=int(params.get("n_restarts", 2)),
            max_iter=int(params.get("max_iter", 50)),
            tol=float(params.get("tol", 1e-6)),
            ridge_alpha=float(params.get("ridge_alpha", 1.0)),
            gating_l2=float(params.get("gating_l2", 0.01)),
            covariance_shrinkage=float(params.get("covariance_shrinkage", 0.2)),
            gating_steps=int(params.get("gating_steps", 8)),
            gating_lr=float(params.get("gating_lr", 0.5)),
            min_responsibility=float(params.get("min_responsibility", 1e-4)),
            seed=seed,
        )
    raise ValueError(f"unknown forecast method: {spec.method}")


def build_model(spec: ModelConfig, n_assets: int, seed: int) -> Allocator | ForecastModel:
    if spec.kind == "allocator":
        return build_allocator(spec, n_assets)
    if spec.kind == "forecast":
        return build_forecast_model(spec, seed)
    raise ValueError(f"unknown model kind: {spec.kind}")
