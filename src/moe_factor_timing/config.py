"""Typed, validated configuration.

The configuration is the contract between the research protocol and the code.
It is validated at load time, hashed into every run manifest, and copied into
the run directory so a result can never be separated from its settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils.io import sha256_json


class ConfigError(ValueError):
    """Raised when a configuration is missing, malformed, or inconsistent."""


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key '{section}.{key}'")
    return mapping[key]


def _as_float(
    mapping: dict[str, Any], key: str, section: str, default: float | None = None
) -> float:
    value = mapping.get(key, default)
    if value is None:
        raise ConfigError(f"missing required key '{section}.{key}'")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{section}.{key}' must be numeric") from exc


def _as_int(mapping: dict[str, Any], key: str, section: str, default: int | None = None) -> int:
    value = mapping.get(key, default)
    if value is None:
        raise ConfigError(f"missing required key '{section}.{key}'")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{section}.{key}' must be an integer") from exc


@dataclass(frozen=True)
class RunConfig:
    id: str | None
    seed: int
    description: str = ""


@dataclass(frozen=True)
class DataConfig:
    source: str
    reference_dir: str
    factors: tuple[str, ...]
    risk_free: str
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class FeatureConfig:
    return_lags: tuple[int, ...]
    vol_window: int
    dropna: bool = True


@dataclass(frozen=True)
class SplitConfig:
    min_train: int
    test_step: int = 1
    holdout_start: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    kind: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintConfig:
    long_only: bool = True
    fully_invested: bool = True
    max_weight: float = 0.35
    turnover_cap: float | None = None


@dataclass(frozen=True)
class CostConfig:
    one_way_bps: float = 10.0
    strategy_drag_bps_per_month: float = 0.0


@dataclass(frozen=True)
class SolverConfig:
    name: str = "slsqp"
    max_iter: int = 500
    ftol: float = 1e-10
    fallback_max_rate: float = 0.05


@dataclass(frozen=True)
class DecisionConfig:
    utility: str
    risk_aversion: float
    constraints: ConstraintConfig
    cost: CostConfig
    solver: SolverConfig


@dataclass(frozen=True)
class BootstrapConfig:
    n_repetitions: int = 1000
    block_length: int = 6
    seed: int = 42


@dataclass(frozen=True)
class DeflatedSharpeConfig:
    enabled: bool = True
    n_trials: int = 1


@dataclass(frozen=True)
class PowerConfig:
    alpha: float = 0.05
    power: float = 0.80


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str = "sharpe"
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    deflated_sharpe: DeflatedSharpeConfig = field(default_factory=DeflatedSharpeConfig)
    power: PowerConfig = field(default_factory=PowerConfig)


@dataclass(frozen=True)
class OutputConfig:
    runs_dir: str = "runs"
    reports_dir: str = "reports"
    save_plots: bool = True


@dataclass(frozen=True)
class Config:
    run: RunConfig
    data: DataConfig
    features: FeatureConfig
    split: SplitConfig
    models: tuple[ModelConfig, ...]
    decision: DecisionConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    source_path: Path
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run": asdict(self.run),
            "data": asdict(self.data),
            "features": asdict(self.features),
            "split": asdict(self.split),
            "models": [asdict(model) for model in self.models],
            "decision": asdict(self.decision),
            "evaluation": asdict(self.evaluation),
            "output": asdict(self.output),
        }
        return payload

    def config_hash(self) -> str:
        return sha256_json(self.to_dict())


def _load_models(raw: dict[str, Any]) -> tuple[ModelConfig, ...]:
    models_raw = _require(raw, "models", "config")
    if not isinstance(models_raw, list) or not models_raw:
        raise ConfigError("'models' must be a non-empty list")
    models: list[ModelConfig] = []
    seen: set[str] = set()
    for item in models_raw:
        if not isinstance(item, dict):
            raise ConfigError("each model entry must be a mapping")
        name = str(_require(item, "name", "models"))
        kind = str(_require(item, "kind", f"models.{name}"))
        method = str(_require(item, "method", f"models.{name}"))
        params = item.get("params", {}) or {}
        if name in seen:
            raise ConfigError(f"duplicate model name: {name}")
        if kind not in {"allocator", "forecast"}:
            raise ConfigError(f"model '{name}' has invalid kind '{kind}'")
        if not isinstance(params, dict):
            raise ConfigError(f"model '{name}' params must be a mapping")
        seen.add(name)
        models.append(ModelConfig(name=name, kind=kind, method=method, params=params))
    return tuple(models)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file."""
    source = Path(path).resolve()
    if not source.exists():
        raise ConfigError(f"config not found: {source}")
    with source.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    run_raw = _require(raw, "run", "config")
    data_raw = _require(raw, "data", "config")
    features_raw = _require(raw, "features", "config")
    split_raw = _require(raw, "split", "config")
    decision_raw = _require(raw, "decision", "config")

    run = RunConfig(
        id=run_raw.get("id"),
        seed=_as_int(run_raw, "seed", "run", 42),
        description=str(run_raw.get("description", "")),
    )
    data = DataConfig(
        source=str(_require(data_raw, "source", "data")),
        reference_dir=str(_require(data_raw, "reference_dir", "data")),
        factors=tuple(str(x) for x in _require(data_raw, "factors", "data")),
        risk_free=str(_require(data_raw, "risk_free", "data")),
        start=data_raw.get("start"),
        end=data_raw.get("end"),
    )
    if len(data.factors) < 2:
        raise ConfigError("data.factors must contain at least two factors")

    features = FeatureConfig(
        return_lags=tuple(int(x) for x in _require(features_raw, "return_lags", "features")),
        vol_window=_as_int(features_raw, "vol_window", "features", 12),
        dropna=bool(features_raw.get("dropna", True)),
    )
    if not features.return_lags:
        raise ConfigError("features.return_lags must not be empty")
    if any(lag < 0 for lag in features.return_lags):
        raise ConfigError("features.return_lags must be non-negative")
    if features.vol_window < 2:
        raise ConfigError("features.vol_window must be >= 2")

    split = SplitConfig(
        min_train=_as_int(split_raw, "min_train", "split"),
        test_step=_as_int(split_raw, "test_step", "split", 1),
        holdout_start=split_raw.get("holdout_start"),
    )
    if split.min_train < 2:
        raise ConfigError("split.min_train must be >= 2")
    if split.test_step < 1:
        raise ConfigError("split.test_step must be >= 1")

    constraints_raw = decision_raw.get("constraints", {}) or {}
    constraints = ConstraintConfig(
        long_only=bool(constraints_raw.get("long_only", True)),
        fully_invested=bool(constraints_raw.get("fully_invested", True)),
        max_weight=_as_float(constraints_raw, "max_weight", "decision.constraints", 0.35),
        turnover_cap=(
            None
            if constraints_raw.get("turnover_cap") is None
            else float(constraints_raw["turnover_cap"])
        ),
    )
    if not 0.0 < constraints.max_weight <= 1.0:
        raise ConfigError("decision.constraints.max_weight must be in (0, 1]")
    if constraints.long_only is False:
        raise ConfigError("this benchmark currently supports long-only allocations only")
    if constraints.turnover_cap is not None and constraints.turnover_cap <= 0:
        raise ConfigError("decision.constraints.turnover_cap must be positive or null")

    cost_raw = decision_raw.get("cost", {}) or {}
    cost = CostConfig(
        one_way_bps=_as_float(cost_raw, "one_way_bps", "decision.cost", 10.0),
        strategy_drag_bps_per_month=_as_float(
            cost_raw, "strategy_drag_bps_per_month", "decision.cost", 0.0
        ),
    )
    if cost.one_way_bps < 0 or cost.strategy_drag_bps_per_month < 0:
        raise ConfigError("cost values must be non-negative")

    solver_raw = decision_raw.get("solver", {}) or {}
    solver = SolverConfig(
        name=str(solver_raw.get("name", "slsqp")),
        max_iter=_as_int(solver_raw, "max_iter", "decision.solver", 500),
        ftol=_as_float(solver_raw, "ftol", "decision.solver", 1e-10),
        fallback_max_rate=_as_float(solver_raw, "fallback_max_rate", "decision.solver", 0.05),
    )
    if solver.name != "slsqp":
        raise ConfigError("only the deterministic 'slsqp' solver is supported")

    decision = DecisionConfig(
        utility=str(decision_raw.get("utility", "mean_variance")),
        risk_aversion=_as_float(decision_raw, "risk_aversion", "decision", 2.0),
        constraints=constraints,
        cost=cost,
        solver=solver,
    )
    if decision.utility != "mean_variance":
        raise ConfigError("only 'mean_variance' utility is supported")
    if decision.risk_aversion <= 0:
        raise ConfigError("decision.risk_aversion must be positive")

    eval_raw = raw.get("evaluation", {}) or {}
    bootstrap_raw = eval_raw.get("bootstrap", {}) or {}
    deflated_raw = eval_raw.get("deflated_sharpe", {}) or {}
    power_raw = eval_raw.get("power", {}) or {}
    evaluation = EvaluationConfig(
        primary_metric=str(eval_raw.get("primary_metric", "sharpe")),
        bootstrap=BootstrapConfig(
            n_repetitions=_as_int(bootstrap_raw, "n_repetitions", "evaluation.bootstrap", 1000),
            block_length=_as_int(bootstrap_raw, "block_length", "evaluation.bootstrap", 6),
            seed=_as_int(bootstrap_raw, "seed", "evaluation.bootstrap", 42),
        ),
        deflated_sharpe=DeflatedSharpeConfig(
            enabled=bool(deflated_raw.get("enabled", True)),
            n_trials=_as_int(deflated_raw, "n_trials", "evaluation.deflated_sharpe", 1),
        ),
        power=PowerConfig(
            alpha=_as_float(power_raw, "alpha", "evaluation.power", 0.05),
            power=_as_float(power_raw, "power", "evaluation.power", 0.80),
        ),
    )
    if evaluation.primary_metric not in {"sharpe", "utility"}:
        raise ConfigError("evaluation.primary_metric must be 'sharpe' or 'utility'")
    if evaluation.bootstrap.n_repetitions < 100:
        raise ConfigError("evaluation.bootstrap.n_repetitions must be >= 100")
    if evaluation.bootstrap.block_length < 1:
        raise ConfigError("evaluation.bootstrap.block_length must be >= 1")

    output_raw = raw.get("output", {}) or {}
    output = OutputConfig(
        runs_dir=str(output_raw.get("runs_dir", "runs")),
        reports_dir=str(output_raw.get("reports_dir", "reports")),
        save_plots=bool(output_raw.get("save_plots", True)),
    )

    return Config(
        run=run,
        data=data,
        features=features,
        split=split,
        models=_load_models(raw),
        decision=decision,
        evaluation=evaluation,
        output=output,
        source_path=source,
        raw=raw,
    )
