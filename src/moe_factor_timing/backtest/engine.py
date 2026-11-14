"""Walk-forward backtest engine.

The engine is deliberately strict. A model must produce a forecast through the
fitted distribution interface, the decision rule must return a feasible
portfolio, and the realised return must match the feature/target contract.
Unexpected model errors fail the run; solver fallbacks are counted and gated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, ModelConfig
from ..data.schema import MarketData
from ..decision.mean_variance import solve
from ..features.build import FeatureSet
from ..models.base import Allocator, ForecastModel
from ..models.registry import build_model
from ..splits.walk_forward import SplitPlan
from ..utils.io import ensure_dir, write_csv, write_json

logger = logging.getLogger(__name__)


@dataclass
class ModelBacktest:
    name: str
    kind: str
    method: str
    dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex
    weights: pd.DataFrame
    gross_excess: pd.Series
    net_excess: pd.Series
    net_total: pd.Series
    risk_free: pd.Series
    turnover: pd.Series
    traded_notional: pd.Series
    costs: pd.Series
    solver_status: pd.Series
    forecast_mean: pd.DataFrame | None = None
    forecast_cov: np.ndarray | None = None
    gating: pd.DataFrame | None = None
    nll: pd.Series | None = None
    pit: pd.DataFrame | None = None
    fallback_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_probabilistic(self) -> bool:
        return self.forecast_mean is not None and self.forecast_cov is not None

    @property
    def n_splits(self) -> int:
        return len(self.dates)

    def save(self, directory: str | Path) -> None:
        out = ensure_dir(Path(directory))
        weights = self.weights.copy()
        weights.index.name = "decision_date"
        write_csv(out / "weights.csv", weights)

        returns = pd.DataFrame(
            {
                "gross_excess": self.gross_excess,
                "net_excess": self.net_excess,
                "net_total": self.net_total,
                "risk_free": self.risk_free,
                "turnover": self.turnover,
                "traded_notional": self.traded_notional,
                "cost": self.costs,
            }
        )
        returns.index.name = "decision_date"
        write_csv(out / "returns.csv", returns)

        status = self.solver_status.rename("status").to_frame()
        status.index.name = "decision_date"
        write_csv(out / "solver_status.csv", status)

        if self.forecast_mean is not None:
            forecast = self.forecast_mean.copy()
            forecast.index.name = "decision_date"
            write_csv(out / "forecast_mean.csv", forecast)
        if self.forecast_cov is not None:
            np.save(out / "forecast_cov.npy", self.forecast_cov)
        if self.gating is not None:
            gating = self.gating.copy()
            gating.index.name = "decision_date"
            write_csv(out / "gating.csv", gating)
        if self.nll is not None:
            nll = self.nll.rename("nll").to_frame()
            nll.index.name = "decision_date"
            write_csv(out / "nll.csv", nll)
        if self.pit is not None:
            pit = self.pit.copy()
            pit.index.name = "decision_date"
            write_csv(out / "pit.csv", pit)

        summary = {
            "name": self.name,
            "kind": self.kind,
            "method": self.method,
            "n_splits": self.n_splits,
            "first_decision": str(self.dates[0].date()),
            "last_decision": str(self.dates[-1].date()),
            "fallback_count": int(self.fallback_count),
            "fallback_rate": float(self.fallback_count / max(self.n_splits, 1)),
            "mean_turnover": float(self.turnover.mean()),
            "mean_cost": float(self.costs.mean()),
            "mean_deployed": float(self.weights.sum(axis=1).mean()),
            "errors": self.errors,
        }
        write_json(out / "summary.json", summary)


@dataclass
class BacktestSuite:
    models: dict[str, ModelBacktest]
    split_plan: SplitPlan
    config_hash: str
    data_metadata: dict
    feature_manifest: dict

    def save(self, run_dir: str | Path) -> None:
        root = ensure_dir(Path(run_dir))
        backtest_dir = ensure_dir(root / "backtest")
        write_json(backtest_dir / "splits.json", self.split_plan.to_dict())
        write_json(
            backtest_dir / "suite.json",
            {
                "config_hash": self.config_hash,
                "models": list(self.models),
                "n_models": len(self.models),
                "n_splits": len(self.split_plan.splits),
                "data": self.data_metadata,
                "features": self.feature_manifest,
            },
        )
        for name, result in self.models.items():
            result.save(root / "models" / name)


def _drift_weights(weights: np.ndarray, excess_returns: np.ndarray, risk_free: float) -> np.ndarray:
    """Weights after one month of market drift, before the next rebalance.

    A factor sleeve earns ``risk_free + excess_return``. The cash residual earns
    ``risk_free``. The denominator is the total portfolio growth factor, so the
    drifted factor weights sum to the post-return deployed fraction.
    """
    denominator = 1.0 + risk_free + float(weights @ excess_returns)
    if denominator <= 1e-8:
        return np.zeros_like(weights)
    return weights * (1.0 + risk_free + excess_returns) / denominator


def _validate_weights(
    w: np.ndarray,
    n_assets: int,
    config: Config,
    context: str,
    require_fully_invested: bool,
) -> None:
    if w.shape != (n_assets,):
        raise AssertionError(f"{context}: weight shape {w.shape} != ({n_assets},)")
    if not np.isfinite(w).all():
        raise AssertionError(f"{context}: non-finite weights")
    if np.any(w < -1e-9):
        raise AssertionError(f"{context}: negative weight {w.min()}")
    if w.sum() > 1.0 + 1e-6:
        raise AssertionError(f"{context}: weights sum to {w.sum():.6f} > 1")
    if w.max() > config.decision.constraints.max_weight + 1e-6:
        raise AssertionError(
            f"{context}: weight {w.max():.6f} exceeds cap {config.decision.constraints.max_weight}"
        )
    if require_fully_invested and abs(w.sum() - 1.0) > 1e-5:
        raise AssertionError(f"{context}: fully invested weights sum to {w.sum():.6f}")


def _run_single_model(
    spec: ModelConfig,
    market: MarketData,
    features: FeatureSet,
    split_plan: SplitPlan,
    config: Config,
) -> ModelBacktest:
    n_assets = market.n_assets
    model = build_model(spec, n_assets=n_assets, seed=config.run.seed)
    w_prev_target = np.zeros(n_assets)

    dates: list[pd.Timestamp] = []
    target_dates: list[pd.Timestamp] = []
    weights: list[np.ndarray] = []
    gross: list[float] = []
    net_excess: list[float] = []
    net_total: list[float] = []
    risk_free: list[float] = []
    turnover: list[float] = []
    traded: list[float] = []
    costs: list[float] = []
    statuses: list[str] = []
    forecast_means: list[np.ndarray] = []
    forecast_covs: list[np.ndarray] = []
    gating_rows: list[np.ndarray] = []
    nll_values: list[float] = []
    pit_rows: list[np.ndarray] = []

    one_way_cost = config.decision.cost.one_way_bps / 10_000.0
    strategy_drag = config.decision.cost.strategy_drag_bps_per_month / 10_000.0

    for step, split in enumerate(split_plan.splits, start=1):
        t = split.test_idx
        decision_date = split.decision_date
        target_date = split.target_date
        context = f"{spec.name}@{decision_date.date()}"

        if target_date not in market.factors.index:
            raise AssertionError(f"{context}: target date {target_date} missing from market data")
        realized = features.y.iloc[t].to_numpy(dtype=float)
        market_realized = market.factors.loc[target_date].to_numpy(dtype=float)
        np.testing.assert_allclose(realized, market_realized, rtol=1e-12, atol=1e-14)
        rf_value = float(market.risk_free.loc[target_date])

        if step == 1:
            w_prev_effective = np.zeros(n_assets)
        else:
            decision_excess = market.factors.loc[decision_date].to_numpy(dtype=float)
            decision_rf = float(market.risk_free.loc[decision_date])
            w_prev_effective = _drift_weights(w_prev_target, decision_excess, decision_rf)

        if isinstance(model, Allocator):
            history = market.factors.loc[:decision_date]
            w = np.asarray(model.weights(history, decision_date), dtype=float)
            status = "allocator"
            forecast = None
        elif isinstance(model, ForecastModel):
            X_train = features.X.iloc[:t].to_numpy(dtype=float)
            y_train = features.y.iloc[:t].to_numpy(dtype=float)
            x_test = features.X.iloc[t].to_numpy(dtype=float)
            if X_train.shape[0] != y_train.shape[0]:
                raise AssertionError(f"{context}: X/y train row mismatch")
            model.fit(X_train, y_train)
            forecast = model.predict(x_test)
            decision = solve(forecast.mean, forecast.cov, w_prev_effective, config.decision)
            w = decision.weights
            status = decision.status
            forecast_means.append(np.asarray(forecast.mean, dtype=float))
            forecast_covs.append(np.asarray(forecast.cov, dtype=float))
            nll_values.append(forecast.nll(realized))
            pit_rows.append(forecast.pit(realized))
            if spec.method == "moe" and hasattr(model, "predict_proba"):
                gating_rows.append(np.asarray(model.predict_proba(x_test), dtype=float))
        else:  # pragma: no cover - defensive
            raise TypeError(f"unsupported model type: {type(model)!r}")

        _validate_weights(
            w,
            n_assets,
            config,
            context,
            require_fully_invested=(
                isinstance(model, ForecastModel) and config.decision.constraints.fully_invested
            ),
        )
        traded_notional = float(np.abs(w - w_prev_effective).sum())
        turnover_value = 0.5 * traded_notional
        cost = one_way_cost * traded_notional + strategy_drag * float(w.sum())
        gross_excess = float(w @ realized)
        net_excess_value = gross_excess - cost
        net_total_value = rf_value + net_excess_value
        if not np.isfinite([gross_excess, cost, net_total_value]).all():
            raise AssertionError(f"{context}: non-finite return")

        dates.append(decision_date)
        target_dates.append(target_date)
        weights.append(w.copy())
        gross.append(gross_excess)
        net_excess.append(net_excess_value)
        net_total.append(net_total_value)
        risk_free.append(rf_value)
        turnover.append(turnover_value)
        traded.append(traded_notional)
        costs.append(cost)
        statuses.append(status)
        w_prev_target = w

        if step % 50 == 0 or step == len(split_plan.splits):
            logger.info("%s: %d/%d", spec.name, step, len(split_plan.splits))

    date_index = pd.DatetimeIndex(dates)
    weights_frame = pd.DataFrame(weights, index=date_index, columns=market.assets)
    forecast_mean_frame = (
        pd.DataFrame(forecast_means, index=date_index, columns=market.assets)
        if forecast_means
        else None
    )
    forecast_cov_array = np.stack(forecast_covs) if forecast_covs else None
    gating_frame = None
    if gating_rows:
        gating_frame = pd.DataFrame(
            np.vstack(gating_rows),
            index=date_index,
            columns=[f"expert_{i + 1}" for i in range(gating_rows[0].shape[0])],
        )
    nll_series = pd.Series(nll_values, index=date_index, name="nll") if nll_values else None
    pit_frame = (
        pd.DataFrame(pit_rows, index=date_index, columns=market.assets) if pit_rows else None
    )
    status_series = pd.Series(statuses, index=date_index, name="status")
    fallback_count = int(sum(status.startswith("fallback") for status in statuses))
    fallback_rate = fallback_count / max(len(statuses), 1)
    if fallback_rate > config.decision.solver.fallback_max_rate:
        raise RuntimeError(
            f"{spec.name}: solver fallback rate {fallback_rate:.2%} exceeds "
            f"threshold {config.decision.solver.fallback_max_rate:.2%}"
        )

    return ModelBacktest(
        name=spec.name,
        kind=spec.kind,
        method=spec.method,
        dates=date_index,
        target_dates=pd.DatetimeIndex(target_dates),
        weights=weights_frame,
        gross_excess=pd.Series(gross, index=date_index, name="gross_excess"),
        net_excess=pd.Series(net_excess, index=date_index, name="net_excess"),
        net_total=pd.Series(net_total, index=date_index, name="net_total"),
        risk_free=pd.Series(risk_free, index=date_index, name="risk_free"),
        turnover=pd.Series(turnover, index=date_index, name="turnover"),
        traded_notional=pd.Series(traded, index=date_index, name="traded_notional"),
        costs=pd.Series(costs, index=date_index, name="cost"),
        solver_status=status_series,
        forecast_mean=forecast_mean_frame,
        forecast_cov=forecast_cov_array,
        gating=gating_frame,
        nll=nll_series,
        pit=pit_frame,
        fallback_count=fallback_count,
        errors=[],
    )


def run_backtest(
    market: MarketData,
    features: FeatureSet,
    split_plan: SplitPlan,
    config: Config,
) -> BacktestSuite:
    if config.split.test_step != 1:
        raise ValueError("drift-aware accounting currently requires split.test_step == 1")
    results: dict[str, ModelBacktest] = {}
    for spec in config.models:
        logger.info("running model: %s (%s/%s)", spec.name, spec.kind, spec.method)
        results[spec.name] = _run_single_model(spec, market, features, split_plan, config)
    return BacktestSuite(
        models=results,
        split_plan=split_plan,
        config_hash=config.config_hash(),
        data_metadata=market.metadata,
        feature_manifest=features.manifest,
    )
