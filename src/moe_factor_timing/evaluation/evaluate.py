"""Evaluation orchestration: metrics, uncertainty, calibration, and DSR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..backtest.engine import BacktestSuite
from ..config import Config
from ..utils.io import ensure_dir, write_csv, write_json
from .bootstrap import bootstrap_table
from .calibration import calibration_by_asset, calibration_by_model
from .deflated_sharpe import deflated_sharpe_table
from .metrics import cost_sensitivity_frame, metrics_frame, subperiod_frame
from .power import power_summary


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    holdout_metrics: pd.DataFrame
    bootstrap_full: pd.DataFrame
    bootstrap_holdout: pd.DataFrame
    calibration: pd.DataFrame
    calibration_assets: pd.DataFrame
    deflated: pd.DataFrame
    subperiods: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    power: dict
    meta: dict

    def save(self, directory: str | Path) -> None:
        out = ensure_dir(Path(directory))
        write_csv(out / "metrics.csv", self.metrics)
        write_csv(out / "holdout_metrics.csv", self.holdout_metrics)
        write_csv(out / "bootstrap_full.csv", self.bootstrap_full)
        write_csv(out / "bootstrap_holdout.csv", self.bootstrap_holdout)
        write_csv(out / "calibration.csv", self.calibration)
        write_csv(out / "calibration_by_asset.csv", self.calibration_assets, index=False)
        write_csv(out / "deflated_sharpe.csv", self.deflated)
        write_csv(out / "subperiods.csv", self.subperiods, index=False)
        write_csv(out / "cost_sensitivity.csv", self.cost_sensitivity, index=False)
        write_json(out / "power.json", self.power)
        write_json(out / "evaluation.json", self.meta)


def evaluate_suite(suite: BacktestSuite, config: Config) -> EvaluationResult:
    model_results = suite.models
    if not model_results:
        raise ValueError("no model results to evaluate")

    benchmark = "one_over_n" if "one_over_n" in model_results else next(iter(model_results))
    reference = model_results[benchmark]
    risk_aversion = config.decision.risk_aversion

    metrics = metrics_frame(model_results, risk_aversion)
    holdout_start = suite.split_plan.holdout_start
    if holdout_start is not None:
        holdout_mask = pd.Series(
            reference.dates >= holdout_start, index=reference.dates, dtype=bool
        )
    else:
        holdout_mask = None
    holdout_metrics = (
        metrics_frame(model_results, risk_aversion, mask=holdout_mask)
        if holdout_mask is not None
        else pd.DataFrame()
    )

    bootstrap_full = bootstrap_table(
        model_results,
        benchmark_name=benchmark,
        n_repetitions=config.evaluation.bootstrap.n_repetitions,
        block_length=config.evaluation.bootstrap.block_length,
        seed=config.evaluation.bootstrap.seed,
    )
    bootstrap_holdout = (
        bootstrap_table(
            model_results,
            benchmark_name=benchmark,
            n_repetitions=config.evaluation.bootstrap.n_repetitions,
            block_length=config.evaluation.bootstrap.block_length,
            seed=config.evaluation.bootstrap.seed,
            mask=holdout_mask,
        )
        if holdout_mask is not None
        else pd.DataFrame()
    )

    calibration = calibration_by_model(model_results)
    calibration_assets = calibration_by_asset(model_results)
    deflated = deflated_sharpe_table(metrics, n_trials=config.evaluation.deflated_sharpe.n_trials)
    subperiods = subperiod_frame(model_results, risk_aversion, n_periods=3)
    cost_sensitivity = cost_sensitivity_frame(model_results)

    n_full = int(len(reference.dates))
    n_holdout = int(holdout_mask.sum()) if holdout_mask is not None else 0
    power = power_summary(
        n_months_full=n_full,
        n_months_holdout=n_holdout,
        alpha=config.evaluation.power.alpha,
        power=config.evaluation.power.power,
    )
    meta = {
        "benchmark": benchmark,
        "primary_metric": config.evaluation.primary_metric,
        "n_models": len(model_results),
        "n_splits": n_full,
        "n_holdout": n_holdout,
        "holdout_start": None if holdout_start is None else str(holdout_start.date()),
        "first_decision": str(reference.dates[0].date()),
        "last_decision": str(reference.dates[-1].date()),
        "risk_aversion": risk_aversion,
        "one_way_cost_bps": config.decision.cost.one_way_bps,
        "strategy_drag_bps_per_month": config.decision.cost.strategy_drag_bps_per_month,
        "n_trials": config.evaluation.deflated_sharpe.n_trials,
        "config_hash": suite.config_hash,
        "bootstrap": asdict(config.evaluation.bootstrap),
        "power_config": asdict(config.evaluation.power),
    }

    return EvaluationResult(
        metrics=metrics,
        holdout_metrics=holdout_metrics,
        bootstrap_full=bootstrap_full,
        bootstrap_holdout=bootstrap_holdout,
        calibration=calibration,
        calibration_assets=calibration_assets,
        deflated=deflated,
        subperiods=subperiods,
        cost_sensitivity=cost_sensitivity,
        power=power,
        meta=meta,
    )
