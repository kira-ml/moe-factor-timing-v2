"""Decision-rule sensitivity from saved forecast distributions.

Re-solves the mean-variance decision at alternative risk-aversion and position-cap
values without refitting the forecasting models. This isolates the contribution
of the decision rule from the contribution of the forecast.

Usage:
    python scripts/decision_sensitivity.py --run-dir runs/reference
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_factor_timing.backtest.engine import _drift_weights  # noqa: E402
from moe_factor_timing.config import load_config  # noqa: E402
from moe_factor_timing.decision.mean_variance import solve  # noqa: E402
from moe_factor_timing.evaluation.metrics import (  # noqa: E402
    annualized_return,
    annualized_volatility,
    maximum_drawdown,
    sharpe_ratio,
)
from moe_factor_timing.utils.io import ensure_dir, write_csv  # noqa: E402


def _metrics(
    net_total: np.ndarray, net_excess: np.ndarray, turnover: np.ndarray, costs: np.ndarray
) -> dict:
    total = pd.Series(net_total)
    excess = pd.Series(net_excess)
    return {
        "sharpe": sharpe_ratio(excess),
        "ann_return": annualized_return(total),
        "ann_vol": annualized_volatility(excess),
        "max_drawdown": maximum_drawdown(total),
        "avg_turnover": float(np.mean(turnover)),
        "avg_cost_bps": float(np.mean(costs) * 10_000.0),
    }


def _run_model(
    model_name: str,
    run_dir: Path,
    config,
    risk_aversion: float,
    max_weight: float,
    cov_scale: float,
) -> dict:
    forecast_mean = pd.read_csv(
        run_dir / "models" / model_name / "forecast_mean.csv", index_col=0, parse_dates=True
    )
    forecast_cov = np.load(run_dir / "models" / model_name / "forecast_cov.npy")
    y = pd.read_csv(run_dir / "features" / "y.csv", index_col=0, parse_dates=True)
    y = y.loc[forecast_mean.index]
    dates = pd.read_csv(
        run_dir / "features" / "dates.csv", parse_dates=["decision_date", "target_date"]
    ).set_index("decision_date")
    factors = pd.read_csv(run_dir / "data" / "factors.csv", index_col=0, parse_dates=True)
    risk_free = pd.read_csv(run_dir / "data" / "risk_free.csv", index_col=0, parse_dates=True)["RF"]

    decision_config = replace(
        config.decision,
        risk_aversion=risk_aversion,
        constraints=replace(config.decision.constraints, max_weight=max_weight),
    )
    cost_per_unit = decision_config.cost.one_way_bps / 10_000.0
    w_prev_target = np.zeros(factors.shape[1])

    net_total: list[float] = []
    net_excess: list[float] = []
    turnover: list[float] = []
    costs: list[float] = []
    statuses: list[str] = []

    for i in range(len(forecast_mean)):
        decision_date = forecast_mean.index[i]
        target_date = dates.loc[decision_date, "target_date"]
        if i == 0:
            drifted = np.zeros(factors.shape[1])
        else:
            drifted = _drift_weights(
                w_prev_target,
                factors.loc[decision_date].to_numpy(dtype=float),
                float(risk_free.loc[decision_date]),
            )
        mu = forecast_mean.iloc[i].to_numpy(dtype=float)
        cov = forecast_cov[i] * cov_scale
        result = solve(mu, cov, drifted, decision_config)
        w = result.weights
        traded = float(np.abs(w - drifted).sum())
        cost = cost_per_unit * traded
        realized = y.loc[decision_date].to_numpy(dtype=float)
        rf_value = float(risk_free.loc[target_date])
        gross = float(w @ realized)
        net_excess_value = gross - cost
        net_total.append(rf_value + net_excess_value)
        net_excess.append(net_excess_value)
        turnover.append(0.5 * traded)
        costs.append(cost)
        statuses.append(result.status)
        w_prev_target = w

    metrics = _metrics(
        np.asarray(net_total), np.asarray(net_excess), np.asarray(turnover), np.asarray(costs)
    )
    metrics.update(
        {
            "model": model_name,
            "risk_aversion": risk_aversion,
            "max_weight": max_weight,
            "cov_scale": cov_scale,
            "fallback_rate": float(np.mean([s.startswith("fallback") for s in statuses])),
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/reference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="reports/decision_sensitivity.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(args.config)
    scenarios = [
        (2.0, 0.35, 1.0),
        (5.0, 0.35, 1.0),
        (10.0, 0.35, 1.0),
        (2.0, 0.25, 1.0),
        (2.0, 0.50, 1.0),
        (2.0, 0.35, 2.0),
        (2.0, 0.35, 4.0),
    ]
    rows: list[dict] = []
    for model_dir in sorted((run_dir / "models").iterdir()):
        if not (model_dir / "forecast_mean.csv").exists():
            continue
        for risk_aversion, max_weight, cov_scale in scenarios:
            rows.append(
                _run_model(
                    model_dir.name,
                    run_dir,
                    config,
                    risk_aversion=risk_aversion,
                    max_weight=max_weight,
                    cov_scale=cov_scale,
                )
            )
            print(
                f"{model_dir.name} lambda={risk_aversion} cap={max_weight} "
                f"cov_scale={cov_scale} done"
            )

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    ensure_dir(output.parent)
    write_csv(output, frame, index=False)
    print(json.dumps({"rows": len(frame), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
