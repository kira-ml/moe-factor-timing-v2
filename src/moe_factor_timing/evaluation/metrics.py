"""Performance and risk metrics.

Returns are decimal. ``net_total`` includes the risk-free rate; ``net_excess``
is the part attributable to factor allocation after costs. Sharpe, Sortino and
mean-variance utility use ``net_excess``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

TRADING_MONTHS = 12


def annualized_return(net_total: pd.Series) -> float:
    values = np.asarray(net_total, dtype=float)
    if len(values) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + values))
    if growth <= 0:
        return -1.0
    return float(growth ** (TRADING_MONTHS / len(values)) - 1.0)


def annualized_volatility(net_excess: pd.Series) -> float:
    values = np.asarray(net_excess, dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) * np.sqrt(TRADING_MONTHS))


def sharpe_ratio(net_excess: pd.Series) -> float:
    values = np.asarray(net_excess, dtype=float)
    if len(values) < 2:
        return float("nan")
    std = np.std(values, ddof=1)
    if std <= 1e-15:
        return float("nan")
    return float(np.mean(values) / std * np.sqrt(TRADING_MONTHS))


def sortino_ratio(net_excess: pd.Series) -> float:
    values = np.asarray(net_excess, dtype=float)
    if len(values) < 2:
        return float("nan")
    downside = np.minimum(values, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    if downside_deviation <= 1e-15:
        return float("nan")
    return float(np.mean(values) / downside_deviation * np.sqrt(TRADING_MONTHS))


def drawdown_series(net_total: pd.Series) -> pd.Series:
    wealth = (1.0 + np.asarray(net_total, dtype=float)).cumprod()
    running_max = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_max - 1.0
    return pd.Series(drawdowns, index=net_total.index, name="drawdown")


def maximum_drawdown(net_total: pd.Series) -> float:
    if len(net_total) == 0:
        return float("nan")
    return float(drawdown_series(net_total).min())


def calmar_ratio(net_total: pd.Series) -> float:
    annual = annualized_return(net_total)
    drawdown = maximum_drawdown(net_total)
    if not np.isfinite(drawdown) or abs(drawdown) < 1e-15:
        return float("nan")
    return float(annual / abs(drawdown))


def mean_variance_utility(net_excess: pd.Series, risk_aversion: float) -> float:
    values = np.asarray(net_excess, dtype=float)
    if len(values) < 2:
        return float("nan")
    return float((np.mean(values) - 0.5 * risk_aversion * np.var(values, ddof=1)) * TRADING_MONTHS)


def hit_rate(net_excess: pd.Series) -> float:
    values = np.asarray(net_excess, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values > 0.0))


def rolling_sharpe(net_excess: pd.Series, window: int = 36) -> pd.Series:
    values = pd.Series(net_excess, dtype=float)
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std(ddof=1)
    return (mean / std * np.sqrt(TRADING_MONTHS)).rename("rolling_sharpe")


def compute_metrics(
    net_total: pd.Series,
    net_excess: pd.Series,
    weights: pd.DataFrame,
    turnover: pd.Series,
    costs: pd.Series,
    risk_aversion: float,
) -> dict[str, float]:
    values = np.asarray(net_excess, dtype=float)
    return {
        "n_obs": int(len(values)),
        "ann_return": annualized_return(net_total),
        "ann_vol": annualized_volatility(net_excess),
        "sharpe": sharpe_ratio(net_excess),
        "sortino": sortino_ratio(net_excess),
        "max_drawdown": maximum_drawdown(net_total),
        "calmar": calmar_ratio(net_total),
        "utility": mean_variance_utility(net_excess, risk_aversion),
        "hit_rate": hit_rate(net_excess),
        "skew": float(skew(values, bias=False)) if len(values) > 2 else float("nan"),
        "excess_kurtosis": (
            float(kurtosis(values, fisher=True, bias=False)) if len(values) > 3 else float("nan")
        ),
        "avg_turnover": float(turnover.mean()),
        "annual_turnover": float(turnover.mean() * TRADING_MONTHS),
        "avg_cost_bps": float(costs.mean() * 10_000),
        "annual_cost_drag": float(costs.mean() * TRADING_MONTHS),
        "avg_deployed": float(weights.sum(axis=1).mean()),
        "avg_max_weight": float(weights.max(axis=1).mean()),
    }


def metrics_frame(
    model_results: dict, risk_aversion: float, mask: pd.Series | None = None
) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for name, result in model_results.items():
        if mask is None:
            net_total = result.net_total
            net_excess = result.net_excess
            weights = result.weights
            turnover = result.turnover
            costs = result.costs
        else:
            net_total = result.net_total.loc[mask]
            net_excess = result.net_excess.loc[mask]
            weights = result.weights.loc[mask]
            turnover = result.turnover.loc[mask]
            costs = result.costs.loc[mask]
        rows[name] = compute_metrics(
            net_total=net_total,
            net_excess=net_excess,
            weights=weights,
            turnover=turnover,
            costs=costs,
            risk_aversion=risk_aversion,
        )
        rows[name]["gross_sharpe"] = sharpe_ratio(
            result.gross_excess.loc[mask] if mask is not None else result.gross_excess
        )
    frame = pd.DataFrame(rows).T
    frame.index.name = "model"
    return frame.astype(float)


def subperiod_frame(model_results: dict, risk_aversion: float, n_periods: int = 3) -> pd.DataFrame:
    """Equal-length chronological subperiods of the common evaluation window."""
    if not model_results:
        return pd.DataFrame()
    reference = next(iter(model_results.values()))
    index = reference.dates
    chunks = np.array_split(np.arange(len(index)), n_periods)
    rows: list[dict] = []
    for period, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue
        start, end = index[chunk[0]], index[chunk[-1]]
        mask = (reference.dates >= start) & (reference.dates <= end)
        frame = metrics_frame(
            model_results, risk_aversion, mask=pd.Series(mask, index=reference.dates)
        )
        for model in frame.index:
            rows.append(
                {
                    "model": model,
                    "period": period,
                    "start": start,
                    "end": end,
                    "n_obs": int(frame.loc[model, "n_obs"]),
                    "sharpe": float(frame.loc[model, "sharpe"]),
                    "ann_return": float(frame.loc[model, "ann_return"]),
                    "max_drawdown": float(frame.loc[model, "max_drawdown"]),
                }
            )
    return pd.DataFrame(rows)


def cost_sensitivity_frame(
    model_results: dict, cost_levels_bps: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)
) -> pd.DataFrame:
    """Sharpe ratios under alternative one-way cost assumptions.

    The decisions are held fixed; only the realised cost is recomputed from the
    recorded traded notional. This isolates cost sensitivity from the optimizer's
    response to a different cost parameter.
    """
    rows: list[dict] = []
    for name, result in model_results.items():
        for cost_bps in cost_levels_bps:
            cost = cost_bps / 10_000.0 * result.traded_notional
            net_excess = result.gross_excess - cost
            net_total = result.risk_free + net_excess
            rows.append(
                {
                    "model": name,
                    "cost_bps": float(cost_bps),
                    "sharpe": sharpe_ratio(net_excess),
                    "ann_return": annualized_return(net_total),
                    "ann_vol": annualized_volatility(net_excess),
                    "max_drawdown": maximum_drawdown(net_total),
                }
            )
    return pd.DataFrame(rows)
