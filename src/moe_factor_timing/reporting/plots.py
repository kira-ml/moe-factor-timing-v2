"""Static report figures (Agg backend, deterministic styling)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..backtest.engine import BacktestSuite, ModelBacktest  # noqa: E402
from ..evaluation.calibration import pit_histogram  # noqa: E402
from ..evaluation.metrics import drawdown_series, rolling_sharpe  # noqa: E402
from ..utils.io import ensure_dir  # noqa: E402

_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#17becf",
    "#7f7f7f",
]


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_returns(suite: BacktestSuite, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (name, result) in enumerate(suite.models.items()):
        wealth = (1.0 + result.net_total).cumprod()
        ax.plot(
            wealth.index, wealth.values, label=name, linewidth=1.6, color=_COLORS[i % len(_COLORS)]
        )
    ax.set_yscale("log")
    _style(ax, "Cumulative net wealth (log scale)", "Wealth index")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


def plot_drawdowns(suite: BacktestSuite, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, result) in enumerate(suite.models.items()):
        drawdowns = drawdown_series(result.net_total)
        ax.plot(
            drawdowns.index,
            drawdowns.values * 100.0,
            label=name,
            linewidth=1.2,
            color=_COLORS[i % len(_COLORS)],
        )
    _style(ax, "Drawdowns", "Drawdown (%)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


def plot_rolling_sharpe(suite: BacktestSuite, path: Path, window: int = 36) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, result) in enumerate(suite.models.items()):
        series = rolling_sharpe(result.net_excess, window=window)
        ax.plot(
            series.index, series.values, label=name, linewidth=1.2, color=_COLORS[i % len(_COLORS)]
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    _style(ax, f"Rolling {window}-month Sharpe ratio", "Sharpe")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


def plot_weights(result: ModelBacktest, path: Path) -> None:
    weights = result.weights.copy()
    cash = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.stackplot(
        weights.index,
        [weights[column].to_numpy() for column in weights.columns],
        labels=list(weights.columns),
        colors=_COLORS[: weights.shape[1]],
        alpha=0.9,
    )
    if cash.max() > 1e-6:
        ax.fill_between(weights.index, 0.0, cash.to_numpy(), color="#dddddd", label="cash")
    _style(ax, f"{result.name}: portfolio weights", "Weight")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def plot_gating(result: ModelBacktest, path: Path) -> None:
    if result.gating is None:
        return
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.stackplot(
        result.gating.index,
        [result.gating[column].to_numpy() for column in result.gating.columns],
        labels=list(result.gating.columns),
        colors=_COLORS[: result.gating.shape[1]],
        alpha=0.9,
    )
    _style(ax, f"{result.name}: gating probabilities", "Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def plot_pit_histograms(suite: BacktestSuite, path: Path) -> None:
    histograms = pit_histogram(suite.models)
    if not histograms:
        return
    n = len(histograms)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    for ax, (name, (counts, edges)) in zip(axes[0], histograms.items(), strict=True):
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#1f77b4", alpha=0.8)
        ax.axhline(counts.mean(), color="black", linewidth=0.8, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("PIT")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25, linewidth=0.6)
    _save(fig, path)


def generate_all_plots(suite: BacktestSuite, directory: Path) -> list[Path]:
    out = ensure_dir(directory)
    written: list[Path] = []
    targets = [
        (plot_cumulative_returns, out / "cumulative_returns.png"),
        (plot_drawdowns, out / "drawdowns.png"),
        (plot_rolling_sharpe, out / "rolling_sharpe.png"),
        (plot_pit_histograms, out / "pit_histograms.png"),
    ]
    for function, path in targets:
        function(suite, path)
        if path.exists():
            written.append(path)
    for name, result in suite.models.items():
        path = out / f"weights_{name}.png"
        plot_weights(result, path)
        written.append(path)
    if "moe" in suite.models:
        path = out / "gating_moe.png"
        plot_gating(suite.models["moe"], path)
        if path.exists():
            written.append(path)
    return written
