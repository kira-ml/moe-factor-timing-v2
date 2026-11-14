"""Stationary block bootstrap for Sharpe-ratio differences.

Monthly strategy returns are serially dependent and heteroskedastic. An iid
bootstrap would understate uncertainty. We resample circular blocks with a
fixed block length, which preserves short-range dependence and is fully
reproducible given the seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def circular_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_length)
    indices = np.concatenate([(start + offsets) % n for start in starts])
    return indices[:n]


def _sharpe(values: np.ndarray) -> float:
    std = np.std(values, ddof=1)
    if std <= 1e-15:
        return float("nan")
    return float(np.mean(values) / std * np.sqrt(12.0))


def sharpe_difference(
    model_excess: pd.Series,
    benchmark_excess: pd.Series,
    n_repetitions: int = 1000,
    block_length: int = 6,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap the Sharpe difference (model minus benchmark)."""
    aligned = pd.concat([model_excess, benchmark_excess], axis=1, join="inner").dropna()
    if len(aligned) < 12:
        return {
            "n_obs": int(len(aligned)),
            "sharpe_model": float("nan"),
            "sharpe_benchmark": float("nan"),
            "diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }
    model = aligned.iloc[:, 0].to_numpy(dtype=float)
    benchmark = aligned.iloc[:, 1].to_numpy(dtype=float)
    observed_diff = _sharpe(model) - _sharpe(benchmark)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_repetitions)
    for i in range(n_repetitions):
        indices = circular_block_indices(len(model), block_length, rng)
        draws[i] = _sharpe(model[indices]) - _sharpe(benchmark[indices])
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return {
            "n_obs": int(len(aligned)),
            "sharpe_model": _sharpe(model),
            "sharpe_benchmark": _sharpe(benchmark),
            "diff": observed_diff,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }
    lower, upper = np.percentile(draws, [2.5, 97.5])
    fraction_below_zero = float(np.mean(draws <= 0.0))
    p_value = float(min(1.0, 2.0 * min(fraction_below_zero, 1.0 - fraction_below_zero)))
    return {
        "n_obs": int(len(aligned)),
        "sharpe_model": _sharpe(model),
        "sharpe_benchmark": _sharpe(benchmark),
        "diff": observed_diff,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "p_value": p_value,
    }


def bootstrap_table(
    model_results: dict,
    benchmark_name: str,
    n_repetitions: int,
    block_length: int,
    seed: int,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    if benchmark_name not in model_results:
        return pd.DataFrame()
    benchmark = model_results[benchmark_name].net_excess
    rows: dict[str, dict] = {}
    for name, result in model_results.items():
        if name == benchmark_name:
            continue
        model = result.net_excess
        if mask is not None:
            model = model.loc[mask]
            benchmark_series = benchmark.loc[mask]
        else:
            benchmark_series = benchmark
        rows[name] = sharpe_difference(
            model,
            benchmark_series,
            n_repetitions=n_repetitions,
            block_length=block_length,
            seed=seed,
        )
    frame = pd.DataFrame(rows).T
    frame.index.name = "model"
    frame = frame.astype(float)
    frame["benchmark"] = benchmark_name
    return frame
