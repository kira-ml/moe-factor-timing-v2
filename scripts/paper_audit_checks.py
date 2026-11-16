"""Post-publication audit diagnostics for the MoE factor-timing paper.

The reference protocol did not include a block-permutation null, a signal-delay
test, or gating diagnostics. This script computes them from the saved reference
run so the paper's Table 9 can be reproduced without refitting the model.

Usage:
    python scripts/paper_audit_checks.py --run-dir runs/reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan")


def _row_rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(a.shape[0])[:, None]
    ranks[rows, order] = np.arange(a.shape[1])[None, :]
    return ranks


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return (a * b).sum(axis=1) / denom


def _blocks(n: int, size: int) -> list[np.ndarray]:
    idx = np.arange(n)
    return [idx[i * size : min((i + 1) * size, n)] for i in range(int(np.ceil(n / size)))]


def _load(run_dir: Path, data_path: Path):
    factors = pd.read_csv(data_path, parse_dates=["date"], index_col="date").sort_index()
    factors.index = pd.DatetimeIndex(factors.index)
    weights = pd.read_csv(run_dir / "models" / "moe" / "weights.csv", index_col=0, parse_dates=True)
    one_over_n = pd.read_csv(
        run_dir / "models" / "one_over_n" / "weights.csv", index_col=0, parse_dates=True
    )
    returns = pd.read_csv(run_dir / "models" / "moe" / "returns.csv", index_col=0, parse_dates=True)
    one_returns = pd.read_csv(
        run_dir / "models" / "one_over_n" / "returns.csv", index_col=0, parse_dates=True
    )
    forecast = pd.read_csv(
        run_dir / "models" / "moe" / "forecast_mean.csv", index_col=0, parse_dates=True
    )
    gating = pd.read_csv(run_dir / "models" / "moe" / "gating.csv", index_col=0, parse_dates=True)

    assets = list(weights.columns)
    next_return = factors[assets].shift(-1)
    common = weights.index.intersection(next_return.dropna().index)
    w_moe = weights.loc[common].to_numpy(float)
    w_on = one_over_n.loc[common].to_numpy(float)
    r_next = next_return.loc[common].to_numpy(float)
    cost_moe = returns.loc[common, "cost"].to_numpy(float)
    cost_on = one_returns.loc[common, "cost"].to_numpy(float)
    rf = returns.loc[common, "risk_free"].to_numpy(float)
    f_mean = forecast.loc[common].to_numpy(float)
    gating = gating.loc[common]
    return {
        "factors": factors,
        "assets": assets,
        "common": common,
        "w_moe": w_moe,
        "w_on": w_on,
        "r_next": r_next,
        "cost_moe": cost_moe,
        "cost_on": cost_on,
        "rf": rf,
        "f_mean": f_mean,
        "gating": gating,
        "moe_gross_saved": returns.loc[common, "gross_excess"].to_numpy(float),
        "on_gross_saved": one_returns.loc[common, "gross_excess"].to_numpy(float),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/reference")
    parser.add_argument("--data", default="data/reference/factors.csv")
    parser.add_argument("--output", default="reports/paper_audit_checks.json")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--block", type=int, default=6)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    d = _load(run_dir, Path(args.data))
    w_moe, w_on, r_next = d["w_moe"], d["w_on"], d["r_next"]
    cost_moe, cost_on = d["cost_moe"], d["cost_on"]
    f_mean = d["f_mean"]
    gating = d["gating"]
    n = len(d["common"])

    gross_moe = (w_moe * r_next).sum(axis=1)
    gross_on = (w_on * r_next).sum(axis=1)
    gross_match_moe = float(np.max(np.abs(gross_moe - d["moe_gross_saved"])))
    gross_match_on = float(np.max(np.abs(gross_on - d["on_gross_saved"])))

    obs_moe = _sharpe(gross_moe - cost_moe)
    obs_on = _sharpe(gross_on - cost_on)
    obs_diff = obs_moe - obs_on
    f_rank = _row_rank(f_mean)
    obs_ic = float(_row_corr(f_rank, _row_rank(r_next)).mean())
    holdout_mask = d["common"] >= pd.Timestamp("2006-01-31")
    obs_ic_holdout = float(_row_corr(f_rank[holdout_mask], _row_rank(r_next[holdout_mask])).mean())

    rng = np.random.default_rng(20261117)
    blocks = _blocks(n, args.block)
    moe_null = np.empty(args.reps)
    on_null = np.empty(args.reps)
    diff_null = np.empty(args.reps)
    ic_null = np.empty(args.reps)
    ic_null_holdout = np.empty(args.reps)

    for rep in range(args.reps):
        order = rng.permutation(len(blocks))
        perm = np.concatenate([blocks[i] for i in order])[:n]
        rp = r_next[perm]
        gm = (w_moe * rp).sum(axis=1)
        go = (w_on * rp).sum(axis=1)
        sm = _sharpe(gm - cost_moe)
        so = _sharpe(go - cost_on)
        moe_null[rep] = sm
        on_null[rep] = so
        diff_null[rep] = sm - so
        rp_rank = _row_rank(rp)
        ic_null[rep] = float(_row_corr(f_rank, rp_rank).mean())
        ic_null_holdout[rep] = float(_row_corr(f_rank[holdout_mask], rp_rank[holdout_mask]).mean())

    def quantiles(x: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "q025": float(np.quantile(x, 0.025)),
            "q975": float(np.quantile(x, 0.975)),
        }

    # Signal delay: apply the weights at t to the return at t+1+k.
    delay = {}
    for k in (0, 1, 2, 3):
        target = d["factors"][d["assets"]].shift(-(1 + k))
        common = d["common"].intersection(target.dropna().index)
        ret = (w_moe[: len(common)] * target.loc[common].to_numpy(float)).sum(axis=1) - cost_moe[
            : len(common)
        ]
        delay[str(k)] = {
            "n": int(len(common)),
            "sharpe": _sharpe(ret),
            "ann_return": float(ret.mean() * 12),
        }

    entropy = -(gating * np.log(gating.clip(lower=1e-12))).sum(axis=1)
    dominant = gating.max(axis=1)
    gating_summary = {
        "n_months": int(len(gating)),
        "n_experts": int(gating.shape[1]),
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(entropy.median()),
        "entropy_max": float(np.log(gating.shape[1])),
        "share_entropy_lt_0.1": float((entropy < 0.1).mean()),
        "share_entropy_lt_0.3": float((entropy < 0.3).mean()),
        "share_dominant_gt_0.9": float((dominant > 0.9).mean()),
        "share_dominant_gt_0.99": float((dominant > 0.99).mean()),
        "argmax_counts": {str(k): int(v) for k, v in gating.idxmax(axis=1).value_counts().items()},
        "mean_weight": {str(k): float(v) for k, v in gating.mean().items()},
    }
    periods = {
        "1974-1991": ("1974-06-30", "1991-10-31"),
        "1991-2009": ("1991-11-30", "2009-03-31"),
        "2009-2026": ("2009-04-30", "2026-08-31"),
    }
    gating_periods = {}
    for label, (start, end) in periods.items():
        mask = (gating.index >= start) & (gating.index <= end)
        gating_periods[label] = {
            "n": int(mask.sum()),
            "mean_weight": {str(k): float(v) for k, v in gating.loc[mask].mean().items()},
            "entropy_mean": float(entropy.loc[mask].mean()),
            "share_dominant_gt_0.9": float((dominant.loc[mask] > 0.9).mean()),
        }

    payload = {
        "run_dir": str(run_dir),
        "n_months": int(n),
        "block_size": int(args.block),
        "reps": int(args.reps),
        "seed": 20261117,
        "gross_match_max_abs": {"moe": gross_match_moe, "one_over_n": gross_match_on},
        "observed": {
            "moe_sharpe": obs_moe,
            "one_over_n_sharpe": obs_on,
            "difference": obs_diff,
            "mean_ic": obs_ic,
            "mean_ic_holdout": obs_ic_holdout,
        },
        "null_moe_sharpe": {
            **quantiles(moe_null),
            "p_ge_observed": float((moe_null >= obs_moe).mean()),
        },
        "null_one_over_n_sharpe": {
            **quantiles(on_null),
            "p_ge_observed": float((on_null >= obs_on).mean()),
        },
        "null_difference": {
            **quantiles(diff_null),
            "p_le_observed": float((diff_null <= obs_diff).mean()),
            "p_two_sided": float(
                (np.abs(diff_null - diff_null.mean()) >= abs(obs_diff - diff_null.mean())).mean()
            ),
        },
        "null_mean_ic": {
            **quantiles(ic_null),
            "p_ge_observed": float((ic_null >= obs_ic).mean()),
        },
        "null_mean_ic_holdout": {
            **quantiles(ic_null_holdout),
            "p_ge_observed": float((ic_null_holdout >= obs_ic_holdout).mean()),
        },
        "signal_delay": delay,
        "gating": gating_summary,
        "gating_periods": gating_periods,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
