"""Restart-stability audit for the MoE.

The reference run uses one deterministic k-means initialisation per
walk-forward fit. This script checks whether additional restarts change the
training likelihood or the next-month predictive NLL on a sample of training
windows. It is a diagnostic, not a model-selection procedure.

Usage:
    python scripts/moe_stability.py --n-windows 24 --extra-restarts 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_factor_timing.config import load_config  # noqa: E402
from moe_factor_timing.data.loader import load_market_data  # noqa: E402
from moe_factor_timing.features.build import build_features  # noqa: E402
from moe_factor_timing.models.moe import MixtureOfExpertsModel  # noqa: E402
from moe_factor_timing.utils.io import ensure_dir, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-windows", type=int, default=24)
    parser.add_argument("--extra-restarts", type=int, default=3)
    parser.add_argument("--output", default="reports/moe_stability.json")
    args = parser.parse_args()

    config = load_config(args.config)
    market = load_market_data(config)
    features = build_features(market, config)
    X = features.X.to_numpy(dtype=float)
    y = features.y.to_numpy(dtype=float)

    start = config.split.min_train
    indices = np.linspace(start, len(X) - 1, args.n_windows, dtype=int)
    moe_spec = next(spec for spec in config.models if spec.method == "moe")
    rows: list[dict] = []
    for t in indices:
        params = dict(
            n_experts=int(moe_spec.params.get("n_experts", 3)),
            max_iter=int(moe_spec.params.get("max_iter", 25)),
            gating_steps=int(moe_spec.params.get("gating_steps", 6)),
            seed=config.run.seed,
        )
        base = MixtureOfExpertsModel(n_restarts=1, **params).fit(X[:t], y[:t])
        multi = MixtureOfExpertsModel(n_restarts=args.extra_restarts, **params).fit(X[:t], y[:t])
        forecast_base = base.predict(X[t])
        forecast_multi = multi.predict(X[t])
        rows.append(
            {
                "test_idx": int(t),
                "decision_date": str(features.decision_dates[t].date()),
                "train_ll_base": float(base.log_likelihood_),
                "train_ll_multi": float(multi.log_likelihood_),
                "test_nll_base": float(forecast_base.nll(y[t])),
                "test_nll_multi": float(forecast_multi.nll(y[t])),
                "base_iterations": int(base.n_iter_),
                "multi_iterations": int(multi.n_iter_),
            }
        )

    ll_diff = np.array([row["train_ll_multi"] - row["train_ll_base"] for row in rows])
    nll_diff = np.array([row["test_nll_multi"] - row["test_nll_base"] for row in rows])
    summary = {
        "n_windows": len(rows),
        "extra_restarts": args.extra_restarts,
        "mean_train_ll_gain": float(ll_diff.mean()),
        "max_train_ll_gain": float(ll_diff.max()),
        "fraction_windows_restarts_better_ll": float(np.mean(ll_diff > 1e-9)),
        "mean_test_nll_change": float(nll_diff.mean()),
        "fraction_windows_restarts_better_nll": float(np.mean(nll_diff < -1e-9)),
        "rows": rows,
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
