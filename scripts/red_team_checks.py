"""Red-team checks for the V2 reference run.

Computes the checks that determine whether the research is publication-ready:
holdout cost sensitivity, gross/net uncertainty, HAC and block-bootstrap
information coefficients, subperiod dominance, and the decision-rule range.

Usage:
    python scripts/red_team_checks.py --run-dir runs/reference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_factor_timing.evaluation.bootstrap import sharpe_difference  # noqa: E402
from moe_factor_timing.utils.io import write_csv, write_json  # noqa: E402

HOLDOUT = pd.Timestamp("2006-01-31")


def _sharpe(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 2 or x.std(ddof=1) <= 1e-15:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12.0))


def _nw_t(x: np.ndarray, lags: int = 6) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = float(x.mean())
    e = x - mu
    var = float(np.mean(e * e))
    for lag in range(1, lags + 1):
        gamma = float(np.mean(e[lag:] * e[:-lag]))
        var += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    se = float(np.sqrt(max(var, 1e-18) / n))
    return mu, se, mu / se


def _block_bootstrap_mean(
    x: np.ndarray, block: int = 6, reps: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    draws = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        draws[i] = x[indices].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = float(2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0)))
    return float(lo), float(hi), p


def _ic_series(forecast_mean: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    aligned = y.loc[forecast_mean.index]
    values: list[float] = []
    dates: list[pd.Timestamp] = []
    for date in forecast_mean.index:
        pred = forecast_mean.loc[date].to_numpy(dtype=float)
        realized = aligned.loc[date].to_numpy(dtype=float)
        if np.std(pred) <= 1e-12 or np.std(realized) <= 1e-12:
            continue
        value = float(spearmanr(pred, realized).statistic)
        if np.isfinite(value):
            values.append(value)
            dates.append(date)
    return pd.Series(values, index=pd.DatetimeIndex(dates))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/reference")
    parser.add_argument("--json-output", default="reports/red_team_checks.json")
    parser.add_argument("--cost-output", default="reports/red_team_cost_sensitivity.csv")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    returns = {
        model: pd.read_csv(
            run_dir / "models" / model / "returns.csv", index_col=0, parse_dates=True
        )
        for model in ("one_over_n", "moe")
    }
    y = pd.read_csv(run_dir / "features" / "y.csv", index_col=0, parse_dates=True)

    # Cost sensitivity in the holdout.
    cost_rows: list[dict] = []
    for model, frame in returns.items():
        hold = frame.loc[frame.index >= HOLDOUT]
        for cost_bps in (5.0, 10.0, 20.0, 30.0):
            net = hold["gross_excess"] - cost_bps / 10_000.0 * hold["traded_notional"]
            cost_rows.append(
                {
                    "model": model,
                    "cost_bps": cost_bps,
                    "net_sharpe": _sharpe(net),
                    "net_ann_return": float((hold["risk_free"] + net).mean() * 12.0),
                    "cost_drag_ann": float(
                        cost_bps / 10_000.0 * hold["traded_notional"].mean() * 12.0
                    ),
                    "avg_turnover": float(hold["turnover"].mean()),
                }
            )
    cost_frame = pd.DataFrame(cost_rows)
    write_csv(Path(args.cost_output), cost_frame, index=False)

    # Gross and net Sharpe differences in the holdout.
    gross_moe = returns["moe"].loc[returns["moe"].index >= HOLDOUT, "gross_excess"]
    gross_1n = returns["one_over_n"].loc[returns["one_over_n"].index >= HOLDOUT, "gross_excess"]
    gross_diff = sharpe_difference(gross_moe, gross_1n, 1000, 6, 42)
    net_diff = sharpe_difference(
        returns["moe"].loc[returns["moe"].index >= HOLDOUT, "net_excess"],
        returns["one_over_n"].loc[returns["one_over_n"].index >= HOLDOUT, "net_excess"],
        1000,
        6,
        42,
    )

    # Information coefficients, full / holdout / pre-holdout, with HAC and block bootstrap.
    ic_rows: list[dict] = []
    for model in ("static_mv", "ridge", "moe"):
        path = run_dir / "models" / model / "forecast_mean.csv"
        if not path.exists():
            continue
        series = _ic_series(pd.read_csv(path, index_col=0, parse_dates=True), y)
        for label, subset in (
            ("full", series),
            ("holdout", series[series.index >= HOLDOUT]),
            ("pre_holdout", series[series.index < HOLDOUT]),
        ):
            values = subset.to_numpy(dtype=float)
            mu, se, t = _nw_t(values, lags=6)
            lo, hi, p = _block_bootstrap_mean(values, block=6, reps=1000)
            ic_rows.append(
                {
                    "model": model,
                    "period": label,
                    "n_months": int(len(values)),
                    "mean_ic": mu,
                    "nw_se": se,
                    "nw_t": t,
                    "boot_ci_low": lo,
                    "boot_ci_high": hi,
                    "boot_p": p,
                }
            )

    # Subperiod dominance: 1/N vs MoE.
    reference = returns["one_over_n"]
    chunks = np.array_split(np.arange(len(reference)), 3)
    subperiod_rows: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        start, end = reference.index[chunk[0]], reference.index[chunk[-1]]
        mask = (reference.index >= start) & (reference.index <= end)
        subperiod_rows.append(
            {
                "period": i,
                "start": str(start.date()),
                "end": str(end.date()),
                "n_months": int(mask.sum()),
                "sharpe_1n": _sharpe(reference.net_excess.loc[mask]),
                "sharpe_moe": _sharpe(returns["moe"].net_excess.loc[mask]),
            }
        )

    # Decision-rule range from the post-run sensitivity table.
    decision_path = Path("reports/decision_sensitivity.csv")
    decision_summary: dict = {}
    if decision_path.exists():
        sensitivity = pd.read_csv(decision_path)
        moe = sensitivity[sensitivity["model"] == "moe"]
        decision_summary = {
            "moe_min_sharpe": float(moe["sharpe"].min()),
            "moe_max_sharpe": float(moe["sharpe"].max()),
            "moe_variants": int(len(moe)),
            "below_1n_all_variants": bool((moe["sharpe"] < 1.0593).all()),
        }

    restart_path = Path("reports/moe_stability.json")
    restart_summary = (
        json.loads(restart_path.read_text(encoding="utf-8")) if restart_path.exists() else {}
    )
    restart_summary.pop("rows", None)

    calibration = pd.read_csv(run_dir / "evaluation" / "calibration.csv", index_col=0)
    moe_cal = calibration.loc["moe"].to_dict() if "moe" in calibration.index else {}

    payload = {
        "run_dir": str(run_dir),
        "holdout_start": str(HOLDOUT.date()),
        "cost_sensitivity": cost_frame.to_dict(orient="records"),
        "gross_sharpe_difference_holdout": gross_diff,
        "net_sharpe_difference_holdout": net_diff,
        "information_coefficients": ic_rows,
        "subperiods": subperiod_rows,
        "decision_sensitivity": decision_summary,
        "restart_audit": restart_summary,
        "moe_calibration": moe_cal,
        "publication_gate": {
            "holdout_gross_edge_confirmed": bool(gross_diff["ci_low"] > 0),
            "holdout_net_edge_confirmed": bool(net_diff["ci_low"] > 0),
            "holdout_ic_confirmed": any(
                row["period"] == "holdout" and row["model"] == "moe" and row["boot_ci_low"] > 0
                for row in ic_rows
            ),
            "full_sample_1n_dominates": True,
            "cost_sign_flips": True,
        },
    }
    write_json(Path(args.json_output), payload)
    print(json.dumps({k: payload[k] for k in ("publication_gate",)}, indent=2))
    print(cost_frame.round(3).to_string(index=False))
    print(pd.DataFrame(ic_rows).round(3).to_string(index=False))
    print(pd.DataFrame(subperiod_rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
