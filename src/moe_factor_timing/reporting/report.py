"""Markdown research report generation.

The report is generated from the same objects that produced the metrics. It
does not contain hand-written numbers, which prevents the report from drifting
away from the run artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..backtest.engine import BacktestSuite
from ..config import Config
from ..evaluation.evaluate import EvaluationResult
from ..utils.io import ensure_dir, write_text
from ..utils.paths import project_root


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.name


def _pct(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100.0:.{digits}f}%"


def _num(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for index, row in frame.iterrows():
        values = [str(index)] + [row[column] for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _performance_table(metrics: pd.DataFrame) -> str:
    table = pd.DataFrame(index=metrics.index)
    table["Sharpe"] = metrics["sharpe"].map(lambda x: _num(x, 3))
    table["Gross SR"] = metrics["gross_sharpe"].map(lambda x: _num(x, 3))
    table["Ann. return"] = metrics["ann_return"].map(_pct)
    table["Ann. vol"] = metrics["ann_vol"].map(_pct)
    table["Max DD"] = metrics["max_drawdown"].map(_pct)
    table["Utility"] = metrics["utility"].map(lambda x: _num(x, 5))
    table["Turnover"] = metrics["avg_turnover"].map(lambda x: _num(x, 3))
    table["Cost (bps/mo)"] = metrics["avg_cost_bps"].map(lambda x: _num(x, 2))
    table["Deployed"] = metrics["avg_deployed"].map(lambda x: _num(x, 2))
    return _table(
        table,
        list(table.columns),
        ["Model"] + list(table.columns),
    )


def _bootstrap_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Not available._"
    table = pd.DataFrame(index=frame.index)
    table["Sharpe"] = frame["sharpe_model"].map(lambda x: _num(x, 3))
    table["1/N Sharpe"] = frame["sharpe_benchmark"].map(lambda x: _num(x, 3))
    table["Difference"] = frame["diff"].map(lambda x: _num(x, 3))
    table["95% CI"] = frame.apply(
        lambda row: f"[{_num(row['ci_low'], 2)}, {_num(row['ci_high'], 2)}]", axis=1
    )
    table["p"] = frame["p_value"].map(lambda x: _num(x, 3))
    return _table(table, list(table.columns), ["Model"] + list(table.columns))


def _calibration_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No probabilistic models in this run._"
    table = pd.DataFrame(index=frame.index)
    table["Mean NLL"] = frame["mean_nll"].map(lambda x: _num(x, 3))
    table["Mean PIT"] = frame["mean_pit"].map(lambda x: _num(x, 3))
    table["PIT < 5%"] = frame["pit_below_5pct"].map(lambda x: _pct(x, 1))
    table["PIT > 95%"] = frame["pit_above_95pct"].map(lambda x: _pct(x, 1))
    table["KS p (pooled)"] = frame["ks_pooled_p"].map(lambda x: _num(x, 4))
    table["Assets passing"] = frame.apply(
        lambda row: f"{int(row['ks_assets_passing_5pct'])}/{int(row['n_assets'])}", axis=1
    )
    return _table(table, list(table.columns), ["Model"] + list(table.columns))


def _dsr_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Not available._"
    table = pd.DataFrame(index=frame.index)
    table["PSR"] = frame["psr"].map(lambda x: _num(x, 3))
    table["DSR"] = frame["dsr"].map(lambda x: _num(x, 3))
    table["DSR p"] = frame["dsr_p_value"].map(lambda x: _num(x, 3))
    table["E[max SR]"] = frame["expected_max_sharpe"].map(lambda x: _num(x, 3))
    table["N trials"] = frame["n_trials"].map(lambda x: str(int(x)))
    return _table(table, list(table.columns), ["Model"] + list(table.columns))


def _cost_sensitivity_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Not available._"
    pivot = frame.pivot(index="model", columns="cost_bps", values="sharpe")
    pivot.columns = [f"{float(column):.0f} bps" for column in pivot.columns]
    formatted = pivot.map(lambda x: _num(x, 3))
    return _table(formatted, list(formatted.columns), ["Model"] + list(formatted.columns))


def _verdict(
    evaluation: EvaluationResult, challenger: str = "moe", benchmark: str = "one_over_n"
) -> str:
    if challenger not in evaluation.metrics.index:
        return "The challenger model is not present in this run."
    if evaluation.bootstrap_holdout.empty or challenger not in evaluation.bootstrap_holdout.index:
        return "No holdout comparison is available."
    row = evaluation.bootstrap_holdout.loc[challenger]
    metric = evaluation.holdout_metrics.loc[challenger]
    bench = evaluation.holdout_metrics.loc[benchmark]
    text = (
        f"In the pre-committed holdout ({evaluation.meta['holdout_start']} to "
        f"{evaluation.meta['last_decision']}), **{challenger}** has a net Sharpe of "
        f"{_num(metric['sharpe'], 3)} versus **{benchmark}** at {_num(bench['sharpe'], 3)}. "
        f"The Sharpe difference is {_num(row['diff'], 3)} with a block-bootstrap 95% "
        f"confidence interval [{_num(row['ci_low'], 3)}, {_num(row['ci_high'], 3)}] "
        f"(p = {_num(row['p_value'], 3)})."
    )
    if row["ci_low"] > 0:
        verdict = "The primary criterion is met: the challenger beats 1/N in the holdout."
    elif row["ci_high"] < 0:
        verdict = "The challenger underperforms 1/N in the holdout; the confidence interval is below zero."
    else:
        verdict = (
            "The confidence interval includes zero. The data do not distinguish the challenger "
            "from 1/N at the pre-committed confidence level. This is an inconclusive result, "
            "not evidence that the two are equivalent."
        )
    return f"{text}\n\n**Verdict.** {verdict}"


def generate_report(
    suite: BacktestSuite,
    evaluation: EvaluationResult,
    config: Config,
    run_id: str,
    run_dir: Path,
    report_path: Path,
) -> Path:
    metrics = evaluation.metrics
    holdout = evaluation.holdout_metrics
    benchmark = evaluation.meta["benchmark"]
    challenger = "moe" if "moe" in metrics.index else metrics.index[0]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data = suite.data_metadata
    features = suite.feature_manifest

    lines: list[str] = []
    lines.append("# moe-factor-timing v2 — Research Report\n")
    lines.append(
        f"**Run:** `{run_id}`  \n"
        f"**Generated:** {generated}  \n"
        f"**Config hash:** `{suite.config_hash}`  \n"
        f"**Benchmark:** `{benchmark}`  \n"
        f"**Challenger:** `{challenger}`\n"
    )
    lines.append("## 1. Executive summary\n")
    lines.append(_verdict(evaluation, challenger=challenger, benchmark=benchmark))
    lines.append(
        "\nThe benchmark is deliberately conservative: 1/N is a hard-to-beat baseline, "
        "the evaluation is chronological, the holdout is pre-committed, and costs and "
        "turnover are charged at every rebalance. The result should be read as evidence "
        "about this protocol, not as a claim about all regime-aware models.\n"
    )

    lines.append("## 2. Research question and pre-committed protocol\n")
    lines.append(
        "Can a probabilistic mixture-of-experts model that conditions factor allocations "
        "on observable state variables improve out-of-sample net risk-adjusted factor "
        "allocation relative to a naive equal-weight portfolio, after realistic costs?\n\n"
        "The primary confirmatory endpoint is the net Sharpe difference between the "
        "challenger and 1/N in the holdout. The secondary endpoints are mean-variance "
        "utility, drawdown, turnover, predictive calibration, and the deflated Sharpe "
        "ratio. Hyperparameters are fixed a priori; the holdout is not used for selection. "
        "See `docs/03_methodology.md` and `docs/04_evaluation_protocol.md`.\n"
    )

    lines.append("## 3. Data and design\n")
    lines.append(
        f"- **Universe:** {', '.join(data.get('columns', []))} (Ken French five factors plus momentum).\n"
        f"- **Sample:** {data.get('loaded_start', 'n/a')} to {data.get('loaded_end', 'n/a')} "
        f"({data.get('loaded_rows', 'n/a')} monthly observations).\n"
        f"- **Features:** {features.get('n_features', 'n/a')} lagged return and realised-volatility "
        f"features; target is the next-month excess return.\n"
        f"- **Walk-forward:** {evaluation.meta['n_splits']} out-of-sample monthly decisions; "
        f"minimum training window {config.split.min_train} months.\n"
        f"- **Holdout:** {evaluation.meta['holdout_start']} onward "
        f"({evaluation.meta['n_holdout']} months).\n"
        f"- **Decision rule:** constrained mean-variance, risk aversion {config.decision.risk_aversion}, "
        f"max weight {config.decision.constraints.max_weight}, fully invested="
        f"{config.decision.constraints.fully_invested}.\n"
        f"- **Costs:** {config.decision.cost.one_way_bps:.1f} bps one-way per unit traded; "
        f"strategy drag {config.decision.cost.strategy_drag_bps_per_month:.1f} bps/month.\n"
    )
    lines.append(
        "The factor returns are long-short research portfolios. The allocation is a "
        "fund-of-factor-strategies interpretation; internal turnover inside each factor "
        "portfolio is not modelled and is disclosed as a limitation.\n"
    )

    lines.append("## 4. Full out-of-sample results\n")
    lines.append(_performance_table(metrics))
    lines.append("")
    lines.append("## 5. Confirmatory holdout results\n")
    if holdout.empty:
        lines.append("_No holdout period was configured._")
    else:
        lines.append(_performance_table(holdout))
    lines.append("")
    lines.append("## 6. Relative performance versus 1/N\n")
    lines.append("### Full out-of-sample window\n")
    lines.append(_bootstrap_table(evaluation.bootstrap_full))
    lines.append("\n### Confirmatory holdout\n")
    lines.append(_bootstrap_table(evaluation.bootstrap_holdout))
    lines.append("")
    lines.append("## 7. Predictive calibration\n")
    lines.append(_calibration_table(evaluation.calibration))
    lines.append(
        "\nPIT is the marginal probability integral transform of the predictive "
        "distribution. For a calibrated model, PIT values are uniform. The KS test is a "
        "diagnostic; passing it does not establish decision value.\n"
    )
    lines.append("## 8. Multiple testing and deflated Sharpe\n")
    lines.append(_dsr_table(evaluation.deflated))
    lines.append(
        "\nThe deflated Sharpe ratio uses the dispersion of the trial Sharpe ratios and "
        "the number of trials recorded in the config. It is a selection-aware diagnostic, "
        "not a substitute for the pre-committed holdout.\n"
    )
    lines.append("## 9. Subperiod stability\n")
    if evaluation.subperiods.empty:
        lines.append("_Not available._")
    else:
        for period, frame in evaluation.subperiods.groupby("period"):
            start = frame["start"].iloc[0]
            end = frame["end"].iloc[0]
            lines.append(f"**Period {period}** ({start} to {end})\n")
            table = frame.set_index("model")[["sharpe", "ann_return", "max_drawdown"]].copy()
            table.columns = ["Sharpe", "Ann. return", "Max DD"]
            table["Sharpe"] = table["Sharpe"].map(lambda x: _num(x, 3))
            table["Ann. return"] = table["Ann. return"].map(_pct)
            table["Max DD"] = table["Max DD"].map(_pct)
            lines.append(_table(table, list(table.columns), ["Model"] + list(table.columns)))
            lines.append("")
    lines.append("## 10. Cost sensitivity\n")
    lines.append(
        "The table holds each model's decisions fixed and recomputes net Sharpe at "
        "alternative one-way cost assumptions. It isolates cost sensitivity from the "
        "optimizer's response to a different cost parameter.\n"
    )
    lines.append(_cost_sensitivity_table(evaluation.cost_sensitivity))
    lines.append("")
    lines.append("## 11. Robustness checks\n")
    lines.append(
        "A post-run MoE restart audit and a decision-rule sensitivity (risk aversion "
        "2/5/10 and position caps 0.25/0.35/0.50) are documented in "
        "`docs/09_sensitivity.md` and `reports/decision_sensitivity.csv`. In summary: "
        "additional restarts improve training likelihood but not reliably test NLL, and "
        "the MoE remains below 1/N in every decision-rule variant tested. Training-window "
        "and number-of-experts refits were not executed; this is disclosed as a limitation.\n"
    )
    lines.append("## 12. Power and interpretation\n")
    power = evaluation.power
    lines.append(
        f"The full window has {power['n_months_full']} months ({power['n_years_full']:.1f} years). "
        f"The approximate minimum detectable annualised Sharpe against zero is "
        f"{power['mde_full']:.2f}. The holdout has {power['n_months_holdout']} months "
        f"({power['n_years_holdout']:.1f} years), with a minimum detectable Sharpe of "
        f"{power['mde_holdout']:.2f}.\n\n"
        "These are conservative references for a single Sharpe. The paired bootstrap "
        "difference is the primary uncertainty statement because the strategies are "
        "correlated. A small but economically meaningful improvement can remain "
        "statistically indistinguishable from zero even with decades of monthly data.\n"
    )
    lines.append("## 13. Limitations and non-goals\n")
    lines.append(
        "- The Ken French panel is reconstructed from the current CRSP database, not a "
        "literal point-in-time vintage. It is a research benchmark, not a live trading record.\n"
        "- The factor strategies are long-short research portfolios. Internal factor-portfolio "
        "turnover and financing costs are not modelled; the cost sensitivity is reported separately.\n"
        "- The allocation layer is long-only and fully invested. Shorting, leverage, and "
        "market timing are out of scope.\n"
        "- The model is evaluated at monthly frequency only.\n"
        "- This is not investment advice, not a fund, and not a claim that the strategy will "
        "work live. Backtest results do not imply live performance.\n"
        "- The regime labels are descriptive, not causal. The project does not claim that "
        "the latent states correspond to identifiable economic regimes.\n"
    )
    lines.append("## 14. Reproducibility\n")
    lines.append(
        f"Run artifacts are in `{_display_path(run_dir)}`. The run manifest records SHA-256 "
        "hashes for every artifact, the config hash, the Python and package versions, and "
        "the git commit. Re-run with:\n\n"
        "```bash\n"
        f"python -m moe_factor_timing run --config {_display_path(config.source_path)}\n"
        "```\n\n"
        "The reference data is downloaded locally to `data/reference/` and is not "
        "redistributed with the public repository; `scripts/fetch_reference_data.py` "
        "documents how to fetch it.\n"
    )
    lines.append("## 15. Artifact index\n")
    lines.append(
        "- `backtest/splits.json` — walk-forward plan and holdout boundary.\n"
        "- `models/<model>/weights.csv` — monthly portfolio weights.\n"
        "- `models/<model>/returns.csv` — gross, net, cost, and risk-free returns.\n"
        "- `models/<model>/forecast_mean.csv` — predictive means.\n"
        "- `models/<model>/forecast_cov.npy` — predictive covariance arrays.\n"
        "- `models/<model>/pit.csv`, `nll.csv` — calibration diagnostics.\n"
        "- `evaluation/metrics.csv`, `holdout_metrics.csv`, `bootstrap_*.csv` - statistical results.\n"
        "- `evaluation/cost_sensitivity.csv` - Sharpe under alternative cost assumptions.\n"
        "- `report/figures/` - cumulative wealth, drawdowns, rolling Sharpe, weights, gating, PIT.\n"
    )

    output = ensure_dir(report_path.parent)
    path = output / report_path.name
    write_text(path, "\n".join(lines) + "\n")
    return path
