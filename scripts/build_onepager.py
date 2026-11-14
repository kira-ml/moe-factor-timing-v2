"""Build the one-page V2 research summary.

Reads the reference run and the red-team checks, generates the charts, writes a
print-ready HTML page, renders it with headless Chrome, and converts the PDF to
a high-resolution PNG.

Usage:
    python scripts/build_onepager.py
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymupdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "reference"
OUT = ROOT / "publication"
ASSETS = OUT / "assets"
FONT_DIR = OUT / "fonts"


def _chrome_path() -> Path:
    configured = os.environ.get("CHROME_PATH")
    if configured:
        return Path(configured)
    for name in ("chrome", "msedge", "chromium"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


CHROME = _chrome_path()

INK = "#0B1F33"
SLATE = "#3E4C59"
MUTED = "#6B7785"
RULE = "#D5CFC3"
PAPER = "#F7F5F0"
TEAL = "#173B57"
RUST = "#7A2633"
GOLD = "#A3854F"

from matplotlib import font_manager  # noqa: E402

font_manager.fontManager.addfont(str(FONT_DIR / "IBMPlexSans-Variable.ttf"))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.edgecolor": RULE,
        "axes.labelcolor": SLATE,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "text.color": INK,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def _png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _font_face(family: str, filename: str, weight: str = "400") -> str:
    path = FONT_DIR / filename
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f"@font-face {{ font-family: '{family}'; "
        f"src: url(data:font/ttf;base64,{payload}) format('truetype'); "
        f"font-weight: {weight}; font-style: normal; }}"
    )


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def chart_cost(checks: dict, path: Path) -> None:
    rows = pd.DataFrame(checks["cost_sensitivity"])
    one = rows[rows["model"] == "one_over_n"].sort_values("cost_bps")
    moe = rows[rows["model"] == "moe"].sort_values("cost_bps")
    fig, ax = plt.subplots(figsize=(4.3, 2.25))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.axvspan(5, 10, color=GOLD, alpha=0.10, linewidth=0)
    ax.plot(
        one["cost_bps"],
        one["net_sharpe"],
        color=TEAL,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        label="1/N",
    )
    ax.plot(
        moe["cost_bps"],
        moe["net_sharpe"],
        color=RUST,
        marker="s",
        markersize=4.2,
        linewidth=1.8,
        label="MoE",
    )
    ax.axvline(10, color=INK, linestyle=(0, (2, 2)), linewidth=1.0)
    ax.text(10.4, 0.585, "pre-committed\n10 bps", fontsize=6.6, color=INK, va="top")
    ax.annotate(
        "sign flips\n5 → 10 bps",
        xy=(7.5, 0.515),
        xytext=(5.8, 0.565),
        fontsize=6.4,
        color=GOLD,
        arrowprops=dict(arrowstyle="-", color=GOLD, linewidth=0.8),
    )
    ax.text(31.0, one["net_sharpe"].iloc[-1], " 1/N", color=TEAL, fontsize=7.4, va="center")
    ax.text(31.0, moe["net_sharpe"].iloc[-1], " MoE", color=RUST, fontsize=7.4, va="center")
    ax.set_xlim(3.5, 34)
    ax.set_ylim(0.14, 0.61)
    ax.set_xticks([5, 10, 20, 30])
    ax.set_xlabel("One-way cost assumption (bps)", fontsize=7.2)
    ax.set_ylabel("Holdout net Sharpe", fontsize=7.2)
    ax.tick_params(labelsize=6.8, length=2)
    _clean_axes(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def chart_ic(checks: dict, path: Path) -> None:
    rows = pd.DataFrame(checks["information_coefficients"])
    full = rows[(rows["model"] == "moe") & (rows["period"] == "full")].iloc[0]
    hold = rows[(rows["model"] == "moe") & (rows["period"] == "holdout")].iloc[0]
    labels = ["Full sample\n627 months", "Holdout\n248 months"]
    means = [full["mean_ic"], hold["mean_ic"]]
    lower = [means[0] - full["boot_ci_low"], means[1] - hold["boot_ci_low"]]
    upper = [full["boot_ci_high"] - means[0], hold["boot_ci_high"] - means[1]]
    fig, ax = plt.subplots(figsize=(2.85, 1.45))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.bar([0, 1], means, width=0.48, color=[TEAL, RUST], alpha=0.92)
    ax.errorbar(
        [0, 1],
        means,
        yerr=[lower, upper],
        fmt="none",
        ecolor=INK,
        elinewidth=0.9,
        capsize=3,
        capthick=0.9,
    )
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylim(-0.06, 0.13)
    ax.set_ylabel("Mean cross-sectional IC", fontsize=6.8)
    ax.tick_params(labelsize=6.4, length=2)
    ax.text(0, means[0] + upper[0] + 0.008, f"t = {full['nw_t']:.2f}", ha="center", fontsize=6.4)
    ax.text(1, means[1] + upper[1] + 0.008, f"t = {hold['nw_t']:.2f}", ha="center", fontsize=6.4)
    ax.text(1, -0.045, "not significant", ha="center", fontsize=6.0, color=RUST)
    ax.set_title("MoE information coefficient", fontsize=7.4, loc="left", color=INK, pad=4)
    _clean_axes(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def chart_pit(path: Path) -> None:
    pit = pd.read_csv(RUN / "models" / "moe" / "pit.csv", index_col=0).to_numpy().ravel()
    fig, ax = plt.subplots(figsize=(2.85, 1.15))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    counts, edges = np.histogram(pit, bins=10, range=(0.0, 1.0))
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color=RUST,
        alpha=0.82,
        edgecolor=PAPER,
        linewidth=0.6,
    )
    ax.axhline(len(pit) / 10.0, color=INK, linestyle=(0, (2, 2)), linewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=6.0, length=2)
    ax.set_ylim(0, counts.max() * 1.28)
    ax.set_yticks([])
    ax.set_xlabel("PIT value", fontsize=6.4)
    ax.set_title(
        "MoE PIT histogram: 0/6 assets pass KS", fontsize=7.4, loc="left", color=INK, pad=4
    )
    _clean_axes(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def chart_subperiods(path: Path) -> None:
    sub = pd.read_csv(RUN / "evaluation" / "subperiods.csv")
    fig, ax = plt.subplots(figsize=(2.7, 1.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    x = np.arange(3)
    one = sub[sub["model"] == "one_over_n"].sort_values("period")["sharpe"].to_numpy()
    moe = sub[sub["model"] == "moe"].sort_values("period")["sharpe"].to_numpy()
    ax.plot(x, one, color=TEAL, marker="o", markersize=4.5, linewidth=1.6, label="1/N")
    ax.plot(x, moe, color=RUST, marker="s", markersize=4.2, linewidth=1.6, label="MoE")
    for i, value in enumerate(one):
        ax.text(i, value + 0.055, f"{value:.2f}", ha="center", fontsize=6.2, color=TEAL)
    for i, value in enumerate(moe):
        ax.text(i, value - 0.075, f"{value:.2f}", ha="center", fontsize=6.2, color=RUST)
    ax.set_xticks(x)
    ax.set_xticklabels(["1974–91", "1991–2009", "2009–26"], fontsize=6.6)
    ax.set_ylim(0.22, 1.58)
    ax.set_ylabel("Net Sharpe", fontsize=6.8)
    ax.tick_params(labelsize=6.4, length=2)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", ncol=2, handlelength=1.2)
    ax.set_title("1/N wins 3/3 subperiods", fontsize=7.4, loc="left", color=INK, pad=4)
    _clean_axes(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def build_html(numbers: dict, assets: dict[str, str], font_css: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>A Small Signal Is Not a Strategy</title>
<style>
{font_css}
  @page {{ size: 210mm 297mm; margin: 0; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: {PAPER}; }}
  body {{ width: 210mm; height: 297mm; overflow: hidden; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  .page {{
    width: 210mm; height: 297mm; background: {PAPER}; color: {INK};
    padding: 8mm 11mm 5mm 11mm; position: relative;
    font-family: "Plex Sans", Arial, sans-serif; font-size: 8px; line-height: 1.38;
  }}
  .topbar {{ height: 2.2mm; background: {TEAL}; margin-bottom: 4.5mm; }}
  .masthead {{ display: flex; justify-content: space-between; align-items: baseline; }}
  .kicker {{ font-size: 6.6pt; letter-spacing: 1.6pt; text-transform: uppercase; color: {TEAL}; font-weight: 600; }}
  .date {{ font-size: 6.4pt; letter-spacing: 1.2pt; text-transform: uppercase; color: {MUTED}; }}
  h1 {{
    font-family: "SS4", Georgia, serif; font-weight: 400;
    font-size: 29.5pt; line-height: 1.0; letter-spacing: -0.3pt;
    margin: 1.8mm 0 1.2mm 0; max-width: 176mm;
  }}
  .deck {{ font-size: 8.4pt; color: {SLATE}; max-width: 168mm; line-height: 1.35; }}
  .byline {{ font-size: 6.8pt; color: {SLATE}; letter-spacing: 0.4pt; margin: 1.0mm 0 0 0; }}
  .rule {{ border-top: 1.1px solid {INK}; margin: 3.2mm 0 3.2mm 0; }}
  .intro {{ display: grid; grid-template-columns: 1.62fr 1fr; gap: 5.5mm; }}
  .lead {{ font-size: 8.2pt; line-height: 1.4; margin: 0 0 2.1mm 0; color: {INK}; }}
  .abstract p {{ margin: 0 0 1.3mm 0; text-align: justify; }}
  .abstract {{ font-size: 7.1pt; line-height: 1.42; color: {SLATE}; }}
  .pull {{ border-left: 2.2px solid {GOLD}; padding-left: 3mm; margin-top: 2.6mm; font-family: "SS4", Georgia, serif; font-size: 8.2pt; line-height: 1.3; color: {INK}; }}
  .glance {{ background: #EFE9DF; border-left: 2.4px solid {TEAL}; padding: 3.0mm 3.2mm 2.0mm 3.2mm; }}
  .glance-title {{ font-size: 6.3pt; letter-spacing: 1.4pt; text-transform: uppercase; color: {TEAL}; font-weight: 700; margin-bottom: 2.2mm; }}
  .stat {{ margin-bottom: 2.0mm; }}
  .stat .num {{ font-family: "Plex Mono", "Courier New", monospace; font-size: 13.0pt; font-weight: 500; letter-spacing: -0.4pt; display: block; line-height: 1.0; }}
  .stat .label {{ display: block; font-size: 6.1pt; color: {SLATE}; line-height: 1.25; margin-top: 0.7mm; }}
  .evidence {{ display: grid; grid-template-columns: 1.44fr 1fr; gap: 6mm; margin-top: 3.4mm; }}
  h2 {{ font-size: 7.3pt; letter-spacing: 0.7pt; text-transform: uppercase; color: {TEAL}; margin: 0 0 1.6mm 0; font-weight: 700; }}
  h3 {{ font-size: 7.3pt; letter-spacing: 0.35pt; color: {INK}; margin: 0 0 1.2mm 0; font-weight: 600; }}
  .chart-main img {{ width: 100%; display: block; }}
  .chart-side {{ display: flex; flex-direction: column; gap: 2.6mm; }}
  .mini img {{ width: 100%; display: block; }}
  .caption {{ font-size: 5.9pt; line-height: 1.3; color: {MUTED}; margin: 1.4mm 0 0 0; }}
  .bottom {{ display: grid; grid-template-columns: 0.95fr 1.0fr 1.15fr; gap: 6mm; margin-top: 3.4mm; }}
  .bottom h2 {{ margin-bottom: 1.4mm; }}
  ul {{ margin: 0; padding: 0 0 0 3.2mm; }}
  li {{ font-size: 6.2pt; line-height: 1.3; margin-bottom: 1.0mm; color: {SLATE}; }}
  li::marker {{ color: {TEAL}; }}
  .not li::marker {{ color: {RUST}; }}
  .bottom img {{ width: 100%; display: block; }}
  .falsification {{
    margin-top: 3.4mm; padding: 2.0mm 0; border-top: 0.8px solid {RULE};
    border-bottom: 0.8px solid {RULE}; display: flex; gap: 4mm; align-items: baseline;
  }}
  .falsification-label {{ font-size: 6.0pt; letter-spacing: 1.2pt; text-transform: uppercase; color: {TEAL}; font-weight: 700; white-space: nowrap; }}
  .falsification-text {{ font-size: 6.4pt; color: {SLATE}; line-height: 1.3; }}
  .method {{
    margin-top: 7.5mm; border-top: 1px solid {RULE}; padding-top: 2.2mm;
    display: flex; justify-content: space-between; gap: 8mm; align-items: flex-start;
  }}
  .method-left {{ font-size: 5.7pt; color: {SLATE}; line-height: 1.25; max-width: 128mm; }}
  .method-right {{ font-size: 5.7pt; color: {MUTED}; text-align: right; line-height: 1.25; }}
  .mono {{ font-family: "Plex Mono", "Courier New", monospace; font-size: 5.7pt; color: {INK}; }}
  .footer-rule {{ position: absolute; bottom: 6.0mm; left: 11mm; right: 11mm; border-top: 0.8px solid {RULE}; }}
  .footer {{ position: absolute; bottom: 3.0mm; left: 11mm; right: 11mm; display: flex; justify-content: space-between; align-items: flex-start; gap: 5mm; font-size: 5.2pt; line-height: 1.25; color: {MUTED}; }}
  .footer-right {{ white-space: nowrap; flex-shrink: 0; }}
  .footer a {{ color: inherit; text-decoration: none; border-bottom: 0.5px solid {RULE}; }}
</style>
</head>
<body>
<div class="page">
  <div class="topbar"></div>
  <div class="masthead">
    <div class="kicker">Quantitative Research Note &nbsp;·&nbsp; Factor Allocation</div>
    <div class="date">November 2026</div>
  </div>
  <h1>A Small Signal<br>Is Not a Strategy</h1>
  <div class="deck">A probabilistic mixture-of-experts model over 63 years of US equity factor returns</div>
  <div class="byline">Ken Ira Lacson Talingting</div>
  <div class="rule"></div>

  <section class="intro">
    <div class="abstract">
      <p class="lead">Probabilistic regime models are often judged by predictive accuracy. This note asks a stricter question: can a mixture of experts turn a small cross-sectional signal into net factor-allocation value?</p>
      <p>Over {numbers["n_months"]} monthly observations from 1963 to 2026, a three-expert mixture of linear Gaussian experts produces a positive full-sample information coefficient of {numbers["ic_full"]:.3f} (HAC t = {numbers["ic_full_t"]:.2f}). In the pre-committed 2006–2026 holdout, the IC falls to {numbers["ic_hold"]:.3f} (t = {numbers["ic_hold_t"]:.2f}), the net Sharpe difference versus equal weight is {numbers["net_diff"]:.2f} (95% CI [{numbers["net_ci_low"]:.2f}, {numbers["net_ci_high"]:.2f}]), and the sign of the net difference flips between 5 and 10 bps costs.</p>
      <p>Equal weight remains ahead over the full sample and in all three subperiods. The contribution is a benchmark and a caution: <strong>predictive signal, gross Sharpe, and net decision value are three different objects.</strong></p>
      <div class="pull">A full-sample signal is not a holdout result. A gross Sharpe is not a strategy.</div>
    </div>
    <aside class="glance">
      <div class="glance-title">At a glance</div>
      <div class="stat"><span class="num">1.06 / 0.79</span><span class="label">Full-sample net Sharpe<br>1/N vs MoE</span></div>
      <div class="stat"><span class="num">−0.03</span><span class="label">Holdout net Sharpe difference<br>95% CI [−0.47, 0.39]</span></div>
      <div class="stat"><span class="num">0.55 → 0.48 → 0.34</span><span class="label">MoE holdout net Sharpe<br>at 5 / 10 / 20 bps</span></div>
      <div class="stat" style="margin-bottom:0"><span class="num">40.6%</span><span class="label">Monthly turnover · 0.97% annual cost drag at 10 bps</span></div>
    </aside>
  </section>

  <section class="evidence">
    <div class="chart-main">
      <h2>Holdout net Sharpe is cost-sensitive</h2>
      <img src="{assets["cost"]}" alt="Cost sensitivity chart">
      <p class="caption">The MoE is ahead at 5 bps, behind at 10 bps, and clearly behind at 20 bps. The sign flip is within noise; the pre-committed cost is 10 bps. A 5 bps result must not be read as an edge.</p>
    </div>
    <div class="chart-side">
      <div class="mini">
        <h3>Full-sample IC does not survive the holdout</h3>
        <img src="{assets["ic"]}" alt="Information coefficient chart">
      </div>
      <div class="mini">
        <h3>The predictive distribution is overconfident</h3>
        <img src="{assets["pit"]}" alt="PIT histogram">
        <p class="caption">10.9% of PIT values in each 5% tail; a calibrated model would be uniform.</p>
      </div>
    </div>
  </section>

  <section class="bottom">
    <div>
      <h2>What the evidence supports</h2>
      <ul>
        <li>1/N beat the MoE over the full 1974–2026 window on net (1.06 vs 0.79) and gross (1.07 vs 0.93) Sharpe, and in all three subperiods.</li>
        <li>The MoE trades 40.6% of notional per month; at 10 bps, cost drag is 0.97% per year.</li>
        <li>The distribution is miscalibrated: 10.9% of PIT values in each 5% tail; 0/6 assets pass KS.</li>
        <li>The holdout is inconclusive, not equivalent: −0.03 net Sharpe, 95% CI [−0.47, 0.39]; MDE 0.62.</li>
      </ul>
    </div>
    <div class="not">
      <h2>What it does not support</h2>
      <ul>
        <li>A holdout gross edge: 0.62 vs 0.51, but the 95% CI [−0.32, 0.51] includes zero.</li>
        <li>A holdout IC: 0.040, HAC t = 1.35; full-sample significance is not holdout confirmation.</li>
        <li>A 5 bps net edge: the sign flips at 10 bps and the gross difference is within noise.</li>
        <li>A precise MoE point estimate: one initialization; the restart audit shows local-optimum sensitivity.</li>
        <li>Live performance, capacity, or generalization beyond US equity factors at monthly frequency.</li>
      </ul>
    </div>
    <div>
      <h2>Equal weight wins every subperiod</h2>
      <img src="{assets["subperiod"]}" alt="Subperiod Sharpe chart">
    </div>
  </section>

  <div class="falsification">
    <span class="falsification-label">Falsification criterion</span>
    <span class="falsification-text">A holdout IC with a block-bootstrap CI above zero, and a net Sharpe that survives 20 bps. Neither is present.</span>
  </div>

  <div class="method">
    <div class="method-left">
      <strong>Method.</strong> {numbers["n_months"]} monthly observations · {numbers["n_oos"]} out-of-sample decisions · {numbers["n_holdout"]}-month holdout · 30 causal features · 3 experts · 1/N benchmark · 10 bps base one-way cost · drift-aware turnover.<br>
      <span class="mono">Evidence: 1,000 bootstrap draws · 6-lag HAC · PIT/KS · MDE 0.62 · 45 tests · red-team gate.</span><br>
      <span class="mono">Config 2a035de · data SHA-256 cc805ab0 · archive 716ff99a</span>
    </div>
    <div class="method-right">
      Data: Ken French Data Library<br>
      Fama–French five factors + momentum
    </div>
  </div>

  <div class="footer-rule"></div>
  <div class="footer">
    <span>Retrospective panel; reconstructed factors, not vintage. Sleeve turnover not modeled. Not investment advice.</span>
    <span class="footer-right"><a href="https://github.com/kira-ml/moe-factor-timing-v2">Open-source code &amp; report: github.com/kira-ml/moe-factor-timing-v2</a></span>
  </div>
</div>
</body>
</html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    checks = json.loads((ROOT / "reports" / "red_team_checks.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RUN / "evaluation" / "metrics.csv", index_col=0)
    holdout = pd.read_csv(RUN / "evaluation" / "holdout_metrics.csv", index_col=0)
    net = checks["net_sharpe_difference_holdout"]
    ic_full = next(
        row
        for row in checks["information_coefficients"]
        if row["model"] == "moe" and row["period"] == "full"
    )
    ic_hold = next(
        row
        for row in checks["information_coefficients"]
        if row["model"] == "moe" and row["period"] == "holdout"
    )
    numbers = {
        "n_months": 759,
        "n_oos": 627,
        "n_holdout": 248,
        "net_diff": net["diff"],
        "net_ci_low": net["ci_low"],
        "net_ci_high": net["ci_high"],
        "ic_full": ic_full["mean_ic"],
        "ic_full_t": ic_full["nw_t"],
        "ic_hold": ic_hold["mean_ic"],
        "ic_hold_t": ic_hold["nw_t"],
        "full_1n": metrics.loc["one_over_n", "sharpe"],
        "full_moe": metrics.loc["moe", "sharpe"],
        "hold_1n": holdout.loc["one_over_n", "sharpe"],
        "hold_moe": holdout.loc["moe", "sharpe"],
    }

    chart_cost(checks, ASSETS / "cost.png")
    chart_ic(checks, ASSETS / "ic.png")
    chart_pit(ASSETS / "pit.png")
    chart_subperiods(ASSETS / "subperiod.png")
    assets = {
        "cost": _png(ASSETS / "cost.png"),
        "ic": _png(ASSETS / "ic.png"),
        "pit": _png(ASSETS / "pit.png"),
        "subperiod": _png(ASSETS / "subperiod.png"),
    }
    font_css = "\n".join(
        [
            _font_face("SS4", "SourceSerif4-Variable.ttf", "100 900"),
            _font_face("Plex Sans", "IBMPlexSans-Variable.ttf", "100 900"),
            _font_face("Plex Mono", "IBMPlexMono-Regular.ttf", "400"),
        ]
    )
    html = build_html(numbers, assets, font_css)
    html_path = OUT / "one_pager.html"
    html_path.write_text(html, encoding="utf-8")

    pdf_path = OUT / "one_pager.pdf"
    with tempfile.TemporaryDirectory(prefix="moe_chrome_") as profile:
        chrome_cmd = [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            f"--user-data-dir={profile}",
            "--virtual-time-budget=3000",
            html_path.as_uri(),
        ]
        subprocess.run(chrome_cmd, check=True, capture_output=True, text=True)

    doc = pymupdf.open(pdf_path)
    if doc.page_count != 1:
        raise RuntimeError(f"expected 1 page, got {doc.page_count}")
    page = doc[0]
    pix = page.get_pixmap(dpi=200)
    png_path = OUT / "one_pager.png"
    pix.save(png_path)
    doc.set_metadata(
        {
            "title": "A Small Signal Is Not a Strategy",
            "subject": "Mixture-of-experts factor timing over 63 years of US equity factor returns",
            "author": "Ken Ira Lacson Talingting",
            "keywords": "factor timing, mixture of experts, regime models, transaction costs, reproducible research",
            "creator": "moe-factor-timing v2",
        }
    )
    doc.saveIncr()
    doc.close()
    print(f"wrote {html_path}")
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size} bytes)")
    print(f"wrote {png_path} ({pix.width}x{pix.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
