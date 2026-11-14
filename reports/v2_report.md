# moe-factor-timing v2 — Research Report

**Run:** `reference`  
**Generated:** 2026-11-14 08:10 UTC  
**Config hash:** `2a035de2d1a091455b816690492ae20a6bef2c1936ebffd0e51ed4f4c3daaa5f`  
**Benchmark:** `one_over_n`  
**Challenger:** `moe`

## 1. Executive summary

In the pre-committed holdout (2006-01-31 to 2026-08-31), **moe** has a net Sharpe of 0.479 versus **one_over_n** at 0.505. The Sharpe difference is -0.026 with a block-bootstrap 95% confidence interval [-0.470, 0.389] (p = 0.862).

**Verdict.** The confidence interval includes zero. The data do not distinguish the challenger from 1/N at the pre-committed confidence level. This is an inconclusive result, not evidence that the two are equivalent.

The benchmark is deliberately conservative: 1/N is a hard-to-beat baseline, the evaluation is chronological, the holdout is pre-committed, and costs and turnover are charged at every rebalance. The result should be read as evidence about this protocol, not as a claim about all regime-aware models.

## 2. Research question and pre-committed protocol

Can a probabilistic mixture-of-experts model that conditions factor allocations on observable state variables improve out-of-sample net risk-adjusted factor allocation relative to a naive equal-weight portfolio, after realistic costs?

The primary confirmatory endpoint is the net Sharpe difference between the challenger and 1/N in the holdout. The secondary endpoints are mean-variance utility, drawdown, turnover, predictive calibration, and the deflated Sharpe ratio. Hyperparameters are fixed a priori; the holdout is not used for selection. See `docs/03_methodology.md` and `docs/04_evaluation_protocol.md`.

## 3. Data and design

- **Universe:** MKT_RF, SMB, HML, RMW, CMA, MOM, RF (Ken French five factors plus momentum).
- **Sample:** 1963-07-31 to 2026-09-30 (759 monthly observations).
- **Features:** 30 lagged return and realised-volatility features; target is the next-month excess return.
- **Walk-forward:** 627 out-of-sample monthly decisions; minimum training window 120 months.
- **Holdout:** 2006-01-31 onward (248 months).
- **Decision rule:** constrained mean-variance, risk aversion 2.0, max weight 0.35, fully invested=True.
- **Costs:** 10.0 bps one-way per unit traded; strategy drag 0.0 bps/month.

The factor returns are long-short research portfolios. The allocation is a fund-of-factor-strategies interpretation; internal turnover inside each factor portfolio is not modelled and is disclosed as a limitation.

## 4. Full out-of-sample results

| Model | Sharpe | Gross SR | Ann. return | Ann. vol | Max DD | Utility | Turnover | Cost (bps/mo) | Deployed |
|---|---|---|---|---|---|---|---|---|---|
| one_over_n | 1.059 | 1.066 | 8.89% | 4.16% | -12.25% | 0.04237 | 0.012 | 0.23 | 1.00 |
| inverse_vol | 1.005 | 1.026 | 8.12% | 3.65% | -7.36% | 0.03538 | 0.032 | 0.65 | 1.00 |
| momentum | 0.766 | 0.799 | 10.52% | 8.02% | -17.17% | 0.05495 | 0.113 | 2.25 | 0.97 |
| cash | n/a | n/a | 4.31% | 0.00% | 0.00% | 0.00000 | 0.000 | 0.00 | 0.00 |
| static_mv | 0.874 | 0.878 | 9.81% | 6.12% | -25.60% | 0.04980 | 0.010 | 0.21 | 1.00 |
| ridge | 0.849 | 0.975 | 10.56% | 7.20% | -18.29% | 0.05597 | 0.379 | 7.57 | 1.00 |
| moe | 0.790 | 0.934 | 10.03% | 7.11% | -17.54% | 0.05115 | 0.422 | 8.45 | 1.00 |

## 5. Confirmatory holdout results

| Model | Sharpe | Gross SR | Ann. return | Ann. vol | Max DD | Utility | Turnover | Cost (bps/mo) | Deployed |
|---|---|---|---|---|---|---|---|---|---|
| one_over_n | 0.505 | 0.511 | 3.77% | 4.23% | -12.25% | 0.01956 | 0.011 | 0.22 | 1.00 |
| inverse_vol | 0.498 | 0.518 | 3.55% | 3.82% | -7.36% | 0.01755 | 0.032 | 0.64 | 1.00 |
| momentum | 0.468 | 0.511 | 4.83% | 7.08% | -17.17% | 0.02814 | 0.127 | 2.53 | 0.95 |
| cash | n/a | n/a | 1.68% | 0.00% | 0.00% | 0.00000 | 0.000 | 0.00 | 0.00 |
| static_mv | 0.509 | 0.513 | 4.78% | 6.32% | -25.60% | 0.02816 | 0.011 | 0.23 | 1.00 |
| ridge | 0.371 | 0.507 | 4.11% | 7.06% | -18.29% | 0.02119 | 0.404 | 8.08 | 1.00 |
| moe | 0.479 | 0.622 | 4.80% | 6.80% | -17.54% | 0.02794 | 0.406 | 8.11 | 1.00 |

## 6. Relative performance versus 1/N

### Full out-of-sample window

| Model | Sharpe | 1/N Sharpe | Difference | 95% CI | p |
|---|---|---|---|---|---|
| inverse_vol | 1.005 | 1.059 | -0.054 | [-0.21, 0.10] | 0.508 |
| momentum | 0.766 | 1.059 | -0.294 | [-0.53, -0.06] | 0.008 |
| cash | n/a | 1.059 | n/a | [n/a, n/a] | n/a |
| static_mv | 0.874 | 1.059 | -0.185 | [-0.39, 0.02] | 0.074 |
| ridge | 0.849 | 1.059 | -0.210 | [-0.46, 0.03] | 0.074 |
| moe | 0.790 | 1.059 | -0.269 | [-0.52, 0.01] | 0.058 |

### Confirmatory holdout

| Model | Sharpe | 1/N Sharpe | Difference | 95% CI | p |
|---|---|---|---|---|---|
| inverse_vol | 0.498 | 0.505 | -0.007 | [-0.18, 0.19] | 0.926 |
| momentum | 0.468 | 0.505 | -0.036 | [-0.38, 0.29] | 0.768 |
| cash | n/a | 0.505 | n/a | [n/a, n/a] | n/a |
| static_mv | 0.509 | 0.505 | 0.004 | [-0.25, 0.29] | 0.996 |
| ridge | 0.371 | 0.505 | -0.134 | [-0.54, 0.22] | 0.448 |
| moe | 0.479 | 0.505 | -0.026 | [-0.47, 0.39] | 0.862 |

## 7. Predictive calibration

| Model | Mean NLL | Mean PIT | PIT < 5% | PIT > 95% | KS p (pooled) | Assets passing |
|---|---|---|---|---|---|---|
| static_mv | -12.648 | 0.496 | 4.9% | 5.1% | 0.0001 | 2/6 |
| ridge | -12.079 | 0.498 | 6.3% | 6.1% | 0.1886 | 3/6 |
| moe | -8.156 | 0.497 | 10.9% | 10.9% | 0.0000 | 0/6 |

PIT is the marginal probability integral transform of the predictive distribution. For a calibrated model, PIT values are uniform. The KS test is a diagnostic; passing it does not establish decision value.

## 8. Multiple testing and deflated Sharpe

| Model | PSR | DSR | DSR p | E[max SR] | N trials |
|---|---|---|---|---|---|
| one_over_n | 1.000 | 1.000 | 0.000 | 0.192 | 7 |
| inverse_vol | 1.000 | 1.000 | 0.000 | 0.192 | 7 |
| momentum | 1.000 | 1.000 | 0.000 | 0.192 | 7 |
| cash | n/a | n/a | n/a | 0.192 | 7 |
| static_mv | 1.000 | 1.000 | 0.000 | 0.192 | 7 |
| ridge | 1.000 | 1.000 | 0.000 | 0.192 | 7 |
| moe | 1.000 | 1.000 | 0.000 | 0.192 | 7 |

The deflated Sharpe ratio uses the dispersion of the trial Sharpe ratios and the number of trials recorded in the config. It is a selection-aware diagnostic, not a substitute for the pre-committed holdout.

## 9. Subperiod stability

**Period 1** (1974-06-30 00:00:00 to 1991-10-31 00:00:00)

| Model | Sharpe | Ann. return | Max DD |
|---|---|---|---|
| one_over_n | 1.400 | 13.82% | -4.79% |
| inverse_vol | 1.555 | 12.65% | -2.31% |
| momentum | 0.981 | 16.17% | -9.46% |
| cash | n/a | 8.00% | 0.00% |
| static_mv | 1.089 | 14.47% | -5.49% |
| ridge | 1.244 | 16.77% | -12.00% |
| moe | 1.141 | 16.24% | -11.93% |

**Period 2** (1991-11-30 00:00:00 to 2009-03-31 00:00:00)

| Model | Sharpe | Ann. return | Max DD |
|---|---|---|---|
| one_over_n | 1.165 | 8.88% | -10.43% |
| inverse_vol | 1.166 | 8.57% | -6.06% |
| momentum | 0.741 | 10.55% | -15.94% |
| cash | n/a | 3.66% | 0.00% |
| static_mv | 0.751 | 8.79% | -23.14% |
| ridge | 0.909 | 11.23% | -13.01% |
| moe | 0.768 | 9.80% | -16.34% |

**Period 3** (2009-04-30 00:00:00 to 2026-08-31 00:00:00)

| Model | Sharpe | Ann. return | Max DD |
|---|---|---|---|
| one_over_n | 0.659 | 4.19% | -8.06% |
| inverse_vol | 0.504 | 3.35% | -7.36% |
| momentum | 0.575 | 5.11% | -11.23% |
| cash | n/a | 1.38% | 0.00% |
| static_mv | 0.822 | 6.32% | -9.57% |
| ridge | 0.419 | 4.05% | -18.29% |
| moe | 0.473 | 4.37% | -17.54% |

## 10. Cost sensitivity

The table holds each model's decisions fixed and recomputes net Sharpe at alternative one-way cost assumptions. It isolates cost sensitivity from the optimizer's response to a different cost parameter.

| Model | 5 bps | 10 bps | 20 bps | 30 bps |
|---|---|---|---|---|
| cash | n/a | n/a | n/a | n/a |
| inverse_vol | 1.016 | 1.005 | 0.984 | 0.963 |
| moe | 0.862 | 0.790 | 0.647 | 0.503 |
| momentum | 0.783 | 0.766 | 0.732 | 0.698 |
| one_over_n | 1.063 | 1.059 | 1.053 | 1.046 |
| ridge | 0.912 | 0.849 | 0.723 | 0.596 |
| static_mv | 0.876 | 0.874 | 0.870 | 0.866 |

## 11. Robustness checks

A post-run MoE restart audit and a decision-rule sensitivity (risk aversion 2/5/10 and position caps 0.25/0.35/0.50) are documented in `docs/09_sensitivity.md` and `reports/decision_sensitivity.csv`. In summary: additional restarts improve training likelihood but not reliably test NLL, and the MoE remains below 1/N in every decision-rule variant tested. Training-window and number-of-experts refits were not executed; this is disclosed as a limitation.

## 12. Power and interpretation

The full window has 627 months (52.2 years). The approximate minimum detectable annualised Sharpe against zero is 0.39. The holdout has 248 months (20.7 years), with a minimum detectable Sharpe of 0.62.

These are conservative references for a single Sharpe. The paired bootstrap difference is the primary uncertainty statement because the strategies are correlated. A small but economically meaningful improvement can remain statistically indistinguishable from zero even with decades of monthly data.

## 13. Limitations and non-goals

- The Ken French panel is reconstructed from the current CRSP database, not a literal point-in-time vintage. It is a research benchmark, not a live trading record.
- The factor strategies are long-short research portfolios. Internal factor-portfolio turnover and financing costs are not modelled; the cost sensitivity is reported separately.
- The allocation layer is long-only and fully invested. Shorting, leverage, and market timing are out of scope.
- The model is evaluated at monthly frequency only.
- This is not investment advice, not a fund, and not a claim that the strategy will work live. Backtest results do not imply live performance.
- The regime labels are descriptive, not causal. The project does not claim that the latent states correspond to identifiable economic regimes.

## 14. Reproducibility

Run artifacts are in `runs/reference`. The run manifest records SHA-256 hashes for every artifact, the config hash, the Python and package versions, and the git commit. Re-run with:

```bash
python -m moe_factor_timing run --config configs/default.yaml
```

The reference data is downloaded locally to `data/reference/` and is not redistributed with the public repository; `scripts/fetch_reference_data.py` documents how to fetch it.

## 15. Artifact index

- `backtest/splits.json` — walk-forward plan and holdout boundary.
- `models/<model>/weights.csv` — monthly portfolio weights.
- `models/<model>/returns.csv` — gross, net, cost, and risk-free returns.
- `models/<model>/forecast_mean.csv` — predictive means.
- `models/<model>/forecast_cov.npy` — predictive covariance arrays.
- `models/<model>/pit.csv`, `nll.csv` — calibration diagnostics.
- `evaluation/metrics.csv`, `holdout_metrics.csv`, `bootstrap_*.csv` - statistical results.
- `evaluation/cost_sensitivity.csv` - Sharpe under alternative cost assumptions.
- `report/figures/` - cumulative wealth, drawdowns, rolling Sharpe, weights, gating, PIT.

