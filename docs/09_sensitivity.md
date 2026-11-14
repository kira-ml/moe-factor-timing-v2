# 09 - Robustness and Sensitivity

**Status:** Post-run diagnostic
**Reference run:** `runs/reference`
**Primary conclusion:** unchanged by the checks below

This document records two robustness checks that were executed after the reference run. Neither is used to select a model or to rescue the primary endpoint. They are reported because a flagship result should show where it is fragile.

---

## 1. MoE restart stability

The reference run uses one deterministic k-means initialisation per walk-forward fit (`n_restarts = 1`). A separate audit refits the same model with three additional restarts on 24 evenly spaced training windows.

Source: `reports/moe_stability.json`.

| Statistic | Value |
|---|---:|
| Windows audited | 24 |
| Extra restarts | 3 |
| Mean training log-likelihood gain | +33.1 |
| Maximum training log-likelihood gain | +148.4 |
| Windows where restarts improve training LL | 70.8% |
| Mean next-month test NLL change | -0.38 (lower is better) |
| Windows where restarts improve test NLL | 33.3% |

**Interpretation.** The single initialisation is not always the best in-sample solution: additional restarts improve the training likelihood in about 71% of windows. But the improvement does not translate reliably into better next-month predictive performance; only one third of windows improve on test NLL, and the mean change is small relative to the model's calibration error.

This is a genuine limitation of the reference run, not a hidden tuning opportunity. A three-restart reference run would be more stable in-sample but would multiply the walk-forward runtime by roughly three. The negative/inconclusive primary result is not an artifact of a lucky initialisation in the sense that the MoE is not close to beating 1/N; but the exact MoE Sharpe is sensitive to local optima, as the audit shows.

**Action.** Keep the one-restart reference result. Disclose the audit. A future run with more compute should use at least three restarts and report the restart distribution.

---

## 2. Decision-rule sensitivity

The reference run's forecast means and covariances are saved. This allows the decision rule to be re-solved at alternative risk-aversion and position-cap values without refitting the forecasting models.

Source: `reports/decision_sensitivity.csv`.

### Full out-of-sample Sharpe

| Model | lambda=2, cap 0.35 | lambda=5, cap 0.35 | lambda=10, cap 0.35 | lambda=2, cap 0.25 | lambda=2, cap 0.50 |
|---|---:|---:|---:|---:|---:|
| MoE | 0.790 | 0.818 | 0.829 | 0.791 | 0.663 |
| Ridge | 0.849 | 0.850 | 0.869 | 1.006 | 0.767 |
| Static MV | 0.874 | 0.887 | 0.888 | 0.946 | 0.685 |
| 1/N (reference) | 1.059 | 1.059 | 1.059 | 1.059 | 1.059 |

### Interpretation

- **Higher risk aversion does not rescue the MoE.** Its Sharpe rises modestly from 0.790 at lambda=2 to 0.829 at lambda=10, still far below 1/N at 1.059.
- **Tighter position caps do not rescue the MoE.** At cap 0.25, the MoE Sharpe is essentially unchanged (0.791), while ridge improves to 1.006 and static mean-variance to 0.946. The MoE's problem is not merely concentration.
- **Looser caps hurt everyone.** At cap 0.50, the MoE falls to 0.663 and static mean-variance to 0.685, with larger drawdowns.
- **The ranking is stable.** 1/N is above the MoE in every decision-rule variant tested. The conclusion is not an artifact of lambda=2 or the 0.35 cap.

The decision-rule sensitivity is a diagnostic of the same forecasts. It is not a new set of independent trials, and it is not used to select a different lambda for the headline.

---

## 3. Covariance inflation

The MoE's PIT tails are overrepresented (10.9% below 5% and 10.9% above 95% in the reference run), which means its predictive distribution is overconfident. An overconfident covariance makes the mean-variance optimizer take larger, noisier positions.

The decision-rule sensitivity also multiplies the predictive covariance by 1, 2, and 4 before solving, holding lambda=2 and the cap at 0.35. This is a diagnostic of the calibration-decision link, not a new headline strategy.

| Model | Scale 1 | Scale 2 | Scale 4 |
|---|---:|---:|---:|
| MoE | 0.790 | 0.813 | 0.827 |
| Ridge | 0.849 | 0.850 | 0.859 |
| Static MV | 0.874 | 0.877 | 0.888 |
| 1/N (reference) | 1.059 | 1.059 | 1.059 |

**Interpretation.** Inflating the covariance moves the MoE in the right direction (0.790 to 0.827) but does not change the ranking: it remains far below 1/N. Turnover barely falls (0.422 to 0.419), because the portfolio is fully invested and the mean term dominates the risk term at these parameter values. Covariance inflation is not a rescue.

The pre-committed primary rule uses the model's covariance as-is. The inflation result is a post-hoc diagnostic, reported for completeness.

---

## 4. Gross signal vs net decision value

The portfolio-level gross/net decomposition is informative, but it is **not** a positive-alpha result.

### Holdout

| Model | Gross Sharpe | Net Sharpe | Net - gross | Avg. turnover | Annual cost drag |
|---|---:|---:|---:|---:|---:|
| 1/N | 0.511 | 0.505 | -0.006 | 0.011 | 0.03% |
| MoE | 0.622 | 0.479 | -0.143 | 0.406 | 0.97% |

The MoE's holdout gross Sharpe is numerically above 1/N (0.622 vs 0.511), but the gross difference is **not statistically confirmed**: 95% CI [-0.324, 0.514], p = 0.60. This is a descriptive observation, not an edge.

The cost sensitivity is important. At 5 bps, the MoE net Sharpe is 0.551 vs 1/N 0.508; at 10 bps, 0.479 vs 0.505; at 20 bps, 0.336 vs 0.498. The sign flips between 5 and 10 bps, but the underlying gross difference is within noise. The 5 bps result must not be presented as an edge.

### Cross-sectional information coefficient

For each decision month, we compute the Spearman correlation between the predicted factor returns and the realised next-month returns across the six factors. The mean IC, HAC t-statistic (6 lags), and block-bootstrap interval are:

| Model | Period | Mean IC | HAC t | 95% block CI | p |
|---|---|---:|---:|---:|---:|
| MoE | Full | 0.057 | 2.97 | [0.019, 0.096] | 0.008 |
| MoE | 2006-2026 holdout | 0.040 | 1.35 | [-0.020, 0.102] | 0.194 |
| MoE | Pre-2006 | 0.069 | 2.72 | [0.020, 0.120] | 0.006 |
| Ridge | Full | 0.081 | 4.00 | [0.038, 0.119] | 0.000 |
| Ridge | 2006-2026 holdout | 0.015 | 0.49 | [-0.042, 0.072] | 0.618 |
| Static MV | Full | 0.060 | 3.06 | [0.019, 0.100] | 0.002 |
| Static MV | 2006-2026 holdout | 0.046 | 1.59 | [-0.011, 0.100] | 0.098 |

Source: `reports/red_team_checks.json`, `reports/forecast_ic.csv`.

The full-sample signal is positive and statistically detectable, but it is not confirmed in the holdout. It is also weaker than ridge's full-sample IC. The MoE's gross holdout Sharpe is therefore a noisy descriptive statistic, not a validated gross alpha.

Over the full sample, the MoE's gross Sharpe (0.934) is also below 1/N's gross Sharpe (1.066), so the full-sample shortfall is not only a cost effect.

**Interpretation.** A full-sample predictive signal does not imply net decision value, and a holdout gross Sharpe above the benchmark does not imply a statistically confirmed edge. The missing links are the holdout, the cost of trading a noisy signal, and the uncertainty around the gross difference.

---

## 5. What was not run

The evaluation protocol pre-specified training-window and number-of-experts robustness. Those require refitting the model and were not executed in the reference run:

- `min_train` in {60, 120, 180}.
- `K` in {2, 3, 4}.

The report states this explicitly. A future run should execute the grid and report the distribution, not a best cell. The current conclusion is conditional on the pre-committed configuration and the checks that were run.

This is a disclosed deviation from the planned robustness grid, not a silent omission. The reference result does not beat 1/N in any decision-rule variant, and the gross signal is small. Additional configurations would add multiple-testing burden without a pre-committed reason to expect a different conclusion.

---

## 6. Net effect on the conclusion

The primary holdout result is **inconclusive**: the MoE is statistically indistinguishable from 1/N over 2006-2026 (net Sharpe 0.479 vs 0.505; difference -0.026; 95% CI [-0.470, 0.389]; p=0.862). The full-sample result is negative: the MoE trails 1/N by 0.269 Sharpe (95% CI [-0.518, 0.006]; p=0.058).

The gross/net decomposition is more interesting than the headline. In the holdout, the MoE's **gross** Sharpe is 0.622 versus 1/N's 0.511, but its **net** Sharpe is 0.479 versus 0.505. The model trades about 40.6% of notional per month, producing roughly 0.97% annual cost drag. The full-sample signal is positive but small (mean cross-sectional IC 0.057, t=2.97); the holdout does not confirm it, and it does not survive the cost of expressing it. Over the full sample, even the gross Sharpe is below 1/N (0.934 vs 1.066), so the long-run shortfall is not only a cost effect.

The model is also miscalibrated in the tails: 10.9% of PIT values fall in each 5% tail, and none of the six assets passes the 5% KS threshold. The restart audit shows local-optimum sensitivity in the training likelihood.

The honest summary is: **the MoE produced a small gross signal in the holdout, but it did not earn its complexity after costs; the net holdout is indistinguishable from 1/N, and the full-sample evidence points the other way.**
