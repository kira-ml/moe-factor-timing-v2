# 04 - Evaluation Protocol

**Status:** Pre-committed before the reference run
**Principle:** The evaluation window is not the selection window.

---

## 1. Estimand and primary endpoint

The primary estimand is the difference in **annualised net Sharpe ratio** between the MoE model and the 1/N benchmark in the pre-committed holdout:

```
Delta_SR = SR_MoE(holdout) - SR_1/N(holdout)
```

Net returns are after transaction costs. The Sharpe ratio uses monthly excess returns:

```
SR = mean(r_net - rf) / std(r_net - rf) * sqrt(12)
```

The primary decision rule is:

- **Supported:** the block-bootstrap 95% confidence interval for `Delta_SR` lies entirely above zero, and the deflated Sharpe diagnostic does not reject the result at the 5% level.
- **Inconclusive:** the confidence interval includes zero. This is not evidence of equivalence.
- **Negative:** the confidence interval lies entirely below zero.

The full out-of-sample window (1974-2026) is reported as a secondary result. It is not the confirmatory test.

---

## 2. Secondary endpoints

| Endpoint | Definition | Role |
|---|---|---|
| Mean-variance utility | `mean(r_net) - (lambda/2) var(r_net)`, annualised | Economic significance |
| Maximum drawdown | Minimum cumulative wealth drawdown | Tail risk |
| Calmar ratio | Annual return / absolute max drawdown | Risk-adjusted return |
| Turnover | Mean two-way monthly turnover | Implementation burden |
| Cost drag | Mean monthly cost, annualised | Cost survival |
| Gross vs net return | Before and after costs | Whether costs explain the result |
| Predictive NLL | Mean negative log predictive density | Forecast quality |
| PIT calibration | KS test on marginal PIT values | Distributional calibration |
| Deflated Sharpe | Selection-adjusted Sharpe probability | Multiple-testing control |
| Subperiod stability | Sharpe by thirds of the OOS window | Non-stationarity |

Secondary endpoints are reported in full. They are not used to rescue a failed primary endpoint.

---

## 3. Holdout

- **Start:** 2006-01-31.
- **End:** the last available month in the frozen panel.
- **Length:** 248 monthly decisions at the reference snapshot.
- **Use:** confirmatory evaluation only.
- **Prohibited:** fitting, hyperparameter selection, feature selection, model selection, or "one more configuration" after seeing holdout results.

The holdout starts after the period in which the V1 and prior-V2 lessons were learned. It is not perfectly clean - the researcher has seen the broad historical record - but no V2 model parameter is chosen from it.

---

## 4. Uncertainty

### Block bootstrap

Monthly strategy returns are serially dependent and heteroskedastic. An iid bootstrap would understate uncertainty. The protocol uses a circular block bootstrap:

- 1000 repetitions,
- block length 6 months,
- paired resampling of the model and benchmark return series,
- fixed seed (42).

The reported interval is the 2.5th to 97.5th percentile of the bootstrapped Sharpe difference. The p-value is the two-sided bootstrap tail probability.

The bootstrap is a **sampling** statement about the historical path. It does not account for model uncertainty, publication bias, or the possibility that the factor panel itself is revised.

### Power

The minimum detectable annualised Sharpe against zero is approximated by:

```
MDE = (z_{1-alpha/2} + z_{power}) / sqrt(years)
```

with `alpha = 0.05` and `power = 0.80`. This is conservative for a paired difference, because the two strategies are correlated.

The report includes the MDE for the full window and the holdout. A result that is smaller than the MDE is reported as inconclusive, not as proof of no effect.

---

## 5. Multiple testing

The reference comparison evaluates seven models:

1. `one_over_n`
2. `inverse_vol`
3. `momentum`
4. `cash`
5. `static_mv`
6. `ridge`
7. `moe`

`N_trials = 7` is recorded in the config. Robustness configurations add to the trial count; the final report discloses the total.

The deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) adjusts for the number of trials and the dispersion of trial Sharpe ratios. The implementation uses:

- the observed Sharpe,
- the sample length,
- skewness and kurtosis,
- the expected maximum Sharpe under `N_trials`,
- a trial-Sharpe dispersion estimated from the cross-section of models, with a floor at the null standard error.

A high deflated Sharpe is necessary but not sufficient for a confirmatory claim. The holdout is the primary guard against selection.

---

## 6. Calibration

For each probabilistic model and each asset:

```
u_{j,t} = F_j(y_{j,t} | x_t)
```

where `F_j` is the marginal predictive CDF. For a Gaussian model, this is `Phi((y - mu)/sigma)`. For the MoE, it is the exact marginal mixture CDF, not a Gaussian approximation.

A calibrated model has `u ~ Uniform(0,1)`. The protocol reports:

- mean PIT,
- fraction below 5% and above 95%,
- pooled KS statistic and p-value,
- per-asset KS results,
- PIT histograms.

Calibration is a diagnostic. A calibrated distribution can still produce poor decisions; a miscalibrated distribution can produce good decisions by accident.

---

## 7. Deflated Sharpe ratio

The probabilistic Sharpe ratio (PSR) is:

```
PSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew*SR + ((kurt - 1)/4)*SR^2) )
```

using monthly Sharpe units and `T` monthly observations. The deflated Sharpe ratio (DSR) replaces `SR0` with the expected maximum Sharpe under `N_trials` independent trials:

```
SR0 = sqrt(V[SR]) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)))
```

where `gamma` is the Euler-Mascheroni constant and `V[SR]` is the variance of trial Sharpe ratios.

The implementation reports `PSR`, `DSR`, `DSR p-value`, `E[max SR]`, and `N_trials`. The assumptions are stated in the report.

---

## 8. Robustness

The following dimensions are pre-specified. They are reported as a distribution, not used to select a winner:

| Dimension | Values | Status |
|---|---|---|
| Training window | 60, 120, 180 months | Planned |
| Number of experts | 2, 3, 4 | Planned |
| Risk aversion | 2, 5, 10 | Sensitivity from recorded forecasts |
| One-way cost | 5, 10, 20 bps | Sensitivity from recorded turnover |
| Max weight | 0.25, 0.35, 0.50 | Sensitivity from recorded forecasts |
| Turnover cap | none, 0.50 | Sensitivity from recorded forecasts |

The **decision-rule sensitivities** can be computed from the saved forecast means and covariances without refitting. The **model sensitivities** require refitting; if a dimension is not executed, the report says so explicitly.

No robustness result overrides the pre-committed holdout. A configuration that only works after the fact is a robustness failure, not a discovery.

---

## 9. Stop and kill criteria

The project stops when any of the following is true:

1. **Primary failure:** the holdout Sharpe difference is negative with a confidence interval entirely below zero.
2. **Inconclusive:** the confidence interval includes zero and the study is underpowered for the observed effect. Report the uncertainty and stop; do not tune on the holdout.
3. **Calibration failure plus no decision value:** the MoE fails PIT calibration and does not beat 1/N. Report both.
4. **Implementation failure:** a contract test, leakage test, determinism test, or solver-fallback gate fails. Fix the pipeline before interpreting any result.

The iteration budget is **one reference run plus the pre-specified robustness grid**. There is no "one more K" or "one more feature" after seeing the holdout.

---

## 10. Reporting standard

Every result report must include:

- the primary endpoint and its confidence interval,
- the full out-of-sample and holdout tables,
- turnover, cost drag, and average deployment,
- predictive NLL and PIT calibration,
- the deflated Sharpe and `N_trials`,
- power and minimum detectable effects,
- subperiod stability,
- limitations and non-goals,
- the run manifest and config hash.

A report that omits any of these is incomplete, regardless of the headline number.

---

## 11. What is not tested

- Intraday execution, market impact, or capacity.
- Short selling, leverage, or derivatives.
- International, credit, commodity, or currency factors.
- Deep neural or reinforcement-learning models.
- Vintage-aware macroeconomic features.
- Live trading.

These are explicit non-goals, not silent omissions.
