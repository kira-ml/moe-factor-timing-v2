# Lessons from Version 1

**Status:** Reference document for V2
**Archived tree:** local-only `archive/v1/` (not redistributed; tag `v1.0-final`)
**Rule:** This document is a diagnosis, not a retrospective celebration. It records what failed, why it failed, and what V2 changes.

---

## 1. What V1 claimed

V1 reported a Mixture of Experts (MoE) model with a Sharpe ratio of **1.49** and an annualised return of **40.61%** over 42 out-of-sample months, outperforming momentum, rolling average, linear regression, random forest, and persistence. It also reported four latent regimes with distinct return and volatility characteristics.

The archive shows that these numbers were produced by a pipeline that changed materially from day to day. The same project reported MoE Sharpe ratios of approximately:

| Date / configuration | Sharpe | Annual return | Max drawdown | Notes |
|---|---:|---:|---:|---|
| Day 1 initial run | 1.7667 | n/a | n/a | All models had identical Sharpe because metrics used actual returns, not model allocations |
| Day 2 after allocation fix | 0.3793 | 0.09% | -58.94% | 83 predictions, 28 features |
| Day 3 after tuning | 0.5060 | 13.94% | -52.29% | K=4, 100 EM iterations, magnitude-weighted |
| PyTorch MoE, single split | 3.1391 | 70.88% | -4.92% | Not comparable to the expanding-window result |
| PyTorch MoE, expanding | 0.1528 | -6.20% | -72.03% | Same model, different evaluation |
| Final `min_train=96` | 1.49 | 40.61% | -13.73% | 42 predictions; selected after testing 60/96/132 |

A result that moves by more than an order of magnitude across implementation choices is not a stable empirical finding. It is a symptom of a pipeline that was not frozen before evaluation.

---

## 2. What V1 got right

These are real strengths and V2 preserves them in spirit:

1. **End-to-end intent.** V1 connected data ingestion, features, models, backtesting, evaluation, regime analysis, and a written report.
2. **A baseline family.** Persistence, rolling average, momentum, linear regression, and random forest were all implemented.
3. **A reproducibility instinct.** The project saved configurations, predictions, and `pip freeze` output, and eventually documented a run timestamp.
4. **A research question worth asking.** Whether probabilistic regime models improve factor allocation is a legitimate question.
5. **Willingness to report limitations.** The final README contains a non-claims section and a limitations list.

V2 does not discard these goals. It rebuilds the execution around them.

---

## 3. What went wrong

### 3.1 The binding constraint was sample size and data quality, not model sophistication

V1 used 155 months of data (2013-08 to 2026-07), with 42 out-of-sample predictions in the headline configuration. A 42-month window cannot distinguish a Sharpe of 1.3 from 1.1 with meaningful confidence. The project responded by adding model complexity (more experts, more iterations, a PyTorch variant, more features), which is the wrong response to a power constraint.

**V2 response:** use the Ken French five-factor plus momentum panel from 1963-07 onward (759 monthly observations), with a pre-committed 2006+ holdout. This is a twenty-year confirmatory window and a fifty-year full out-of-sample window.

### 3.2 Configuration selection was performed on the evaluation window

The final configuration (`min_train=96`, K=4, 100 EM iterations, magnitude-weighted) was selected after observing the performance of `min_train=60`, `96`, and `132`. The README presents `min_train=96` as a balance between stability and sample size, but the stability criterion was the realised Sharpe. This is selection on the test set.

**V2 response:** hyperparameters are fixed a priori in `configs/default.yaml`. The holdout is not used for model selection. Robustness configurations are reported as a distribution, not used to pick a winner. The number of trials is disclosed and deflated.

### 3.3 The universe was not investable

V1 included the VIX spot index as a "factor" and allocated to it. VIX is not directly tradable; its monthly percentage change is not a portfolio return. The six "factors" were also five long-only US equity ETFs with substantial overlap plus VIX, so the exercise was closer to equity-beta timing than factor timing. The ETF history begins in 2013 because MTUM/QUAL/USMV were launched then, which caused the short sample.

**V2 response:** use six long-short factor portfolios (market, size, value, profitability, investment, momentum) from Ken French. The allocation is a fund-of-factor-strategies interpretation. VIX is excluded. The sample starts in 1963.

### 3.4 The feature set was too large for the training window

V1 used 96 features (lagged returns plus FRED transformations) with 96 training months. The linear model's RMSE of 192 is a symptom of an ill-conditioned design. The MoE was fitting a high-dimensional gating and expert problem with roughly as many parameters as observations.

**V2 response:** 30 features (four lags per factor plus twelve-month realised volatility), with 120+ training months and standardisation fit inside each training window. The model-to-observations ratio is reported. The feature set is deliberately parsimonious and only uses information observable at the decision date.

### 3.5 Macro features were not point-in-time

V1 loaded FRED's latest revised series, added transformations, and lagged them. This does not reconstruct what an investor knew at the time. Revisions, publication delays, and the difference between first-release and final data are not modelled. The project's own archived postmortem from the prior V2 attempt later confirmed that ALFRED vintages were never implemented.

**V2 response:** the main analysis uses no revised macro series. Features are lagged returns and realised volatility computed from the return panel itself. This removes a whole class of look-ahead risk. Macro extensions are deferred to a separate, vintage-aware study.

### 3.6 The Mixture of Experts was not a probabilistic mixture

The `SimpleMoE` implementation used `exp(-MSE)` as a likelihood without a variance term, trained the gating network on in-sample responsibilities, had no convergence criterion, used `np.random.seed` in the constructor, and did not produce a calibrated predictive distribution. It was a soft-clustering heuristic, not a mixture of experts with a likelihood.

**V2 response:** implement a genuine mixture of linear Gaussian experts trained by EM, with a multinomial-logit gate, weighted ridge experts, shrunk covariance estimates, a training log-likelihood, a convergence tolerance, and a predictive mixture distribution. Calibration is evaluated with PIT and KS tests.

### 3.7 Transaction costs were not actually charged

The V1 cost function was called once per one-row split. Its turnover loop started at index 1, so for a single-row prediction it never compared against the previous month's weights. The reported "transaction costs have negligible impact" (Sharpe 0.4417 to 0.4415) was an artifact of zero turnover, not a finding about costs.

**V2 response:** turnover is computed against the **drifted** pre-rebalance weights, the cost is charged every month, the first rebalance is charged explicitly, and the accounting identity is tested in `tests/test_backtest_accounting.py`. The reference report includes turnover and cost drag.

### 3.8 There was no 1/N benchmark

V1 compared models to each other, not to the strongest simple alternative. An `equal_weight` allocation strategy existed in the code but was not part of the final comparison. DeMiguel, Garlappi, and Uppal (2009) is the relevant base rate: 1/N is hard to beat out of sample, especially with estimation error.

**V2 response:** 1/N is the primary benchmark. Inverse volatility and momentum are additional simple alternatives. Static mean-variance and ridge isolate the value of conditional forecasting from the value of the decision rule.

### 3.9 The evaluation layer had no uncertainty statement

V1 reported point estimates: Sharpe, return, drawdown, win rate. It had no confidence intervals, no bootstrap, no power analysis, no multiple-testing correction, and no deflated Sharpe ratio. The prior V2 attempt later identified the same gaps (E3, E7, E8, E9, S1-S3) and did not close all of them.

**V2 response:** block-bootstrap confidence intervals for Sharpe differences, a power analysis with minimum detectable effects, PIT calibration, and a deflated Sharpe ratio with explicit trial-count assumptions.

### 3.10 Two implementations drifted apart

V1 had both a scikit-learn `SimpleMoE` and a separate PyTorch `TorchMoE`. They used different data preparation, training loops, and evaluation paths. The PyTorch model's single-split Sharpe of 3.14 and expanding-window Sharpe of 0.15 were never reconciled. This is a single-source-of-truth failure.

**V2 response:** one model implementation per method, one backtest engine, one evaluation path, one report generator. There is no second implementation to drift.

### 3.11 Engineering hygiene was weak

The V1 bug tracker records 17 fixed issues, including an iterator consumed by the first model, nested arrays stored as `object` dtype, stale factor names, missing `--quick-test` logic, and a `try/except` without an `except`. There were no tests, no manifest, no config schema, and no deterministic-run check. The prior V2 attempt added some of these but still had solver nondeterminism and a retroactive Phase 1 gate.

**V2 response:** typed config validation, a package layout, 40+ contract and integration tests, a deterministic solver, a manifest with artifact hashes, and an archive integrity fingerprint.

---

## 4. Root causes

The individual bugs are symptoms. The deeper causes are:

1. **Optimising the model before securing the evidence.** The binding constraint was a short, non-investable sample. More experts and more features could not fix it.
2. **No pre-commitment.** The evaluation window was also the selection window. Every new result created pressure for "one more configuration."
3. **A metric without a counter-metric.** Sharpe was the headline; turnover, cost drag, exposure, and calibration were not gating criteria. A model with a 33.68 RMSE and a 1.49 Sharpe should have triggered a "what is driving this?" audit.
4. **A pipeline that failed silently.** Zero transaction costs, identical Sharpes, and consumed iterators did not stop the run. The system preferred a number to an error.
5. **Complexity as a substitute for identification.** The project could not say why the MoE should work, so it added capacity. V2 states the mechanism (soft regime membership in conditional factor means), the null (1/N), and the test that would falsify it.

---

## 5. What V2 changes, one-to-one

| V1 failure | V2 design response | Where it lives |
|---|---|---|
| 42-month test window | 1963-2026 panel; 2006+ holdout | `data/reference/`, `configs/default.yaml` |
| Selection on test | Fixed hyperparameters; holdout not used for selection | `docs/03`, `docs/04`, `DECISIONS.md` |
| VIX as a tradable factor | Six long-short Ken French factor portfolios; VIX excluded | `docs/02`, `data/reference/` |
| 96 features / 96 months | 30 causal features; training-only scaling | `features/build.py` |
| FRED latest + lag | No revised macro data in the main analysis | `docs/02`, `docs/08` |
| Heuristic MoE | Proper EM mixture of linear Gaussian experts | `models/moe.py` |
| Costs never charged | Drift-aware turnover and monthly cost accounting | `backtest/engine.py`, `tests/test_backtest_accounting.py` |
| No 1/N | 1/N is the primary benchmark | `models/benchmarks.py` |
| Point estimates only | Block bootstrap, power, PIT, DSR | `evaluation/` |
| Two drifting implementations | One registry, one engine, one report | `models/registry.py`, `run/orchestrator.py` |
| No tests or manifest | Contract tests, end-to-end test, hashed manifest | `tests/`, `run/manifest.py` |
| Silent failures | Fail-fast model errors; solver fallback rate gate | `backtest/engine.py` |

---

## 6. What V2 does not fix

V2 is a stronger benchmark, not a solution to every problem in the research area.

- **The Ken French panel is reconstructed, not vintage.** It is a research benchmark, not a literal point-in-time trading record.
- **Internal factor-portfolio turnover is not modelled.** The cost model covers rebalancing across factor sleeves, not the turnover inside each long-short portfolio.
- **The factor universe is fixed and well known.** Publication and crowding effects are not modelled.
- **The model class is deliberately small.** A linear-Gaussian MoE can only capture regime-dependent linear conditional means. It is not a test of deep regime models.
- **Monthly frequency only.** Daily or intraday dynamics are out of scope.
- **No live deployment.** The result is evidence about a backtest protocol, not a forecast of live performance.

These residuals are recorded in `docs/08_risk_register.md` and in the report's limitations section.

---

## 7. The one-sentence lesson

**Version 1 did not fail because the MoE was not sophisticated enough; it failed because the evidence was too weak, the universe was not investable, the costs were not charged, and the evaluation window was also the selection window.**

V2 is designed so that each of those sentences would be false.
