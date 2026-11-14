# 08 - Risk Register and Pre-Mortem

**Purpose:** identify how this project could produce a false positive, a false negative, or an unreproducible result, and state the mitigation.

---

## 1. Pre-mortem

It is 2027. The V2 report claims a result that does not survive scrutiny. Which failure is most likely?

1. **The holdout was contaminated by implementation choices.** The model was "only" tuned on the full out-of-sample period, but the feature set, K, and decision rule were influenced by what the researcher saw. Mitigation: fixed config, holdout not used for selection, decision log, trial count.
2. **The costs are too low.** The model trades 40-50% of notional per month; 10 bps may understate factor-sleeve turnover and financing. Mitigation: cost sensitivity at 5/10/20 bps, turnover reported, gross vs net.
3. **The factor panel is revised.** Ken French data are reconstructed, so the historical signal is cleaner than what was available live. Mitigation: explicit caveat, no claim of live tradability, vintage-aware extension deferred.
4. **The result is a single lucky configuration.** A different K or training window reverses it. Mitigation: pre-specified robustness grid, distribution not best-cell reporting.
5. **The software is wrong.** A drift, alignment, or cost bug makes the model look better or worse. Mitigation: contract tests, accounting identity, future-perturbation test, manifest, independent review.
6. **The negative result is overclaimed.** A 20-year holdout has limited power; "no evidence of improvement" is not "evidence of no improvement." Mitigation: power analysis, confidence intervals, explicit inconclusive category.
7. **The regime labels are overinterpreted.** The mixture components are statistical clusters, not identified economic regimes. Mitigation: labels are descriptive; no causal claim; gating diagnostics only.

---

## 2. Risk table

| ID | Risk | Likelihood | Impact | Mitigation | Residual risk |
|---|---|---|---|---|---|
| R1 | Look-ahead in features | Medium | High | Causal construction; future-perturbation and truncation tests; runtime target assertion | Revised source data remain |
| R2 | Survivorship / backfill | Medium | Medium | Ken French panel is standard; disclose reconstruction | Cannot fully eliminate |
| R3 | Cost understatement | High | High | Drift-aware turnover; 10 bps base; 5/20 bps sensitivity; gross vs net | Internal factor turnover not modelled |
| R4 | Multiple testing / p-hacking | High | High | Fixed config; holdout; N_trials; DSR; no post-hoc tuning | Researcher degrees of freedom remain |
| R5 | Low power | High | Medium | MDE reported; inconclusive category; full + holdout windows | Cannot detect small effects |
| R6 | Model instability | Medium | Medium | Regularisation; covariance shrinkage; deterministic k-means; restart audit | EM local optima remain |
| R7 | Solver failure | Low | Medium | Deterministic SLSQP; fallback rate gate at 5%; status recorded | Ill-conditioned problems |
| R8 | Implementation drift | Low | High | One registry, one engine, one report; tests; manifest | Future contributors can still fork |
| R9 | Data revision | Medium | Medium | Disclosed; no live claim; vintage extension deferred | Historical panel is not vintage |
| R10 | Regime labels overinterpreted | Medium | Low | Descriptive language; no causal claim; diagnostics only | Readers may overinterpret |
| R11 | Factor crowding / publication | Medium | Medium | Disclosed; fixed well-known universe | Not modelled |
| R12 | Backtest-to-live gap | High | High | Explicit non-goal; no live claim; costs and turnover reported | Always present |
| R13 | Negative result overclaimed | Medium | Medium | Power analysis; "inconclusive" category; confidence intervals | Readers may conflate |
| R14 | Archive modification | Low | High | Read-only policy; fingerprint test; manifest fingerprint | Accidental edits |
| R15 | Environment drift | Low | Medium | Package versions and git commit in manifest; pinned dependencies | OS/BLAS differences |

---

## 3. Counter-metrics

Every headline metric is paired with an orthogonal counter-metric:

| Headline | Counter-metric | Why |
|---|---|---|
| Sharpe | Turnover, cost drag | A Sharpe that is bought with trading is fragile |
| Sharpe | Max drawdown, Sortino | Tail risk is not captured by volatility |
| Mean-variance utility | Average deployment, max weight | Utility can be gamed by cash or concentration |
| Predictive NLL | PIT calibration | A sharp but miscalibrated distribution is not useful |
| Best model | Deflated Sharpe, N_trials | Selection inflates the maximum |
| Holdout result | Full OOS, subperiods | A single window can be lucky |

---

## 4. Kill criteria

Stop the current direction if:

1. The holdout Sharpe difference is negative with a 95% CI entirely below zero.
2. The confidence interval includes zero and the observed effect is below the minimum detectable effect.
3. The MoE fails PIT calibration and does not beat 1/N on the decision metric.
4. A leakage, accounting, or determinism test fails.
5. The solver fallback rate exceeds 5%.

Do not respond to a failed criterion by adding features, experts, or configurations. Record the outcome and stop.

---

## 5. Rollback

- Code: git revert to the last clean tag.
- Data: restore `data/reference/` from the committed snapshot.
- Results: the run directory is immutable; a failed run is kept with its manifest and marked failed.
- Archive: never modified; fingerprint test is the guard.
