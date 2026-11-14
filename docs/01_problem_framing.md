# 01 - Problem Framing

**Project:** moe-factor-timing v2
**Status:** Pre-committed before the reference run
**Last updated:** 2026-11-14

---

## 1. Motivation

Equity factor premia are not constant. Value can underperform for years, momentum can reverse violently, and the profitability and investment premia have different macroeconomic exposures. A natural hypothesis is that a model which represents **latent regimes** and conditions factor allocations on observable state should make better allocation decisions than a static rule.

Version 1 tested a version of this hypothesis and reported a Sharpe ratio of 1.49. The archived implementation cannot support that claim: the out-of-sample window was 42 months, VIX was treated as a tradable factor, the feature count roughly equalled the training sample, transaction costs were never charged because of an implementation bug, and the final configuration was selected after observing the evaluation window. See `docs/06_lessons_from_v1.md`.

Version 2 asks the same substantive question with an evidence design that can answer it.

---

## 2. Research question

> **Can a probabilistic mixture-of-experts model that conditions factor allocations on observable state variables improve out-of-sample net risk-adjusted factor allocation relative to a naive equal-weight portfolio, after realistic costs?**

The question is deliberately narrow:

- **Population:** six long-short US equity factor portfolios (market, size, value, profitability, investment, momentum).
- **Decision:** monthly long-only allocation across factor sleeves, fully invested, with a maximum weight.
- **Benchmark:** equal weight across the six sleeves, rebalanced monthly.
- **Outcome:** net Sharpe ratio after costs, with mean-variance utility and drawdown as secondary outcomes.
- **Evidence:** chronological walk-forward with a pre-committed holdout.

---

## 3. Hypotheses

### H1 - Primary (confirmatory)

The MoE model has a higher net Sharpe ratio than 1/N in the pre-committed holdout (2006-01 onward), with a block-bootstrap 95% confidence interval excluding zero.

**Null H1_0:** the holdout Sharpe difference is zero or negative.

### H2 - Forecast-layer value (secondary)

The MoE beats the static mean-variance and ridge forecasts on the same decision rule. This isolates whether conditional regime information adds value beyond a simple plug-in forecast.

### H3 - Calibration (secondary)

The MoE predictive distribution is approximately calibrated: pooled PIT values are consistent with a uniform distribution, and the tails are not systematically overrepresented.

Calibration is necessary but not sufficient for H1. A calibrated distribution can still produce poor decisions, and a miscalibrated distribution can produce good decisions by accident.

### H4 - Cost survival (secondary)

Any gross forecast improvement survives realistic turnover costs. This is tested by reporting gross and net performance, turnover, and cost drag, and by a cost-sensitivity analysis.

---

## 4. Why 1/N is the benchmark

DeMiguel, Garlappi, and Uppal (2009) show that naive diversification is hard to beat out of sample because estimation error in expected returns and covariances can overwhelm the theoretical gains from optimisation. A factor-timing model must therefore beat 1/N, not merely beat cash or a weak linear model.

The benchmark is monthly-rebalanced equal weight. It is not a straw man: it is diversified, transparent, has low turnover, and is the relevant outside option for a systematic allocator.

---

## 5. Mechanism

The hypothesised mechanism is:

1. Factor premia are state-dependent.
2. The state is partially observable through lagged returns and realised volatility.
3. A mixture of linear experts can approximate a state-dependent conditional mean while retaining a tractable predictive distribution.
4. A risk-aware decision rule can convert a better conditional distribution into better allocations.

The mechanism fails if:

- the state signal is too weak relative to factor-return noise;
- the mixture overfits the training window;
- the decision rule amplifies estimation error into turnover;
- or the costs of trading the signal exceed its value.

These failure modes are the subject of the pre-mortem in `docs/08_risk_register.md`.

---

## 6. Scope

### In scope

- US equity factor portfolios, monthly frequency, long-only allocation.
- Probabilistic forecasts with a full predictive covariance.
- Transaction costs and turnover accounting.
- Chronological validation, a holdout, uncertainty quantification, and multiple-testing disclosure.
- Reproducible artifacts and a written report.

### Out of scope

- Live trading, execution, market impact, or capacity estimation.
- Short selling, leverage, derivatives, or market timing.
- Daily or intraday frequency.
- International, credit, commodity, or currency factors.
- Deep neural networks or reinforcement learning.
- Causal claims about regimes.
- A claim that the Ken French research panel is a literal point-in-time trading record.

---

## 7. Contributions

1. **A rigorous replication attempt** of the V1 hypothesis on a long, investable factor panel.
2. **A diagnosis of V1** that separates implementation bugs from research-design failures.
3. **A reusable benchmark** with typed contracts, a proper probabilistic MoE, drift-aware cost accounting, and a manifest.
4. **An honest evidence statement**: positive, negative, or inconclusive, with the uncertainty reported.
5. **A decision-focused evaluation** that reports calibration, turnover, and deflated Sharpe alongside returns.

The project's value does not depend on a positive result. A well-identified negative result is a contribution.

---

## 8. Success and kill criteria

The full protocol is in `docs/04_evaluation_protocol.md`. In summary:

- **Confirmatory success:** H1 confidence interval excludes zero in the holdout and the deflated Sharpe diagnostic does not reject the result as selection.
- **Inconclusive:** the confidence interval includes zero. This is not evidence of equivalence.
- **Failure:** the confidence interval is entirely below zero, or the model fails calibration and decision value simultaneously.
- **Kill:** if H1 fails in the holdout, do not re-tune on the holdout. Report the negative or inconclusive result and stop the current direction.

The number of configurations examined is disclosed. Robustness configurations are reported as a distribution, not used to select a winner.

---

## 9. What would falsify the hypothesis

- MoE's holdout Sharpe is below 1/N with a confidence interval entirely below zero.
- MoE's gross performance is positive but its net performance is not, and the cost sensitivity shows that no plausible cost assumption rescues it.
- MoE's predictive distribution fails calibration and its decisions are indistinguishable from a static forecast.
- The apparent improvement disappears under a reasonable change in training window, number of experts, or risk aversion.

A single favorable configuration is not evidence. The protocol is designed to make that clear.
