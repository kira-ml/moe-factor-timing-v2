# 11 - Red-Team Validation and Publication Gate

**Status:** Publication gate
**Run:** `runs/reference`
**Reviewer stance:** skeptical external referee
**Question:** Is the V2 result robust and defensible enough to publish as a one-page research note?

---

## 1. Gate decision

**Publishable as a rigorous negative/inconclusive benchmark. Not publishable as a positive gross-alpha or net-alpha claim.**

The evidence supports a narrower and more useful thesis:

> A probabilistic mixture-of-experts model found a small, statistically detectable full-sample cross-sectional signal in 63 years of factor returns. The signal did not survive the 2006-2026 holdout, did not produce a net Sharpe above 1/N at the pre-committed cost, and the sign of the holdout net difference flips between 5 and 10 bps. Equal weight remains hard to beat.

This is a contribution about **evidence design and the gap between predictive signal and net decision value**, not a claim that the model works.

---

## 2. How the review was conducted

1. **Inversion.** Asked how the result could look more favorable than it is: cost assumptions, holdout selection, single initialization, gross-vs-net framing, data revisions, and multiple testing.
2. **Falsification.** Searched for the observation that would disprove the emerging story. The decisive test was the holdout gross/IC significance and the cost-sensitivity sign flip.
3. **Adversarial review.** Treated the result as an external referee would: Is the benchmark fair? Is the holdout clean? Is the gross signal statistically real? Does the decision rule give the model a fair chance? Are the data point-in-time?
4. **Recomputation.** All checks were rerun from the saved run artifacts with `scripts/red_team_checks.py`, not copied from the report.

Evidence artifact: `reports/red_team_checks.json`.
Cost table: `reports/red_team_cost_sensitivity.csv`.

---

## 3. Findings

| # | Severity | Finding | Evidence | Disposition |
|---|---|---|---|---|
| R1 | **High** | The holdout net sign flips between 5 and 10 bps. | MoE net Sharpe 0.551 at 5 bps vs 1/N 0.508; 0.479 vs 0.505 at 10 bps; 0.336 vs 0.498 at 20 bps. | Report the crossover. Do not claim the 5 bps result as an edge. The pre-committed 10 bps result is primary. |
| R2 | **High** | The holdout gross edge is not statistically confirmed. | Holdout gross Sharpe difference 0.111, 95% CI [-0.324, 0.514], p=0.60. | Describe the holdout gross Sharpe as noisy/descriptive, not a confirmed edge. |
| R3 | **High** | The holdout information coefficient is not significant. | MoE holdout mean IC 0.040, HAC t=1.35, block-bootstrap 95% CI [-0.020, 0.102], p=0.194. Full-sample IC 0.057, t=2.97; pre-2006 IC 0.069, t=2.72. | The full-sample signal is real; the holdout does not confirm it. Do not call it a holdout signal. |
| R4 | **Medium-High** | Single initialization makes the exact MoE number local-optimum sensitive. | 24-window audit: three restarts improve training LL in 71% of windows but test NLL in only 33%. | Disclose; present the MoE result as a point estimate with sensitivity, not a precise alpha. The 1/N dominance is robust across subperiods. |
| R5 | **Medium-High** | The probabilistic distribution is miscalibrated. | MoE PIT tails 10.9% vs 5% expected; pooled KS p<0.001; 0/6 assets pass. | Do not describe the model as well calibrated. Calibration is a failed secondary claim. |
| R6 | **Medium** | The panel is retrospective and reconstructed. | Ken French factors are built from the current CRSP database, not vintage snapshots. | Label the study as retrospective; no live-performance claim. |
| R7 | **Medium** | Internal factor-sleeve turnover is not modeled. | Costs are charged at the fund-of-strategies level; long-short factor turnover and financing are excluded. | Report cost sensitivity; do not claim precise net alpha. |
| R8 | **Medium** | The holdout is underpowered for small effects. | Minimum detectable annualized Sharpe 0.62 over 248 months. | Use the "inconclusive" category; do not claim equivalence to 1/N. |
| R9 | **Medium** | The decision rule conflates forecast quality with strategy design. | The MoE and ridge use the optimizer; 1/N is a direct allocator. | State that this is a strategy comparison. The static-MV/ridge comparisons isolate the forecast layer partially. |
| R10 | **Medium** | The "pre-registered" label is too strong. | The config was fixed before the final run, but the design was informed by V1 and prior V2. | Use "pre-committed evaluation protocol"; disclose the prior exposure. |
| R11 | **Low-Medium** | The deflated Sharpe is non-discriminating. | Every model has PSR/DSR near 1.0; the test is against zero, not against 1/N. | Do not headline DSR. Explain that the primary comparison is the paired bootstrap. |
| R12 | **Low** | Leakage and alignment checks pass. | Future-perturbation and truncation tests pass; target-date assertions pass; no missing months. | No action. |

---

## 4. What survives scrutiny

These claims are supported by the run artifacts:

1. **1/N dominates the MoE over the full sample.** Net Sharpe 1.059 vs 0.790; difference -0.269; 95% CI [-0.518, 0.006].
2. **1/N wins all three subperiods.** 1974-1991: 1.400 vs 1.141. 1991-2009: 1.165 vs 0.768. 2009-2026: 0.659 vs 0.473.
3. **The confirmatory holdout is inconclusive.** Net difference -0.026; 95% CI [-0.470, 0.389]; p=0.862.
4. **The MoE is a high-turnover strategy.** Average turnover 40.6% of notional per month; cost drag 0.97% per year at 10 bps.
5. **The distribution is miscalibrated.** 10.9% of PIT values in each 5% tail; 0/6 assets pass the 5% KS threshold.
6. **The decision-rule sensitivity does not rescue the MoE.** It remains below 1/N at risk aversion 2/5/10, caps 0.25/0.35/0.50, and covariance scales 1/2/4.
7. **The full-sample gross Sharpe is also below 1/N.** 0.934 vs 1.066; the full-sample shortfall is not only a cost effect.

---

## 5. What does not survive scrutiny

These claims must not appear in the publication:

- "The MoE has a gross alpha in the holdout." The gross Sharpe difference CI includes zero and the holdout IC t-stat is 1.35.
- "The MoE beats 1/N at 5 bps." The sign flips between 5 and 10 bps, but the underlying gross difference is not statistically confirmed. This is a cost-sensitivity warning, not an edge.
- "The model is well calibrated." It is not.
- "The result is pre-registered." It is a pre-committed analysis of a retrospective panel, with prior exposure to V1 and the prior V2 attempt.
- "Regime-aware models do not work." The study tests one model class, one universe, one frequency, and one implementation. The correct statement is about this protocol.
- "The result implies live performance." It does not.

---

## 6. Publication readiness

### Ready

A one-page research note with the thesis:

> **A small signal is not a strategy.** A probabilistic MoE found a small full-sample cross-sectional signal, but it did not survive the holdout or the pre-committed cost. 1/N remained ahead over the full sample and in every subperiod. The result is a benchmark for evaluating regime-aware factor models, not a claim of alpha.

### Not ready

- A positive alpha claim.
- A "gross signal in the holdout" story.
- A 5 bps cherry-pick.
- A live-trading or capacity claim.
- A general statement about all regime models.

### Required corrections before publication

- [x] Recompute holdout cost sensitivity and gross/IC uncertainty.
- [x] Document the gate in this file.
- [x] Correct README, `docs/09`, and `docs/10` to distinguish full-sample vs holdout IC.
- [ ] Ensure the one-pager shows the 5/10/20 bps crossover and the holdout IC insignificance.
- [ ] Use "pre-committed", not "pre-registered", in public-facing copy.
- [ ] Disclose the single-initialization and reconstructed-data limitations in the one-pager footer.

---

## 7. Reproduce the red-team checks

```bash
python scripts/red_team_checks.py --run-dir runs/reference
```

Outputs:

- `reports/red_team_checks.json`
- `reports/red_team_cost_sensitivity.csv`

The checks are deterministic for the saved run artifacts. They do not refit the model.

---

## 8. Sign-off

| Question | Answer |
|---|---|
| Is there a defensible contribution? | Yes: a rigorous negative/inconclusive benchmark and a signal-vs-decision-value lesson. |
| Is the headline claim positive? | No. |
| Is the holdout clean enough for a confirmatory claim? | Inconclusive; underpowered and retrospective. |
| Are the critical weaknesses disclosed? | Yes, in this document and the report limitations. |
| Is the one-pager allowed to present a gross-alpha story? | No. |
| Is the one-pager allowed to present a cost-sensitivity story? | Yes, with the pre-committed 10 bps result as primary and the 5 bps result labeled within-noise. |
| Publication gate | **Pass with corrected claims.** |
