# Decision Log

**Project:** moe-factor-timing v2
**Rule:** Record material, hard-to-reverse decisions before acting. Record the outcome after the reference run.

---

## D1 - Use the long-sample Ken French panel instead of the 2013 ETF panel

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Keep the five ETFs + VIX from V1 (155 months, 42 headline OOS months).
  2. Use Ken French five factors + momentum from 1963 (759 months).
  3. Use a different asset class with long history.
- **Decision:** Option 2.
- **Rationale:** The binding constraint in V1 was statistical power and investability, not model sophistication. The Ken French panel gives a 50-year full OOS window and a 20-year holdout, and removes VIX. It is a standard research benchmark.
- **Assumptions:** The panel is a valid research panel; the reconstruction caveat is disclosed; the allocation is a fund-of-factor-strategies interpretation.
- **Expected outcome:** Enough power to detect moderate effects, and a credible negative or positive result.
- **Review date:** 2027-06-30
- **Outcome:** Executed. 759 monthly observations; 627 out-of-sample decisions; 248-month holdout. The long sample supports a full-window minimum detectable Sharpe of 0.39; the holdout MDE is 0.62. See `docs/10_reference_run.md`.

---

## D2 - Long-only, fully invested fund-of-factor-strategies

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Long-short factor weights (short a factor when its forecast is negative).
  2. Long-only weights across long-short factor sleeves, fully invested.
  3. Long-only with cash allowed.
- **Decision:** Option 2 for the primary protocol; cash-allowed is a robustness variant.
- **Rationale:** Factor timing is a relative allocation problem. Cash would confound factor timing with market timing. Long-short weights would require a different financing and cost model. A fund-of-factor-strategies is transparent and directly comparable to 1/N.
- **Assumptions:** The factor sleeves are investable in principle; internal sleeve turnover is not modelled and is disclosed.
- **Expected outcome:** A clean comparison of relative allocation skill.
- **Review date:** 2027-06-30
- **Outcome:** Executed. 1/N, inverse vol, momentum, cash, static MV, ridge, and MoE all use the same long-only, fully invested frame; 1/N is the primary benchmark. See `docs/10_reference_run.md`.

---

## D3 - Fix hyperparameters a priori and reserve a 2006+ holdout

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Tune K, min_train, and the decision rule on the full OOS window.
  2. Fix them from theory and V1 lessons, reserve a holdout, do not tune on it.
  3. Use a small validation window inside training for tuning.
- **Decision:** Option 2 for the primary run; option 3 is allowed only inside the training window in future work.
- **Rationale:** V1's headline was selected on the evaluation window. A holdout that is never used for selection is the cheapest credible fix. The hyperparameters are few and theory-motivated.
- **Assumptions:** The fixed values are reasonable; robustness will show whether the conclusion is sensitive.
- **Expected outcome:** A result that cannot be attributed to post-hoc tuning.
- **Review date:** 2027-06-30
- **Outcome:** Executed. Reference config hash `2a035de...`; no holdout tuning. The holdout result is inconclusive; the full-sample result is negative. See `docs/10_reference_run.md`.

---
## D4 - Use a proper EM mixture of linear Gaussian experts, not a neural MoE

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Reuse V1's `SimpleMoE` heuristic.
  2. Implement a proper EM mixture with a predictive distribution.
  3. Implement a PyTorch/LSTM MoE.
- **Decision:** Option 2.
- **Rationale:** The hypothesis is about soft regime membership in conditional means, not about deep representation learning. A linear-Gaussian mixture has a tractable likelihood and covariance, can be estimated on 120-700 observations, and supports calibration tests. It is one implementation, not two.
- **Assumptions:** The linear-Gaussian class is expressive enough to test the hypothesis; a neural MoE is a separate study.
- **Expected outcome:** A calibrated or diagnosably miscalibrated predictive distribution and a fair test of the mechanism.
- **Review date:** 2027-06-30
- **Outcome:** Executed. MoE fit by EM; pooled PIT KS p<0.001; 10.9% of PIT values in each 5% tail; 0/6 assets pass. The model has a small gross IC (0.058, t=2.89) but no net edge.

---
## D5 - Drift-aware turnover and cost accounting

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Measure turnover against last month's target weights (V1).
  2. Measure turnover against the drifted pre-rebalance weights.
- **Decision:** Option 2.
- **Rationale:** Portfolio weights drift with returns. Target-to-target turnover understates the trade required to rebalance, and V1's per-row cost function charged nothing at all. Drift-aware accounting is the correct identity.
- **Assumptions:** The factor sleeves earn `rf + excess`; the cash residual earns `rf`; the cost is one-way per unit traded.
- **Expected outcome:** Higher, more realistic turnover for 1/N and all models; a cost drag that can change rankings.
- **Review date:** 2027-06-30
- **Outcome:** Executed. 1/N turnover 1.1%/month; MoE 40.6%/month; MoE cost drag 0.97%/year. Drift-aware accounting changes the ranking versus naive target-to-target turnover.

---
## D6 - Primary metric: net Sharpe; secondary: mean-variance utility

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Sharpe ratio.
  2. Mean-variance utility at a fixed lambda.
  3. Both, with one pre-committed primary.
- **Decision:** Option 3, with net Sharpe as the primary and utility as secondary.
- **Rationale:** Sharpe is scale-invariant and comparable across models with different risk levels. Utility depends on lambda and can be gamed by cash or concentration. Reporting both, with Sharpe primary, avoids a single-metric deployment.
- **Assumptions:** The fully invested constraint removes the cash confound; lambda=2 is a reasonable secondary.
- **Expected outcome:** A primary result that is not an artifact of risk aversion.
- **Review date:** 2027-06-30
- **Outcome:** Executed. Net Sharpe primary: MoE holdout 0.479 vs 1/N 0.505 (inconclusive); full sample 0.790 vs 1.059. Utility is higher for MoE because it takes more risk, which is why Sharpe was primary.

---
## D7 - Rebuild the working tree; preserve prior V2 in the private development repository; never modify `archive/`

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Incrementally patch the prior V2 code.
  2. Delete the prior V2 working tree and rebuild from scratch.
  3. Keep both implementations side by side.
- **Decision:** Option 2, with prior V2 preserved in the private development repository and V1 preserved locally in `archive/v1/`. Neither is part of the public release.
- **Rationale:** The prior V2 had a different research question and a different architecture. Incremental patching would carry its assumptions and its code. A clean rebuild with a frozen protocol is more credible. Git history makes the deletion reversible.
- **Assumptions:** The prior V2 is recoverable from tags; the archive fingerprint test guards V1.
- **Expected outcome:** A coherent V2 with one source of truth.
- **Review date:** 2027-06-30
- **Outcome:** Executed. Archive fingerprint `716ff99a5cc32c63971b668bb5dc6eb0909b6fce820fafa8d1f9c1fdb9d4a058` (183 files); `tests/test_manifest.py` guards it locally. Prior V2 is preserved in the private development repository; neither the archive nor the prior tree is part of the public release.

---

## D8 - No revised macro features in the main analysis

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Use FRED latest + lag, as V1 did.
  2. Use ALFRED vintages.
  3. Exclude macro features from the main analysis.
- **Decision:** Option 3; option 2 is future work.
- **Rationale:** V1's macro features were not point-in-time. Rather than hide revisions behind a lag, the main analysis uses only lagged returns and realised volatility, which are observable at the decision date. This reduces the feature count and removes a class of look-ahead risk.
- **Assumptions:** The return/volatility feature set is sufficient to test the mechanism; macro information is not the only state signal.
- **Expected outcome:** A cleaner causal contract and a more parsimonious model.
- **Review date:** 2027-06-30
- **Outcome:** Executed. 30 features, no macro; max abs feature-target correlation is below the 0.95 flag; future-perturbation and truncation tests pass.

---
## D9 - Deterministic SLSQP instead of cvxpy/OSQP

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. cvxpy with OSQP, as the prior V2 attempt used.
  2. scipy SLSQP with analytic constraints.
  3. A custom projected gradient solver.
- **Decision:** Option 2.
- **Rationale:** OSQP showed run-to-run drift in the prior V2 attempt. SLSQP is deterministic for a fixed problem and starting point, has no extra dependency, and the problem is small (six assets, linear constraints). A fallback path is recorded and gated.
- **Assumptions:** SLSQP finds the global optimum for this convex QP; the fallback rate is monitored.
- **Expected outcome:** Byte-stable decisions across repeated runs.
- **Review date:** 2027-06-30
- **Outcome:** Executed. SLSQP fallback rate 0.0 across 627 rebalances x 7 models; the end-to-end determinism test passes.

---
## D10 - One MoE restart for the reference run; a separate restart audit

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Many restarts per walk-forward fit (robust but slow).
  2. One deterministic k-means initialisation per fit, with a separate stability audit.
- **Decision:** Option 2 for the reference run.
- **Rationale:** The full walk-forward refits the model 627 times. Multiple restarts per fit multiply runtime by 3-5x. A deterministic k-means initialisation plus a separate audit of restart sensitivity gives most of the robustness at a fraction of the cost.
- **Assumptions:** The single-initialisation solution is representative; the audit will check this.
- **Expected outcome:** A reference run that completes in minutes, with a documented stability check.
- **Review date:** 2027-06-30
- **Outcome:** Executed. 24-window audit: restarts improve training likelihood in 71% of windows but test NLL in only 33%. One restart is a compute compromise, disclosed as a limitation.

---

## D11 - Red-team the result before publicizing it

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Publish the one-pager from the headline net result.
  2. Publish the full-sample negative result without further validation.
  3. Run a pre-publication red-team gate on cost sensitivity, holdout gross/IC significance, initialization, and data vintage.
- **Decision:** Option 3.
- **Rationale:** The holdout result is inconclusive, and a publication that oversells a gross signal would repeat V1's error in a more sophisticated form. The gate must distinguish validated findings from promising observations.
- **Assumptions:** The saved artifacts are sufficient for the checks; no refit is required.
- **Expected outcome:** A corrected thesis and a publication that is defensible under adversarial review.
- **Review date:** 2027-06-30
- **Outcome:** Executed. The gate found that the holdout gross edge and holdout IC are not statistically confirmed, and the holdout net sign flips between 5 and 10 bps. Publication is approved only as a negative/inconclusive benchmark with corrected claims. See `docs/11_red_team_validation.md`.

---

## D12 - Remove third-party data from the public tree; keep the archive local-only

- **Date:** 2026-11-14
- **Status:** Decided
- **Options:**
  1. Publish the repository as-is, including the Ken French CSV and V1 archive.
  2. Remove the raw data and archive from the current public tree, keep them local, and document the residual history risk.
  3. Rewrite history or create a fresh public repository.
- **Decision:** Option 2 now; prepare a clean orphan `public-release` branch for Option 3. Do not force-push or publish the link until the user chooses the history remediation.
- **Rationale:** The Ken French page has no explicit redistribution license, and the V1 archive contains third-party ETF/VIX/FRED data. Removing them from HEAD reduces current exposure without deleting the local reference or modifying the archive contents.
- **Assumptions:** The archive remains byte-identical locally; the fingerprint test skips when the archive is absent from a clean clone; the downloader can restore the Ken French panel.
- **Expected outcome:** A clean HEAD and a ready-to-push public snapshot, with the history risk disclosed.
- **Review date:** 2027-06-30
- **Outcome:** Executed and superseded by a fresh public repository. `kira-ml/moe-factor-timing-v2` is public with clean history (121 tracked files at the release audit, no archive/data/runs); the legacy repository is private. The public link now points to the clean repository. See `docs/12_legal_ethical_security_review.md` and `docs/13_validation_report.md`.

---

## D13 - Release scope: polish and distribute; do not build a platform yet

- **Date:** 2026-11-16
- **Status:** Decided
- **Options:**
  1. Freeze the current release.
  2. Spend 25–35 hours on CI, reproducibility, documentation, a preprint, and targeted feedback.
  3. Run the `min_train`/`K` robustness grid and update the paper.
  4. Build a multi-asset benchmark platform or leaderboard.
  5. Aggressive promotion.
- **Decision:** Option 2. Defer options 3 and 4 until there is external demand or a collaborator.
- **Rationale:** The binding constraint is distribution and usability, not model capability. The platform option is dominated under the decision memo's expected-utility model (200 hours, maintenance burden, low probability of strong adoption for a solo maintainer). Option 2 has bounded downside and open upside.
- **Assumptions:** Demand estimates are subjective; the 35-hour cap is binding; no raw data is redistributed; the paper is an independent working paper, not a venue submission.
- **Expected outcome:** A usable, citable release; measurable feedback; a freeze if demand is weak.
- **Review date:** 2026-12-07
- **Outcome:** In progress. Week 1 implements CI, pinned constraints, quickstart notebook, extension guide, paper packaging, and release metadata.

---

## Template for future decisions

```
## D<number> - <title>
- Date:
- Status: Proposed | Decided | Superseded
- Options:
- Decision:
- Rationale:
- Assumptions:
- Expected outcome:
- Review date:
- Outcome:
```
