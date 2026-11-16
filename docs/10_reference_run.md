# 10 - Reference Run Record

**Run ID:** `reference`
**Code commit:** `aa5d16052cf4a07048c8343819ff846ccf55d52c`
**Working tree at manifest:** clean (`git_dirty = false`)
**Config:** `configs/default.yaml`
**Config hash:** `2a035de2d1a091455b816690492ae20a6bef2c1936ebffd0e51ed4f4c3daaa5f`
**Data snapshot SHA-256:** `cc805ab0e62e8280b747febbe047cf85212c5d1fd36a7a476074350d734cf275`
**Archive fingerprint:** `716ff99a5cc32c63971b668bb5dc6eb0909b6fce820fafa8d1f9c1fdb9d4a058` (183 files)
**Manifest:** `runs/reference/manifest.json`
**Artifacts hashed:** 79

> **Validation update (2026-11-16).** A fresh run at current HEAD `1e42680` with the same config and data reproduced **41/41 compared artifacts identically**, including returns, weights, forecasts, covariance arrays, bootstrap tables, calibration, and subperiods. The core model, backtest, and evaluation code is unchanged from the recorded commit. **242/242 independent checks passed.** See [`13_validation_report.md`](13_validation_report.md).

---

## 1. Design

| Item | Value |
|---|---|
| Universe | `MKT_RF`, `SMB`, `HML`, `RMW`, `CMA`, `MOM` |
| Sample | 1963-07-31 to 2026-09-30 (759 months) |
| Features | 30 lagged return and realised-volatility features |
| Walk-forward decisions | 627 |
| Minimum training | 120 months |
| Holdout | 2006-01-31 to 2026-08-31 (248 decisions) |
| Decision rule | constrained mean-variance, fully invested, max weight 0.35 |
| Risk aversion | 2.0 |
| One-way cost | 10 bps |
| Solver fallback rate | 0.0 |

---

## 2. Headline results

### Full out-of-sample

| Model | Net Sharpe | Gross Sharpe | Ann. return | Ann. vol | Max DD | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| 1/N | 1.059 | 1.066 | 8.89% | 4.16% | -12.25% | 0.012 |
| Inverse vol | 1.005 | 1.026 | 8.12% | 3.65% | -7.36% | 0.032 |
| Momentum | 0.766 | 0.799 | 10.52% | 8.02% | -17.17% | 0.113 |
| Static MV | 0.874 | 0.878 | 9.81% | 6.12% | -25.60% | 0.010 |
| Ridge | 0.849 | 0.975 | 10.56% | 7.20% | -18.29% | 0.379 |
| MoE | 0.790 | 0.934 | 10.03% | 7.11% | -17.54% | 0.422 |

### Confirmatory holdout

| Model | Net Sharpe | Gross Sharpe | Ann. return | Ann. vol | Max DD | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| 1/N | 0.505 | 0.511 | 3.77% | 4.23% | -12.25% | 0.011 |
| Inverse vol | 0.498 | 0.518 | 3.55% | 3.82% | -7.36% | 0.032 |
| Momentum | 0.468 | 0.511 | 4.83% | 7.08% | -17.17% | 0.127 |
| Static MV | 0.509 | 0.513 | 4.78% | 6.32% | -25.60% | 0.011 |
| Ridge | 0.371 | 0.507 | 4.11% | 7.06% | -18.29% | 0.404 |
| MoE | 0.479 | 0.622 | 4.80% | 6.80% | -17.54% | 0.406 |

---

## 3. Primary comparison

**MoE vs 1/N, holdout net Sharpe:**

- Difference: **-0.026**
- Block-bootstrap 95% CI: **[-0.470, 0.389]**
- p-value: **0.862**
- Verdict: **inconclusive**. The data do not distinguish the MoE from 1/N at the pre-committed confidence level.

**Full out-of-sample:** difference **-0.269**, 95% CI **[-0.518, 0.006]**, p = **0.058**. The interval just includes zero.

The minimum detectable annualised Sharpe is **0.39** over the full window and **0.62** over the holdout. The holdout is not powered to detect small improvements.

The deflated Sharpe ratio is high for every model (PSR and DSR near 1.0) because the factor panel has a positive unconditional premium. It tests whether each model's Sharpe is distinguishable from zero after selection, not whether the MoE beats 1/N. The primary comparison remains the paired bootstrap in Section 3.

---

## 4. Calibration and signal

| Model | Mean NLL | Mean PIT | PIT < 5% | PIT > 95% | KS p (pooled) | Assets passing |
|---|---:|---:|---:|---:|---:|---:|
| Static MV | -12.65 | 0.496 | 4.9% | 5.1% | 0.0001 | 2/6 |
| Ridge | -12.08 | 0.498 | 6.3% | 6.1% | 0.1886 | 3/6 |
| MoE | -8.16 | 0.497 | 10.9% | 10.9% | 0.0000 | 0/6 |

The MoE is overconfident in the tails. Its full-sample cross-sectional information coefficient is **0.057** (HAC t = 2.97), but the 2006-2026 holdout IC is **0.040** (t = 1.35; block-bootstrap 95% CI [-0.020, 0.102]), so the holdout does not confirm the signal. The holdout gross Sharpe difference (0.622 vs 0.511) has a 95% CI **[-0.324, 0.514]** and p = 0.60; it is a descriptive observation, not a confirmed gross edge.

The cost sensitivity is the key caveat. In the holdout, the MoE net Sharpe is **0.551 at 5 bps**, **0.479 at 10 bps**, and **0.336 at 20 bps**, versus 1/N at 0.508, 0.505, and 0.498. The sign flips between 5 and 10 bps, but the underlying gross difference is within noise. The pre-committed 10 bps result is the primary one; the 5 bps result must not be presented as an edge.

---

## 5. Post-run diagnostics

| Artifact | Purpose |
|---|---|
| `reports/decision_sensitivity.csv` | Risk aversion 2/5/10, caps 0.25/0.35/0.50, covariance scales 1/2/4 |
| `reports/forecast_ic.csv` | Cross-sectional information coefficient by model |
| `reports/moe_stability.json` | One vs three restart audit on 24 training windows |
| `docs/09_sensitivity.md` | Interpretation of the diagnostics |

The diagnostics do not change the primary conclusion. The MoE remains below 1/N in every decision-rule variant tested. Additional restarts improve training likelihood but not reliably test NLL.

---

## 6. What was not tested

- `min_train` in {60, 120, 180} and `K` in {2, 3, 4} refits.
- Vintage-aware macroeconomic features.
- Internal turnover and financing of the long-short factor sleeves.
- Live trading, execution, market impact, or capacity.
- Short selling, leverage, or derivatives.

These are disclosed limitations, not silent omissions.

---

## 7. Reproduce

```bash
python -m pip install -e ".[dev]"
python -m moe_factor_timing validate --config configs/default.yaml
python -m moe_factor_timing run --config configs/default.yaml --run-id reference --publish
python -m pytest tests/ -q
```

The reference run is deterministic for the same environment and config. See `docs/07_reproducibility.md`.
