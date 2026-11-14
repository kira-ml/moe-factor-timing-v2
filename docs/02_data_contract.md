# 02 - Data Contract

**Status:** Frozen before the reference run
**Snapshot:** local-only `data/reference/factors.csv` (not redistributed)
**SHA-256:** recorded in `data/reference/metadata.json`

---

## 1. Source

The reference panel is the Ken French Data Library:

- `F-F_Research_Data_5_Factors_2x3_CSV.zip`
- `F-F_Momentum_Factor_CSV.zip`

The five-factor file provides the market, size, value, profitability, and investment factors plus the one-month Treasury bill rate. The momentum file provides the momentum factor. Both files are monthly, value-weighted, and reconstructed by CRSP.

**Citation.** Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56. Data: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

---

## 2. Why this panel

Version 1 used five long-only US equity ETFs plus the VIX spot index. That universe had three problems:

1. **Not enough history.** MTUM, QUAL, and USMV launched around 2013, leaving 155 months and 42 headline out-of-sample predictions.
2. **Not factor-pure.** Five long-only US equity ETFs share a large common market exposure. The exercise was closer to equity-beta timing than factor timing.
3. **Not investable.** VIX is a spot index, not a tradable asset. Its monthly percentage change is not a portfolio return.

The Ken French panel is a standard research benchmark with a long history and genuine factor spreads. It is not a live trading record, but it is a defensible panel for testing a conditional allocation model.

---

## 3. Schema

| Column | Definition | Units | Sign |
|---|---|---|---|
| `MKT_RF` | Market excess return | Decimal per month | Long-short |
| `SMB` | Small-minus-big size factor | Decimal per month | Long-short |
| `HML` | High-minus-low value factor | Decimal per month | Long-short |
| `RMW` | Robust-minus-weak profitability factor | Decimal per month | Long-short |
| `CMA` | Conservative-minus-aggressive investment factor | Decimal per month | Long-short |
| `MOM` | Momentum factor (12-2 prior return) | Decimal per month | Long-short |
| `RF` | One-month Treasury bill return | Decimal per month | Cash |

- `0.01` means 1%.
- The index is month-end. A row dated `2020-01-31` contains the return for January 2020.
- The panel is continuous monthly from 1963-07 to 2026-09 (759 observations at the time of the snapshot).
- There are no missing values in the frozen snapshot.

---

## 4. Point-in-time contract

The contract is deliberately simple and testable:

| Object | Uses information through | Predicts |
|---|---|---|
| Feature row `X_t` | Month `t` (returns and realised volatility) | - |
| Target `y_t` | - | Month `t + 1` |
| Decision at `t` | `X_t` | Position held over `t + 1` |
| Realised return | `y_t` | Evaluated at the end of `t + 1` |

At the close of month `t`, the investor observes the factor returns for month `t` and all earlier months. The feature row uses those returns. The target is the next month's return vector. There is no gap and no future information.

The contract is enforced by:

- `tests/test_features.py::test_future_perturbation_does_not_change_features`
- `tests/test_features.py::test_truncation_does_not_change_features`
- the runtime assertion in `backtest/engine.py` that `y_t` equals the market return at `target_date`.

### What is not point-in-time

The Ken French files are reconstructed from the current CRSP database. They are not a literal vintage snapshot: historical returns can reflect later data corrections, and the factor definitions are the current ones. This is a **research-panel caveat**, not a hidden look-ahead in the feature/target alignment. It is disclosed in the report and the risk register.

The main analysis deliberately excludes revised macroeconomic series. V1 used FRED latest-plus-lag, which does not reconstruct what was known at the time. A vintage-aware macro extension is future work, not part of this benchmark.

---

## 5. Validation checks

`data/validate.py` runs the following checks before any model sees the data:

| Check | Failure condition |
|---|---|
| `columns_present` | Required factor or risk-free columns missing |
| `datetime_index` | Index is not a `DatetimeIndex` |
| `monotonic_unique` | Dates are unsorted or duplicated |
| `continuous_monthly` | A calendar month is missing |
| `month_end_dates` | Observations are not month-end |
| `minimum_history` | Fewer than 240 monthly observations |
| `no_missing_values` | Any NaN in factors or risk-free |
| `finite_values` | Any infinite value |
| `risk_free_plausible` | Any monthly risk-free return with absolute value >= 10% |
| `non_degenerate_factors` | A factor has zero variance |
| `no_implausible_extremes` | Any monthly factor return with absolute value > 100% |
| `risk_free_aligned` | Risk-free series does not share the factor index |

A failed check stops the run. The validation report is written to `runs/<id>/validation.json`.

---

## 6. Snapshot and refresh

The panel is downloaded to `data/reference/factors.csv` locally. The raw CSV is **not committed** to the public git tree because the source does not state an explicit redistribution license. The repository ships provenance in `data/reference/metadata.json` and the downloader.

To fetch it:

```bash
python scripts/fetch_reference_data.py
```

The refresh is intentionally a separate, explicit step. A research run never downloads data silently; it uses the local panel unless the config points elsewhere.

The snapshot's SHA-256 is recorded in:

- `data/reference/metadata.json`
- every local run manifest's `data.file_sha256`

**Redistribution status:** unresolved. The Ken French page displays a copyright notice but no explicit open license. Do not redistribute the raw file without confirming terms or obtaining permission. See `docs/12_legal_ethical_security_review.md`.
- the run's copied `data/factors.csv` hash

---

## 7. Limitations

1. **Reconstructed panel.** Not a literal vintage or live trading record.
2. **Long-short research portfolios.** The allocation is a fund-of-factor-strategies interpretation. Internal factor-portfolio turnover, financing, and shorting costs are not modelled.
3. **Fixed, well-known universe.** Publication and crowding effects are not modelled.
4. **Monthly frequency.** No daily rebalancing, intraday execution, or market impact.
5. **US only.** No international, credit, commodity, or currency factors.
6. **No macro vintages.** The main analysis avoids macro features rather than pretending to have point-in-time macro data.

These limitations are restated in the report. They are not hidden in an appendix.
