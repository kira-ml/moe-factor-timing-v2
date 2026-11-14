# Reference Data

The research run uses a local panel downloaded from the Ken French Data Library. The raw CSV is **not redistributed** in the public git tree; the repository ships a downloader and provenance metadata.

---

## Files

| File | Description |
|---|---|
| `reference/factors.csv` | Local-only monthly factor excess returns and the risk-free rate (decimal units); not committed |
| `reference/metadata.json` | Source URLs, retrieval time, schema, row count, date range, SHA-256 |

---

## Quick start

```bash
python scripts/fetch_reference_data.py
```

The fetch is explicit. A research run never downloads data silently.

---

## Schema

| Column | Definition |
|---|---|
| `MKT_RF` | Market excess return |
| `SMB` | Small-minus-big size factor |
| `HML` | High-minus-low value factor |
| `RMW` | Robust-minus-weak profitability factor |
| `CMA` | Conservative-minus-aggressive investment factor |
| `MOM` | Momentum factor (12-2 prior return) |
| `RF` | One-month Treasury bill return |

- Frequency: monthly.
- Units: decimal (`0.01` = 1%).
- Index: month-end.
- Coverage: 1963-07 to 2026-09 (759 observations at the snapshot).
- No missing values in the snapshot.

---

## Source

Ken French Data Library:

- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip
- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip

The five-factor file supplies `MKT_RF`, `SMB`, `HML`, `RMW`, `CMA`, and `RF`. The momentum file supplies `MOM`.

**Citation:** Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

### Redistribution

The Ken French Data Library page displays a copyright notice but no explicit open-source or redistribution license. The repository therefore treats the raw CSV as **local-only**. Do not redistribute the file without confirming the source terms or obtaining permission. The derived statistics and figures in `reports/` and `publication/` are research outputs, not a redistribution of the raw panel. This is an unresolved legal item, not a legal opinion; see `docs/12_legal_ethical_security_review.md`.

---

## Refresh

```bash
python scripts/fetch_reference_data.py --force
```

The fetch is explicit. A research run never downloads data silently.

---

## Caveats

1. **Reconstructed panel.** The files are built from the current CRSP database. They are not a literal point-in-time vintage. Historical returns may reflect later corrections.
2. **Research portfolios.** The factors are long-short portfolios, not directly traded ETF returns. The allocation is a fund-of-factor-strategies interpretation.
3. **Internal turnover not modelled.** The cost model covers rebalancing across factor sleeves, not the turnover inside each long-short factor portfolio.
4. **Fixed universe.** The factors are well known; publication and crowding effects are not modelled.
5. **No macro data.** The main analysis deliberately excludes revised macroeconomic series. A vintage-aware macro extension is future work.

See `docs/02_data_contract.md` for the full contract.
