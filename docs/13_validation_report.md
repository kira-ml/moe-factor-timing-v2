# 13 - Independent Validation Report

**Status:** Validation gate
**Date:** 2026-11-16
**Scope:** the `moe-factor-timing v2` implementation, reference run, reported numbers, and public-release tree
**Reviewer stance:** independent reproducibility audit, not a re-analysis of the hypothesis

---

## 1. Verdict

**No unresolved critical issue was found.** The implementation, reference run, and public-release tree are consistent with the reported negative/inconclusive result. The project is complete for the scope it claims: a reproducible benchmark, not a live trading strategy.

The validation found **documentation precision issues**, not result errors:

1. `docs/11` still had three unchecked publication corrections that were already satisfied.
2. `docs/12` described the old public-history risk as unresolved; the public release now uses a fresh repository with clean history.
3. The one-pager uses 40.6%/0.97% (holdout turnover/cost drag) and "759 monthly observations" near the full-sample IC (627 OOS months) without an explicit label. This is a labeling ambiguity, not a numerical error.
4. `manifest._git` records `git_dirty = false` if git is unavailable. In a normal repository checkout this is not triggered; it is a minor robustness edge case.
5. An unmerged branch, `improvements/practical-tweaks`, contains an older `min_train=96` paper generator and TODO. It is **not** part of `main` or the public release and must not be used for the paper.

Items 1 and 2 were corrected in this validation pass. Items 3–5 are disclosed below.

---

## 2. Checks performed

### 2.1 Static checks

```bash
python -m ruff check src tests scripts
python -m pytest tests/ -q
```

- Ruff: **all checks passed**.
- Pytest: **45 passed**.

### 2.2 Data contract

```bash
python -m moe_factor_timing validate --config configs/default.yaml
```

- **12/12** checks passed: columns, DatetimeIndex, monotonicity, monthly continuity, month-end dates, minimum history, missing values, finite values, risk-free plausibility, non-degenerate factors, extreme returns, and risk-free alignment.
- Sample: **759 monthly observations**, 1963-07-31 to 2026-09-30.

### 2.3 Fresh reference run

```bash
python -m moe_factor_timing run --config configs/default.yaml --run-id validation_reference
```

- Run completed successfully at HEAD `1e42680`.
- Config hash: `2a035de2d1a091455b816690492ae20a6bef2c1936ebffd0e51ed4f4c3daaa5f`.
- Data SHA-256: `cc805ab0e62e8280b747febbe047cf85212c5d1fd36a7a476074350d734cf275`.
- Splits: **627** decisions, **248** holdout, first `1974-06-30`, last `2026-08-31`.
- Manifest: `git_dirty = false`.

### 2.4 Artifact reproducibility

A fresh run was compared with the curated `runs/reference` run across **41 artifacts**:

- all seven models' returns and weights;
- static-MV, ridge, and MoE forecast means, covariances, PIT, NLL, and gating;
- features and target dates;
- metrics, holdout metrics, bootstrap tables, calibration, cost sensitivity, subperiods, deflated Sharpe, and splits;
- validation output.

**Result: 41/41 identical.** No numerical drift, sign change, or ranking change.

### 2.5 Independent recomputation

A separate audit script recomputed the headline metrics from raw weights, returns, forecasts, and targets rather than calling the project's evaluation helpers. It covered:

- Sharpe, gross Sharpe, annual return, volatility, drawdown, utility, turnover, and cost drag for all seven models;
- the accounting identities `net = gross − cost` and `net_total = rf + net_excess`;
- drift-aware turnover for every model;
- gross returns from weights and realised targets;
- weight non-negativity, caps, and sum constraints;
- block-bootstrap Sharpe differences for full sample and holdout;
- full/holdout/pre-holdout IC, HAC t-statistics, and block-bootstrap intervals;
- PIT means, tails, pooled KS, and per-asset KS;
- subperiod Sharpes and cost sensitivity at 5/10/20/30 bps;
- manifest artifact hashes and data SHA-256.

**Result: 242/242 checks passed.**

### 2.6 Public-release audit

- Public repository: [`kira-ml/moe-factor-timing-v2`](https://github.com/kira-ml/moe-factor-timing-v2).
- **121 tracked files at the time of the audit**; the release surface adds CI, a notebook, an extension guide, and the paper.
- No `archive/`, no `data/reference/factors.csv`, no `runs/`.
- The raw Ken French panel is downloaded locally, not redistributed.
- The legacy repository `kira-ml/mixture-of-experts-factor-timing` returns 404 to unauthenticated requests (private).
- The public README, `CITATION.cff`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, and `docs/12` are present.
- The public tree contains no absolute local paths or credentials. The only `FRED_API_KEY` mention is a statement that the local `.env` is untracked.
- The v2.0.0 release is archived on Zenodo: [10.5281/zenodo.24280434](https://doi.org/10.5281/zenodo.24280434).

---

## 3. Reference result confirmed

| Endpoint | Result |
|---|---:|
| Full OOS net Sharpe, 1/N | **1.059** |
| Full OOS net Sharpe, MoE | **0.790** |
| Full OOS difference | **−0.269** (95% CI `[−0.518, 0.006]`) |
| Holdout net Sharpe, 1/N | **0.505** |
| Holdout net Sharpe, MoE | **0.479** |
| Holdout difference | **−0.026** (95% CI `[−0.470, 0.389]`; p = 0.862) |
| Full OOS IC | **0.057** (HAC t = 2.97; 627 months) |
| Holdout IC | **0.040** (HAC t = 1.35; 248 months) |
| Holdout gross Sharpe, 1/N vs MoE | **0.511 vs 0.622**; difference CI `[−0.324, 0.514]` |
| Holdout MoE net Sharpe at 5/10/20 bps | **0.551 / 0.479 / 0.336** |
| MoE holdout turnover / cost drag | **40.6% / 0.97% per year** |
| MoE PIT tails / assets passing KS | **10.9% each / 0 of 6** |

The holdout is **inconclusive**, not equivalent. The full OOS result favors 1/N. The MoE’s holdout gross point estimate is above 1/N, but the difference is not statistically confirmed, and costs erase the net advantage.

---

## 4. Issues and disposition

| # | Severity | Issue | Disposition |
|---|---|---|---|
| 1 | Minor | `docs/11` publication checklist was stale. | Corrected; the one-pager already satisfies the items. |
| 2 | Minor | `docs/12` described the old public-history risk as unresolved. | Corrected to record the fresh public repository and private legacy repository. |
| 3 | Minor | One-pager labels 40.6%/0.97% and “759 monthly observations” near full-sample IC without an explicit window label. | Disclosed; the paper uses 627 OOS months and labels holdout turnover/cost drag. |
| 4 | Minor | `manifest._git` falls back to `git_dirty = false` if git is unavailable. | Disclosed; not triggered in a normal checkout. |
| 5 | Process | Unmerged `improvements/practical-tweaks` branch contains an older `min_train=96` paper generator and TODO. | Excluded from the paper; paper uses only `main`/public release and `runs/reference`. |
| 6 | Process | No CI workflow; tests are run locally. | Disclosed; reproducible commands are in `docs/07` and this report. |

No high or critical severity issue was found.

---

## 5. Residual limitations (not defects)

These are part of the research design and are disclosed in the report and README:

- The Ken French panel is reconstructed from the current CRSP database, not a literal vintage.
- The allocation is a fund-of-factor-strategies interpretation; internal sleeve turnover and financing are not modelled.
- The study is long-only, fully invested, monthly, and US-only.
- The holdout is underpowered for small Sharpe improvements; MDE ≈ 0.62.
- The reference MoE uses one deterministic initialization; the restart audit shows local-optimum sensitivity.
- The deflated Sharpe is non-discriminating because all models have high PSR/DSR against zero; the paired bootstrap versus 1/N is the primary comparison.
- No live trading, execution, market impact, capacity, or shorting/leverage analysis.

---

## 6. Validation artifacts

Local audit folder (not part of the public repository):

- `validation_checks.py` — independent recomputation.
- `validation_checks.json` — 242 checks with pass/fail details.
- `validation_checks.md` — summary.
- `compare_runs.py` — fresh-run vs reference-run comparison.
- `reproducibility_comparison.csv` — 41 artifact comparisons.
- `WORKLOG.md` — validation state and findings.

The reference run is in `runs/reference`; the fresh validation run is in `runs/validation_reference`. Both are git-ignored.

---

## 7. Verdict for paper development

**Proceed.** The implementation and reported results are sufficiently verified for a formal research paper. The paper must:

- use the current `runs/reference` and `reports/v2_report.md` numbers, not the obsolete `min_train=96` branch;
- label the full OOS window (1974–2026, 627 months), holdout (2006–2026, 248 months), and raw panel (1963–2026, 759 months) separately;
- describe the result as negative/inconclusive, not as positive alpha;
- disclose calibration failure, cost sensitivity, single-initialization sensitivity, and reconstructed-data caveats;
- cite only verifiable references and claim only the contributions the repository supports.
