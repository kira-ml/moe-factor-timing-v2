# moe-factor-timing v2

**A reproducible benchmark for probabilistic regime-aware factor allocation.**

This repository asks a narrow, testable question: can a probabilistic mixture-of-experts model allocate across equity factor portfolios better than naive equal weight, after realistic costs, out of sample? It is built as a research benchmark, not a trading strategy.

---

## Result

<!-- RESULT_START -->
The mixture-of-experts model **does not beat 1/N at the pre-committed 10 bps cost**.

In the pre-committed 2006-2026 holdout, the MoE has a net Sharpe of **0.479** versus **0.505** for 1/N. The difference is **-0.026** with a block-bootstrap 95% confidence interval **[-0.470, 0.389]** (p = 0.862). The data cannot distinguish the two at the pre-committed confidence level; this is an **inconclusive** result, not evidence of equivalence.

Over the full 1974-2026 out-of-sample window, 1/N is ahead: **1.059** versus **0.790** for the MoE (difference -0.269; 95% CI [-0.518, 0.006]). 1/N also wins all three subperiods.

The gross/net decomposition is informative but is **not** a positive-alpha claim. The MoE's full-sample cross-sectional information coefficient is **0.057** (HAC t = 2.97), but the 2006-2026 holdout IC is **0.040** (t = 1.35), and the holdout gross Sharpe difference (0.622 vs 0.511) has a 95% CI **[-0.324, 0.514]**. The holdout gross edge is not statistically confirmed.

The cost sensitivity is the most important caveat. In the holdout, the MoE's net Sharpe is **0.551 at 5 bps**, **0.479 at 10 bps**, and **0.336 at 20 bps**, versus 1/N at 0.508, 0.505, and 0.498. The sign flips between 5 and 10 bps, but the underlying gross difference is within noise. The 5 bps result must not be presented as an edge. Over the full sample, even the gross Sharpe is below 1/N (0.934 vs 1.066).

The model is also miscalibrated in the tails: **10.9%** of PIT values fall in each 5% tail, and **0/6** assets pass the 5% KS threshold. Restart and decision-rule sensitivities do not change the ranking.

**Bottom line:** the MoE produced a small full-sample signal, but it did not survive the holdout or the pre-committed cost. 1/N remains hard to beat; the confirmatory holdout is inconclusive, not a win for either model.
<!-- RESULT_END -->

The full report, including confidence intervals, calibration, turnover, power, and limitations, is in [`reports/v2_report.md`](reports/v2_report.md).

---

## What this is

- A **long-sample** test: 759 monthly observations from 1963-07 to 2026-09.
- A **pre-committed protocol**: fixed hyperparameters, a 2006+ holdout, and no selection on the evaluation window.
- A **proper probabilistic MoE**: a mixture of linear Gaussian experts trained by EM, with a predictive mean and covariance.
- A **decision-focused evaluation**: net Sharpe, mean-variance utility, drawdown, turnover, cost drag, PIT calibration, block-bootstrap uncertainty, power, and a deflated Sharpe ratio.
- A **reproducible pipeline**: typed configs, contract tests, a deterministic solver, and a manifest with SHA-256 hashes.

## What this is not

- Not a trading strategy, a fund, or investment advice.
- Not a claim that regimes exist or that the latent components are economically identified.
- Not a claim that the Ken French research panel is a literal point-in-time trading record.
- Not a live-performance claim. Backtest results do not imply live results.
- Not a deep-learning benchmark. A neural MoE is a different hypothesis.

See [`docs/01_problem_framing.md`](docs/01_problem_framing.md) for the full scope and non-goals.

---

## Quick start

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

python -m pip install -e ".[dev]"

# Download the reference panel locally (not redistributed with the repository)
python scripts/fetch_reference_data.py

# Validate the data contract
python -m moe_factor_timing validate --config configs/default.yaml

# Run the reference pipeline
python -m moe_factor_timing run --config configs/default.yaml --run-id reference --publish

# Fast implementation smoke test (not a research result)
python -m moe_factor_timing run --config configs/smoke.yaml --run-id smoke

# Tests
python -m pytest tests/ -q

# Lint
python -m ruff check .
```

The Ken French factor panel is downloaded from the source and is **not redistributed** with the public repository. The repository ships a downloader, provenance metadata, and SHA-256 checksums. See `data/README.md` and `docs/12_legal_ethical_security_review.md`.

---

## Repository layout

```
configs/                  Reference and smoke configurations
data/reference/           Provenance metadata and downloader; raw CSV is local-only
docs/                     Problem framing, data contract, methodology, protocol, architecture, lessons, review
reports/                  Curated reference report and figures
runs/                     Immutable run artifacts (git-ignored)
scripts/                  Data refresh and analysis entry points
src/moe_factor_timing/    Installable package
tests/                    Contract, leakage, accounting, evaluation, and end-to-end tests
archive/v1/               Version 1 reference (local-only, read-only, fingerprinted)
DECISIONS.md              Material decisions and review dates
```

---

## Pipeline

```
ingest -> validate -> features -> split -> models -> decision -> backtest -> evaluate -> report -> manifest
```

Every boundary has a typed contract. The engine fails loudly on a model error, counts solver fallbacks, and refuses to publish a result if the fallback rate exceeds 5%. The manifest binds the result to its config, data snapshot, code commit, package versions, and artifact hashes.

---

## Key documents

| Document | Purpose |
|---|---|
| [`docs/01_problem_framing.md`](docs/01_problem_framing.md) | Research question, hypotheses, scope, falsification |
| [`docs/02_data_contract.md`](docs/02_data_contract.md) | Data source, schema, point-in-time contract, caveats |
| [`docs/03_methodology.md`](docs/03_methodology.md) | Features, models, decision rule, costs, walk-forward |
| [`docs/04_evaluation_protocol.md`](docs/04_evaluation_protocol.md) | Endpoints, uncertainty, power, multiple testing, kill criteria |
| [`docs/05_architecture.md`](docs/05_architecture.md) | Module map, interfaces, invariants, extension points |
| [`docs/06_lessons_from_v1.md`](docs/06_lessons_from_v1.md) | Diagnosis of Version 1 and what V2 changes |
| [`docs/07_reproducibility.md`](docs/07_reproducibility.md) | Reproduce, verify, and audit a run |
| [`docs/08_risk_register.md`](docs/08_risk_register.md) | Pre-mortem, risks, counter-metrics, rollback |
| [`docs/09_sensitivity.md`](docs/09_sensitivity.md) | Restart audit and decision-rule sensitivity |
| [`docs/10_reference_run.md`](docs/10_reference_run.md) | Reference run manifest and verification record |
| [`docs/11_red_team_validation.md`](docs/11_red_team_validation.md) | Publication gate and red-team findings |
| [`docs/12_legal_ethical_security_review.md`](docs/12_legal_ethical_security_review.md) | Data, licensing, font, secrets, and publication-safety review |

---

## Limitations

- The Ken French panel is reconstructed from the current CRSP database, not a literal vintage.
- The allocation is a fund-of-factor-strategies interpretation. Internal turnover inside each long-short factor portfolio is not modelled.
- Long-only, fully invested, monthly frequency, US equity factors only.
- The holdout is 20 years; the study is underpowered for small Sharpe improvements. The report states the minimum detectable effect.
- No live trading, execution, market impact, or capacity estimate.
- The raw Ken French panel and the V1 archive are **not redistributed** in the public git tree because their redistribution terms are not explicitly documented. They remain local-only reference material. See `docs/12_legal_ethical_security_review.md`.

The full list is in the report and `docs/08_risk_register.md`.

---

## Version 1 archive

`archive/v1/` contains the complete Version 1 implementation and outputs. It is **read-only reference material** and is never modified. It is **kept local-only and excluded from the public git tree** because it contains third-party data with unresolved redistribution terms. Its local fingerprint is checked by `tests/test_manifest.py` (the test skips on a public clone without the archive) and recorded in every local run manifest.

The prior V2 attempt is preserved in the private development repository and is **not part of this public release**.

> **History note.** Earlier public commits still contain the Ken French snapshot and the V1 archive. Removing them from the current tree does not remove them from git history. A fully clean public release requires a history rewrite or a fresh repository. See `docs/12_legal_ethical_security_review.md`.

---

## Citation

If you use this benchmark, please cite the Fama-French data source and this repository:

```
Talingting, Ken Ira Lacson (2026). moe-factor-timing v2: A reproducible benchmark for
probabilistic regime-aware factor allocation. GitHub repository.
```

Factor data: Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

---

## Legal and data notice

The original code and documentation are MIT-licensed. The raw Ken French panel and the V1 archive are **not redistributed** in the public tree because their source terms do not explicitly grant redistribution. The archive remains local-only and byte-identical. The public git history still contains earlier data snapshots; a fully clean release requires a fresh repository or a history rewrite. See [`NOTICE`](NOTICE), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and [`docs/12_legal_ethical_security_review.md`](docs/12_legal_ethical_security_review.md). This is not legal advice.

---

## License

MIT. See [`LICENSE`](LICENSE).
