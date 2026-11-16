# Documentation Index

**Project:** moe-factor-timing v2
**Purpose:** a reproducible benchmark for probabilistic regime-aware factor allocation.

---

## Reading order

| # | Document | What it answers |
|---|---|---|
| 00 | `00_index.md` | What is here and how to read it |
| 01 | `01_problem_framing.md` | What question is being asked, and what would falsify it |
| 02 | `02_data_contract.md` | What data is used, at what time, and with what caveats |
| 03 | `03_methodology.md` | Features, models, decision rule, costs, walk-forward |
| 04 | `04_evaluation_protocol.md` | Pre-committed endpoints, uncertainty, power, kill criteria |
| 05 | `05_architecture.md` | Module contracts, data flow, invariants, extension points |
| 06 | `06_lessons_from_v1.md` | What V1 got wrong and what V2 changes |
| 07 | `07_reproducibility.md` | How to install, run, and verify the result |
| 08 | `08_risk_register.md` | Pre-mortem, risks, counter-metrics, rollback |
| 09 | `09_sensitivity.md` | Post-run restart audit and decision-rule sensitivity |
| 10 | `10_reference_run.md` | Reference run manifest, results, and verification record |
| 11 | `11_red_team_validation.md` | Publication gate: what survives, what does not, and required corrections |
| 12 | `12_legal_ethical_security_review.md` | Data licensing, fonts, secrets, and public-history review |
| 13 | `13_validation_report.md` | Independent reproducibility audit, 242 checks, public-release verification |
| 14 | `14_extension_guide.md` | How to add a model to the registry and evaluate it |
| - | `../CONTRIBUTING.md` | Development checks and contribution rules |
| - | `../paper/` | Research paper PDF and citation record |
| - | `../DECISIONS.md` | Material decisions, assumptions, review dates |
| - | `../README.md` | Project overview and headline result |
| - | `../reports/v2_report.md` | Generated reference report |
| - | `../archive/v1/` | Read-only Version 1 reference (never modified) |

---

## Status

| Item | Status |
|---|---|
| V1 archive | Frozen, fingerprinted, read-only; local-only, excluded from public git tree |
| V2 data snapshot | Local-only (759 monthly observations, 1963-07 to 2026-09); not redistributed |
| V2 reference config | Frozen before the reference run |
| V2 implementation | Complete; 45 tests passing; Ruff clean |
| V2 reference run | Complete; see `docs/10_reference_run.md` and `reports/v2_report.md` |
| V2 validation | Complete; 242/242 independent recomputation checks (internal audit) and 41/41 reproduced artifacts; see `docs/13_validation_report.md` |
| V2 release surface | CI, pinned constraints, quickstart notebook, extension guide, research paper (Revision 1, post-publication audit), and [Zenodo DOI](https://doi.org/10.5281/zenodo.24280434) |
| Prior V2 attempt | Preserved in the private development repository; not in this public release |

---

## Design in one paragraph

The benchmark asks whether a probabilistic mixture-of-experts model can allocate across six long-short US equity factor portfolios better than 1/N after costs. It uses a 1963-2026 monthly panel, a 2006+ holdout, 30 causal features, a proper EM mixture of linear Gaussian experts, a constrained mean-variance decision rule, drift-aware turnover accounting, block-bootstrap uncertainty, PIT calibration, power analysis, and a deflated Sharpe ratio. The report states whether the result is supported, negative, or inconclusive, and what was not tested.

---

## Non-goals

This is not a trading strategy, a fund, investment advice, or a claim of live performance. It does not claim that regimes exist, that the latent components are economically identified, or that the Ken French panel is a literal point-in-time trading record.
