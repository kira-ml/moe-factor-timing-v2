# Changelog

## 2.0.2 - 2026-11-14

Publication-safety and attribution pass.

### Added

- Full author name and open-source GitHub link in the one-page summary.
- `docs/12_legal_ethical_security_review.md`: data, licensing, font, secret, and public-history review.
- `NOTICE`, `THIRD_PARTY_NOTICES.md`, and `publication/FONT_LICENSE.txt`.
- An evidence-design strip and falsification criterion on the one-pager.

### Changed

- PDF fonts switched from Microsoft system fonts to Source Serif 4 + IBM Plex Sans/Mono (SIL OFL 1.1).
- The raw Ken French CSV and the V1 archive are excluded from the public git tree and retained locally.
- The report uses relative paths instead of absolute local paths.
- Public copy distinguishes full-sample vs holdout IC and "pre-committed" vs "pre-registered".

### Residual

- Earlier public commits still contain the raw data and archive; a fully clean public release requires a fresh repository or a history rewrite.

## 2.0.1 - 2026-11-14

Pre-publication red-team validation.

### Added

- `docs/11_red_team_validation.md`: publication gate, findings, and corrected claims.
- `scripts/red_team_checks.py`: holdout cost sensitivity, gross/net bootstrap, HAC and block-bootstrap IC, subperiod dominance.
- `reports/red_team_checks.json`, `reports/red_team_cost_sensitivity.csv`.

### Corrected

- The holdout gross Sharpe edge is **not** statistically confirmed (CI includes zero; holdout IC t = 1.35).
- The holdout net sign flips between 5 and 10 bps; the 5 bps result is within noise and is not an edge.
- Public copy now distinguishes full-sample IC from holdout IC and uses "pre-committed" rather than "pre-registered".

## 2.0.0 - 2026-11-14

Complete rebuild of the repository around a pre-registered research protocol.

### Added

- Long-sample Ken French five-factor plus momentum panel (1963-07 to 2026-09).
- Typed, validated configuration with a config hash copied into every run.
- Causal feature builder with future-perturbation and truncation invariance tests.
- Proper EM mixture of linear Gaussian experts with a predictive mixture distribution.
- Ridge, static mean-variance, inverse-volatility, momentum, cash, and 1/N benchmarks.
- Constrained mean-variance decision rule with position cap, turnover penalty, and deterministic SLSQP.
- Drift-aware turnover and cost accounting.
- Block-bootstrap Sharpe differences, PIT calibration, power analysis, and deflated Sharpe ratio.
- Manifest with artifact hashes, package versions, git commit, and archive fingerprint.
- Generated Markdown report and figures.
- 40+ tests, including an end-to-end determinism test.

### Changed

- The default universe excludes VIX and uses long-short factor portfolios.
- The primary evaluation is a 2006+ holdout, not the full sample.
- The package uses a `src/moe_factor_timing` layout instead of a top-level `src` package.

### Removed

- The prior V2 working tree (preserved in the private development repository).
- The V1 `SimpleMoE`, PyTorch MoE, FRED macro loader, and report scripts from the active tree.
- `archive/v1/` is untouched and remains the permanent V1 reference.

### Fixed relative to V1

- Transaction costs are charged every month against drifted weights.
- 1/N is a first-class benchmark.
- No selection on the evaluation window.
- No 96-feature / 96-month p-near-n design.
- No dual implementation drift.
- No silent zero-return fallback for failed models.
