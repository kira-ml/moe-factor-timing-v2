# 07 - Reproducibility

**Reference run:** `runs/reference`
**Config:** `configs/default.yaml`
**Data snapshot:** local-only `data/reference/factors.csv` (not redistributed)

---

## 1. Environment

- Python 3.10+
- numpy, pandas, scipy, scikit-learn, PyYAML, matplotlib
- pytest and ruff for development

Install:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

python -m pip install -e ".[dev]"
```

The package is installed from `pyproject.toml`; imports are `moe_factor_timing.*`, not `src.*`.

---

## 2. Data

The reference panel is downloaded locally to `data/reference/factors.csv`; it is **not committed** to the public git tree because the source does not state an explicit redistribution license. Fetch it first:

```bash
python scripts/fetch_reference_data.py
```

Then validate:

```bash
python -m moe_factor_timing validate --config configs/default.yaml
```

Refresh the panel (explicit, separate step):

```bash
python scripts/fetch_reference_data.py --force
```

The refresh records source URLs, retrieval time, row count, date range, and SHA-256 in `data/reference/metadata.json`.

---

## 3. Run

Reference run:

```bash
python -m moe_factor_timing run --config configs/default.yaml --run-id reference --publish
```

Fast implementation smoke test (not a research result):

```bash
python -m moe_factor_timing run --config configs/smoke.yaml --run-id smoke
```

The run directory is immutable by convention. The orchestrator refuses to write into a non-empty run directory.

---

## 4. Artifacts

```
runs/<run_id>/
  config.yaml                 copied source config
  config_resolved.json        validated, serialised config
  validation.json             data-contract report
  data/
    factors.csv               frozen input snapshot
    risk_free.csv
    data_manifest.json
  features/
    X.csv
    y.csv
    dates.csv
    feature_manifest.json
  splits/
    plan.json
  backtest/
    splits.json
    suite.json
  models/<model>/
    weights.csv
    returns.csv               gross, net, risk-free, turnover, cost
    solver_status.csv
    forecast_mean.csv         probabilistic models
    forecast_cov.npy
    gating.csv                MoE
    nll.csv
    pit.csv
    summary.json
  evaluation/
    metrics.csv
    holdout_metrics.csv
    bootstrap_full.csv
    bootstrap_holdout.csv
    calibration.csv
    calibration_by_asset.csv
    deflated_sharpe.csv
    subperiods.csv
    power.json
    evaluation.json
  report/
    report.md
    figures/*.png
  manifest.json
```

`manifest.json` records:

- run id and UTC timestamp,
- config hash and path,
- git commit and dirty flag,
- Python and package versions,
- data metadata,
- validation report,
- evaluation metadata,
- SHA-256 and byte size of every artifact,
- a deterministic fingerprint of `archive/`.

---

## 5. Determinism

Run the test suite:

```bash
python -m pytest tests/ -q
```

The end-to-end test runs the pipeline twice and compares model weights and metrics. The MoE, ridge, solver, and bootstrap all use explicit seeds or deterministic algorithms.

Known sources of non-determinism:

- BLAS/OpenMP reductions can differ across machines at the last floating-point bit.
- SLSQP can terminate at a slightly different active set if the problem is ill-conditioned.
- The manifest's `git_dirty` flag depends on the working tree.

The reference run is produced from a clean commit; the manifest records the commit.

---

## 6. Archive integrity

The V1 archive is local-only read-only reference material. `tests/test_manifest.py` checks its fingerprint when the archive is present; the test skips on a public clone that does not include it.

- file count: 183
- aggregate SHA-256: `716ff99a5cc32c63971b668bb5dc6eb0909b6fce820fafa8d1f9c1fdb9d4a058`

Every local run manifest also records the archive fingerprint. If the local archive changes, the test fails and the manifest records a different fingerprint.

To verify manually:

```bash
python -c "from moe_factor_timing.run.manifest import archive_fingerprint; print(archive_fingerprint())"
```

---

## 7. Reference run checklist

Before publishing a result:

- [ ] `python -m pytest tests/ -q` passes.
- [ ] `python -m ruff check .` passes.
- [ ] `configs/default.yaml` is the config used for the run.
- [ ] The data snapshot hash matches `data/reference/metadata.json`.
- [ ] The run manifest's `git_dirty` is `false` (or the dirty state is disclosed).
- [ ] The report includes the primary endpoint, confidence interval, turnover, calibration, DSR, power, and limitations.
- [ ] The archive fingerprint matches the expected value.

---

## 8. Reference run record

The audit record for the final run is in [`docs/10_reference_run.md`](10_reference_run.md). It records:

- code commit `aa5d16052cf4a07048c8343819ff846ccf55d52c`,
- clean working tree at manifest time,
- config hash `2a035de2d1a091455b816690492ae20a6bef2c1936ebffd0e51ed4f4c3daaa5f`,
- data snapshot SHA-256 `cc805ab0...`,
- archive fingerprint `716ff99a...`,
- 79 hashed artifacts.

The reference report is committed at `reports/v2_report.md`. The authoritative artifacts are in `runs/reference/`.

---

## 9. Prior V2 attempt

The prior V2 implementation and postmortem are preserved in the private development repository. They are not included in this public release, and the public clone does not contain those tags or files.

The archive of V1 is preserved locally at `archive/v1/` and is never modified. It is excluded from the public git tree because it contains third-party data with unresolved redistribution terms.
