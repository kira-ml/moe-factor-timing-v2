# 05 - Architecture

**Goal:** make the research protocol executable, testable, and reproducible.

---

## 1. Design principles

1. **One source of truth per concept.** One model implementation, one backtest engine, one evaluation path, one report generator.
2. **Contracts at every boundary.** Data, features, splits, forecasts, decisions, and results are typed objects with explicit invariants.
3. **Fail loudly.** A model error stops the run. A solver fallback is counted and gated. A data-contract violation stops the run before any model sees the data.
4. **No hidden state.** Seeds are explicit, configuration is hashed, and every artifact is recorded.
5. **Small, composable modules.** A new model or decision rule should not require editing the orchestrator.
6. **The report is generated from artifacts.** No hand-copied numbers.

---

## 2. Layer map

```
data/reference/factors.csv
        |
        v
  data.loader  -->  MarketData  -->  data.validate
        |
        v
  features.build  -->  FeatureSet (X, y, dates, manifest)
        |
        v
  splits.walk_forward  -->  SplitPlan (chronological, holdout)
        |
        v
  models.registry  -->  Allocator or ForecastModel
        |
        v
  decision.mean_variance  -->  weights
        |
        v
  backtest.engine  -->  ModelBacktest per model  -->  BacktestSuite
        |
        v
  evaluation.evaluate  -->  EvaluationResult
        |
        v
  reporting.report + reporting.plots  -->  report.md + figures
        |
        v
  run.manifest  -->  manifest.json (hashes, versions, config, archive fingerprint)
```

---

## 3. Module map

| Module | Responsibility | Key contract |
|---|---|---|
| `config.py` | Load, validate, hash YAML | `Config` dataclass; `ConfigError` |
| `data/schema.py` | Market panel and validation records | `MarketData`, `ValidationReport` |
| `data/french.py` | Download/parse/freeze Ken French panel | `fetch_reference_data` |
| `data/loader.py` | Load the frozen panel with date filters | `load_market_data` |
| `data/validate.py` | Pre-model contract checks | `validate_market_data` |
| `features/build.py` | Causal feature/target construction | `FeatureSet` |
| `splits/walk_forward.py` | Chronological split plan | `Split`, `SplitPlan` |
| `models/base.py` | Forecast and allocator interfaces | `DistributionForecast`, `ForecastModel`, `Allocator` |
| `models/benchmarks.py` | 1/N, cash, inverse vol, momentum | `Allocator` implementations |
| `models/static_mv.py` | Unconditional mean-variance forecast | `StaticMeanVarianceModel` |
| `models/ridge.py` | Regularised linear forecast | `RidgeForecastModel` |
| `models/moe.py` | EM mixture of linear Gaussian experts | `MixtureOfExpertsModel` |
| `models/registry.py` | Strict method/parameter registry | `build_model` |
| `decision/mean_variance.py` | Constrained QP with cost and turnover | `DecisionResult` |
| `backtest/engine.py` | Walk-forward execution and accounting | `ModelBacktest`, `BacktestSuite` |
| `evaluation/metrics.py` | Return, risk, turnover, cost metrics | `compute_metrics` |
| `evaluation/bootstrap.py` | Block-bootstrap Sharpe difference | `sharpe_difference` |
| `evaluation/calibration.py` | PIT and KS diagnostics | `calibration_by_model` |
| `evaluation/power.py` | Minimum detectable effects | `power_summary` |
| `evaluation/deflated_sharpe.py` | PSR and DSR | `deflated_sharpe_table` |
| `evaluation/evaluate.py` | Evaluation orchestration | `EvaluationResult` |
| `reporting/report.py` | Markdown report | `generate_report` |
| `reporting/plots.py` | Static figures | `generate_all_plots` |
| `run/orchestrator.py` | End-to-end pipeline | `run_pipeline` |
| `run/manifest.py` | Artifact hashes and provenance | `write_manifest` |
| `utils/` | Seeds, IO, logging, paths, weight projections | - |

---

## 4. Contracts

### `MarketData`

- `factors`: DataFrame of six factor excess returns, decimal units, month-end index.
- `risk_free`: aligned Series.
- `metadata`: source, date range, row count, SHA-256.
- Invariant: validated before features; no NaN, no missing months, no duplicates.

### `FeatureSet`

- `X`: decision-date features, 30 columns.
- `y`: next-month factor returns, six columns.
- `decision_dates`, `target_dates`.
- `manifest`: feature names, lags, volatility window, correlation audit, target definition.
- Invariant: `X_t` uses data through month `t` only; `y_t` is month `t+1`.

### `SplitPlan`

- Ordered `Split` objects with `train_start_idx=0`, `train_end_idx=test_idx-1`.
- Holdout is a contiguous suffix.
- Invariant: no overlap, chronological test indices, decision date before target date.

### `DistributionForecast`

- `mean`, `cov`, optional mixture `components`.
- Methods: `nll(y)`, `pit(y)`.
- Invariant: covariance is symmetric positive definite; mixture weights sum to 1.

### `DecisionResult`

- `weights`, `status`, `objective`, `traded_notional`.
- Invariant: non-negative, sum <= 1, position cap respected; fully invested when required.

### `ModelBacktest`

- Weights, gross/net/total returns, risk-free, turnover, traded notional, costs, solver status.
- Optional forecast mean/covariance, gating probabilities, NLL, PIT.
- Invariant: returns satisfy the accounting identity; weights are feasible; fallback rate is below the threshold.

### `EvaluationResult`

- Full and holdout metrics, bootstrap tables, calibration, DSR, subperiods, power, metadata.
- Invariant: every metric is computed from the same run artifacts.

---

## 5. Failure handling

| Failure | Behaviour |
|---|---|
| Config missing/invalid | `ConfigError`, exit code 2 |
| Data validation fails | `PipelineError`, no model runs |
| Feature/target misalignment | AssertionError, run stops |
| Model fit/predict raises | Exception propagates, run stops |
| Solver fallback | Recorded; run stops if rate > 5% |
| Non-finite weights/returns | AssertionError, run stops |
| Artifact write failure | Exception propagates, run stops |
| Manifest hash mismatch | Detected by re-running the hash check |

Silent replacement of a failed model with zeros is explicitly forbidden.

---

## 6. Determinism

- Global seeds set once per run (`run.seed`).
- Models receive explicit seeds; they do not read the global RNG.
- The MoE uses a NumPy k-means initialiser with a fixed generator.
- The optimizer is SLSQP, deterministic for a fixed problem and starting point.
- Bootstrap resampling uses a fixed seed.
- Artifacts are CSV/JSON/NPY; no timestamp inside the numerical content.
- The end-to-end test runs the pipeline twice and compares weights and metrics.

The manifest records the git commit and whether the tree was dirty. A reference run should be produced from a clean commit.

---

## 7. Testing strategy

| Test file | What it protects |
|---|---|
| `test_config.py` | Config schema, duplicate models, unknown parameters, hash changes |
| `test_data_contract.py` | Data schema, missing values, missing months, duplicates, short history |
| `test_features.py` | Feature shapes, target definition, future perturbation, truncation invariance |
| `test_splits.py` | Chronology, no overlap, contiguous holdout, test step |
| `test_models.py` | Forecast shapes, covariance PSD, mixture weights, determinism, unfitted errors |
| `test_decision.py` | Feasibility, cap, turnover cap, fallback, water-filling |
| `test_benchmarks.py` | Momentum cash behaviour, finite weights |
| `test_backtest_accounting.py` | Drift, turnover, cost, net-return identity |
| `test_evaluation.py` | Metric identities, bootstrap determinism, PSR/DSR bounds, power monotonicity |
| `test_manifest.py` | Archive integrity fingerprint |
| `test_end_to_end.py` | Full pipeline, artifacts, report, determinism |

The tests are not decoration. The drift-aware accounting test and the future-perturbation test are the two that would have caught V1's most serious errors.

---

## 8. Extension points

### Add a forecast model

1. Implement `ForecastModel.fit` and `predict`, returning a `DistributionForecast`.
2. Add a method and allowed parameters to `models/registry.py`.
3. Add a test for shape, covariance, and determinism.
4. Add the model to a config.

No orchestrator edit is required.

### Add a direct allocator

1. Implement `Allocator.weights(history, decision_date)`.
2. Add it to the registry.
3. Add a test that it never sees data after the decision date.

### Add a decision rule

1. Implement a function with the same `(mu, cov, w_prev, config)` signature.
2. Add it to the decision config and a test for feasibility.
3. Keep the old rule available for comparison.

### Add a data source

1. Write a loader that returns `MarketData`.
2. Extend `validate_market_data` with source-specific checks.
3. Freeze a snapshot and record its hash.
4. Never download silently inside a research run.

---

## 9. Why this is better than V1

| V1 | V2 |
|---|---|
| `main.py` orchestrates everything | Thin CLI calls a typed orchestrator |
| Hard-coded model parameters in `main.py` | Validated YAML config, copied and hashed |
| `src/data_pipeline.py` mixes download, resample, validate, save | Separate source, loader, schema, and validation modules |
| Two MoE implementations | One MoE, one registry, one engine |
| No tests | Contract, leakage, accounting, evaluation, manifest, end-to-end tests |
| Timestamped folders with no manifest | Run directory with config, features, splits, models, evaluation, report, manifest |
| Costs never charged | Drift-aware turnover and cost accounting |
| Point estimates only | Bootstrap, power, calibration, DSR |
| Silent fallbacks | Fail-fast model errors and a solver-fallback gate |

---

## 10. Complexity budget

The architecture adds modules, but each module has one job and a typed boundary. The added complexity buys:

- a testable no-look-ahead contract,
- correct cost accounting,
- reproducible artifact provenance,
- the ability to add a model without touching the engine,
- and a report that cannot drift from the numbers.

The benchmark does not add distributed compute, a database, a web service, or a deep-learning framework. Those would be complexity without a demonstrated need.
