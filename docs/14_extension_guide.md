# 14 - Extension Guide: Adding a Model

**Status:** Contributor guide
**Audience:** researchers and engineers adding a new allocator or forecast model
**Contract:** `src/moe_factor_timing/models/base.py`

---

## 1. Two kinds of model

The pipeline distinguishes two interfaces.

| Interface | Input | Output | Used when |
|---|---|---|---|
| `Allocator` | Price/return history through the decision date | A weight vector | The rule maps directly to weights, with no forecast or optimizer |
| `ForecastModel` | Training features `X`, targets `y`; then a test feature row `x` | A `DistributionForecast` with mean and covariance | The model produces a predictive distribution that the mean-variance decision rule consumes |

If your model produces a point forecast only, wrap it in a `DistributionForecast` with a residual covariance. A point estimate without a covariance cannot be passed to the decision layer.

---

## 2. Forecast model: minimal example

Create a new module, for example `src/moe_factor_timing/models/historical_mean.py`:

```python
from __future__ import annotations

import numpy as np

from .base import DistributionForecast, ForecastModel, shrink_covariance


class HistoricalMeanForecast(ForecastModel):
    """Unconditional historical mean with a shrunk covariance."""

    name = "historical_mean"

    def __init__(self, shrinkage: float = 0.2):
        self.shrinkage = float(shrinkage)
        self.mean_: np.ndarray | None = None
        self.cov_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> HistoricalMeanForecast:
        X, y = self._validate_fit_inputs(X, y)
        if len(y) < 2:
            raise ValueError("historical_mean needs at least two observations")
        self.mean_ = y.mean(axis=0)
        covariance = np.cov(y, rowvar=False, ddof=1)
        self.cov_ = shrink_covariance(np.atleast_2d(covariance), self.shrinkage)
        return self

    def predict(self, x: np.ndarray) -> DistributionForecast:
        if self.mean_ is None or self.cov_ is None:
            raise RuntimeError("model is not fitted")
        return DistributionForecast(
            mean=self.mean_,
            cov=self.cov_,
            model_name=self.name,
        )
```

This model deliberately ignores `x`; it is a baseline, not a conditional model. The important part is the contract: `fit` returns `self`, and `predict` returns a mean vector and a positive-definite covariance.

### Required properties

- `mean.shape == (n_assets,)`
- `cov.shape == (n_assets, n_assets)`
- `cov` is symmetric and positive definite; use `shrink_covariance` or `ensure_covariance` if needed.
- No future information is used in `fit` or `predict`.
- Scaling is fitted on the training rows only.
- The output is deterministic for a fixed seed and input.

---

## 3. Register the model

Add the import and a branch in `src/moe_factor_timing/models/registry.py`:

```python
from .historical_mean import HistoricalMeanForecast
```

Inside `build_forecast_model`:

```python
if spec.method == "historical_mean":
    _check_params(spec.name, params, {"shrinkage"})
    return HistoricalMeanForecast(shrinkage=float(params.get("shrinkage", 0.2)))
```

The `_check_params` call is important: a misspelled parameter must fail loudly rather than be silently ignored.

For an allocator, add a branch in `build_allocator` and implement `weights(history, decision_date)` instead.

---

## 4. Add it to a configuration

Add a model entry to a config file:

```yaml
models:
  - name: one_over_n
    kind: allocator
    method: equal
  - name: historical_mean
    kind: forecast
    method: historical_mean
    params: {shrinkage: 0.2}
  - name: moe
    kind: forecast
    method: moe
    params: {n_experts: 2, max_iter: 10}
```

`name` is the output label used in tables and artifacts. `method` selects the class in the registry. `kind` must be `allocator` or `forecast`.

Use `configs/smoke.yaml` for a fast check. Do not edit `configs/default.yaml` to test a new model unless you intend to change the reference protocol.

---

## 5. Add a test

At minimum, test:

```python
import numpy as np

from moe_factor_timing.models.historical_mean import HistoricalMeanForecast


def test_historical_mean_forecast():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 30))
    y = rng.normal(size=(120, 6))

    model = HistoricalMeanForecast().fit(X, y)
    forecast = model.predict(X[0])

    assert forecast.mean.shape == (6,)
    assert forecast.cov.shape == (6, 6)
    assert np.allclose(forecast.cov, forecast.cov.T)
    assert np.linalg.eigvalsh(forecast.cov).min() > 0
```

Also add a test that the model does not see future data, following the existing feature and split tests.

---

## 6. Run the smoke pipeline

```bash
python -m moe_factor_timing run --config configs/smoke.yaml --run-id ext_smoke
```

Check:

- the model appears in `runs/ext_smoke/models/<name>/weights.csv`;
- `forecast_mean.csv` and `forecast_cov.npy` exist for a forecast model;
- `returns.csv` contains gross, net, cost, and turnover columns;
- `evaluation/metrics.csv` includes the model;
- the run manifest hashes every artifact.

Then run the tests and lint:

```bash
python -m pytest tests/ -q
python -m ruff check src tests scripts
```

---

## 7. Evaluation caveats

Adding a model changes the comparison set. Before interpreting a result:

- **Update the trial count.** If you add a model to a reference comparison, update `evaluation.deflated_sharpe.n_trials` and disclose the change.
- **Do not select on the holdout.** A new model is not a license to tune on the confirmatory window.
- **Report the benchmark.** A new model must be compared with 1/N and the existing forecast baselines, not only with cash.
- **Report costs and turnover.** A better forecast is not a better strategy until it survives the decision rule and costs.
- **Report calibration.** A probabilistic model that is overconfident can take oversized positions even when its mean forecast is useful.

---

## 8. Where to look next

- `docs/03_methodology.md` — features, models, decision rule, costs, walk-forward.
- `docs/04_evaluation_protocol.md` — endpoints, uncertainty, power, multiple testing.
- `docs/05_architecture.md` — module map, interfaces, invariants.
- `src/moe_factor_timing/models/base.py` — `ForecastModel`, `Allocator`, `DistributionForecast`.
- `src/moe_factor_timing/models/registry.py` — model dispatch and parameter validation.

If you add a model that materially changes the benchmark, open a pull request with the model, tests, a smoke config, and the exact command used to reproduce the comparison.
