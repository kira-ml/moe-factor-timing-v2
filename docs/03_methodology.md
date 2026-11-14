# 03 - Methodology

**Status:** Frozen before the reference run
**Code:** `src/moe_factor_timing/`

---

## 1. Notation

- `t`: month-end decision date.
- `r_t`: vector of six factor excess returns for month `t`.
- `rf_t`: one-month risk-free return for month `t`.
- `x_t`: feature vector observed at `t`.
- `y_t = r_{t+1}`: target vector for the decision at `t`.
- `w_t`: portfolio weights chosen at `t`, held over month `t + 1`.
- `c`: one-way transaction cost in decimal per unit traded.
- `lambda`: risk aversion in the mean-variance objective.

Returns and costs are decimal throughout. A value of `0.01` is 1%.

---

## 2. Features

The feature set is deliberately small and causal:

| Feature family | Count | Definition |
|---|---:|---|
| Lagged returns | 24 | For each of six factors, returns at lags 0, 1, 5, and 11 months |
| Realised volatility | 6 | Twelve-month rolling standard deviation of each factor return |
| **Total** | **30** | |

At decision date `t`, lag 0 is the return for month `t` itself, which is observed at the close. Lags 1, 5, and 11 are prior months. No future return is used.

Features are standardised **inside each training window**:

```
x_scaled = (x - mean_train) / std_train
```

The scaler is never fitted on the evaluation row or on the full sample. The target is standardised the same way for the MoE and ridge models, and predictions are transformed back to return units.

### Why not 96 features

V1 used 96 features with roughly 96 training months. That is a p-near-n design: the linear model was ill-conditioned, and the MoE's gating and expert parameters were poorly identified. V2 uses 30 features with at least 120 training months. The feature manifest reports the observations-to-features ratio for every run.

### Why no macro features

Revised FRED series are not point-in-time. A one-month lag does not fix revisions, publication delays, or the difference between first-release and final values. Rather than hide this, the main analysis excludes macro features. A vintage-aware macro extension is a separate study.

---

## 3. Models

### 3.1 Direct allocators

These map observable history to weights without a forecast or an optimizer.

| Model | Rule |
|---|---|
| `one_over_n` | `w_i = 1/6` each month |
| `cash` | `w = 0`; the portfolio earns the risk-free rate |
| `inverse_vol` | `w_i proportional to 1 / sigma_i`, capped at 0.35, normalised to 1 |
| `momentum` | Long factors with positive 12-month cumulative return; weights proportional to momentum; capped; cash if none are positive |

The direct allocators are transparent benchmarks. They are not fitted to the evaluation window.

### 3.2 Static mean-variance

The static model estimates the unconditional mean and covariance from the training window:

```
mu_t = mean(y_train)
Sigma_t = shrink(cov(y_train))
```

It uses the same decision rule as the conditional models. It isolates the value of conditioning from the value of optimisation.

### 3.3 Ridge

The ridge model estimates a linear conditional mean:

```
B = argmin_B ||Y - [1, X] B||^2 + alpha ||B||^2
mu(x) = [1, x] B
Sigma = shrink(residual covariance)
```

The intercept is not penalised. `alpha = 10.0` after standardisation. The residual covariance is shrunk toward its diagonal with `shrinkage = 0.2`, and a small eigenvalue floor keeps it positive definite.

Ridge is a strong, parsimonious conditional-mean benchmark. If the MoE cannot beat ridge, the extra mixture structure has not earned its complexity.

### 3.4 Mixture of Experts

The MoE is a mixture of `K` linear Gaussian experts with a multinomial-logit gate.

For expert `k`:

```
pi_k(x) = softmax_k(W^T [1, x])
p(y | x, k) = N(y | B_k^T [1, x], Sigma_k)
p(y | x) = sum_k pi_k(x) p(y | x, k)
```

The predictive mean and covariance are:

```
mu(x) = sum_k pi_k(x) mu_k(x)
Sigma(x) = sum_k pi_k(x) [Sigma_k + (mu_k(x) - mu(x))(mu_k(x) - mu(x))^T]
```

The covariance includes both within-expert noise and between-expert dispersion. This is the distribution used by the decision rule and by the calibration tests.

#### Training

The model is trained by expectation-maximisation:

1. **Initialise** responsibilities with a deterministic k-means partition of the standardised features. The k-means is a small NumPy implementation, not sklearn's, to avoid per-fit CPU discovery overhead in the walk-forward loop.
2. **E-step:** compute responsibilities
   `r_ik proportional to pi_k(x_i) N(y_i | mu_ik, Sigma_k)`.
3. **M-step:**
   - Experts: weighted ridge regression with weights `r_ik`.
   - Covariances: weighted residual covariance, shrunk toward the diagonal.
   - Gate: L2-regularised multinomial logit fitted by backtracking gradient descent.
4. **Convergence:** stop when the relative change in the training log-likelihood is below `1e-6`, or at `max_iter`.
5. **Restarts:** the config may use more than one initialisation. The restart with the highest **training** log-likelihood is kept. The evaluation window is never used to choose a restart.

#### Regularisation and stability

- `ridge_alpha = 1.0` on standardised features.
- `gating_l2 = 0.01`.
- `covariance_shrinkage = 0.2`.
- A minimum responsibility of `1e-4` prevents complete component collapse.
- Covariances are repaired to the nearest positive-definite matrix with a small eigenvalue floor.
- Experts are canonically ordered by their average predicted first-factor return for diagnostics. This does not change the likelihood or the predictive distribution.

#### Why a linear-Gaussian MoE

The point is to test whether **soft regime membership in conditional factor means** adds decision value. A small, interpretable model is the right instrument:

- It has a tractable likelihood, so calibration can be tested.
- It has a closed-form predictive covariance, so the decision rule is well defined.
- It is regularised and can be estimated on 120-700 monthly observations.
- It does not require a GPU or a second implementation.

A neural MoE is a different hypothesis and a different project. V2 does not claim to test it.

---

## 4. Decision rule

At each decision date, the model produces `(mu, Sigma)` and the allocator solves:

```
max_w   w^T mu - (lambda/2) w^T Sigma w
        - c * ||w - w_prev||_1
        - d * sum(w)
s.t.    w_i >= 0
        sum(w) = 1                (fully invested)
        w_i <= 0.35               (position cap)
        ||w - w_prev||_1 <= cap   (optional turnover cap)
```

where:

- `c = 10 bps` one-way per unit traded,
- `d = 0` strategy drag in the primary run,
- `lambda = 2`,
- `w_prev` is the **drifted** weight vector at the decision date, not last month's target.

The absolute-value term is represented with auxiliary variables, giving a smooth convex quadratic program with linear constraints. The solver is deterministic SLSQP. A solver failure is recorded; if the fallback rate exceeds 5%, the run stops. The optimizer never silently returns zero returns.

### Why fully invested

Factor timing is a **relative allocation** problem. Allowing cash would confound factor timing with market timing and risk-aversion calibration. The primary rule is fully invested; a cash-allowed variant is a robustness check, not the headline.

### Why a position cap

The cap prevents a single factor from dominating the portfolio when its predicted mean is extreme. It is a risk-control constraint, not a return forecast.

---

## 5. Drift-aware turnover and costs

A portfolio's weights drift with returns between rebalances. Turnover must be measured against the **pre-rebalance drifted weights**, not last month's target.

For a factor sleeve with weight `w_i`, the sleeve earns `rf + r_i`. The drifted factor weight is:

```
w_i^drift = w_i (1 + rf + r_i) / (1 + rf + sum_j w_j r_j)
```

The cash residual earns `rf`. The denominator is the total portfolio growth factor.

At each rebalance:

```
traded_notional = sum_i |w_i^target - w_i^drift|
turnover = 0.5 * traded_notional
cost = c * traded_notional + d * sum_i w_i^target
```

The first rebalance is charged explicitly: starting from cash, a fully invested portfolio trades 1.0 of notional (0.5 turnover) and pays 10 bps.

This fixes the V1 cost bug, where a one-row cost function never saw the previous month's weights and therefore reported zero turnover.

**Limitation:** the cost model covers rebalancing across factor sleeves, not the internal turnover of each long-short factor portfolio. This is disclosed and subjected to a cost-sensitivity check.

---

## 6. Walk-forward protocol

```
for t = min_train, min_train + 1, ..., T - 1:
    fit the model on rows [0, t)
    predict the distribution for row t
    solve the decision rule using the drifted weights
    hold the position over month t + 1
    record weights, gross return, cost, net return
```

- `min_train = 120` months.
- `test_step = 1` month.
- No shuffling, no random splits, no overlap.
- The first decision is at the end of the 120th feature month.
- The holdout starts at 2006-01-31 and is never used for fitting or model selection.

The full out-of-sample window is reported as a secondary result. The holdout is the confirmatory result.

---

## 7. Hyperparameters

All primary hyperparameters are fixed before the reference run:

| Parameter | Value | Justification |
|---|---:|---|
| Return lags | 0, 1, 5, 11 | Current, prior, semi-annual, annual; parsimonious |
| Volatility window | 12 | Annual realised volatility |
| Minimum training | 120 | Ten years; enough for 30 features and a stable covariance |
| Holdout start | 2006-01-31 | Twenty-year confirmatory suffix |
| Experts `K` | 3 | Small enough to identify; large enough for distinct regimes |
| MoE restarts | 1 | Main run; restart stability is checked separately |
| MoE EM iterations | 25 | Convergence is monitored; most fits stop earlier |
| Gating steps | 6 | Backtracking gradient steps per M-step |
| Ridge alpha | 10.0 | Standardised features; conservative |
| Covariance shrinkage | 0.2 | Diagonal target; stabilises the predictive covariance |
| Risk aversion | 2.0 | Moderate; the fully invested constraint prevents cash confounding |
| Max weight | 0.35 | Prevents single-factor concentration |
| One-way cost | 10 bps | Liquid factor sleeves; sensitivity at 5 and 20 bps |
| Bootstrap reps | 1000 | Stable 95% intervals |
| Block length | 6 months | Preserves short-range dependence |
| N trials | 7 | Seven models in the reference comparison; robustness adds trials |

The config is copied into every run directory and hashed. A change in any value changes the config hash and the manifest.

---

## 8. Implementation invariants

The engine enforces the following at runtime:

1. `X_train.shape[0] == y_train.shape[0]`.
2. The target row equals the market return at the target date.
3. Weights are non-negative, sum to at most 1, and respect the position cap.
4. Forecast-model weights are fully invested when the config requires it.
5. Returns and costs are finite.
6. Model errors fail the run; they are not replaced by zeros.
7. Solver fallbacks are counted; a rate above 5% stops the run.
8. Every artifact is hashed in the manifest.

These invariants are tested in `tests/test_backtest_accounting.py`, `tests/test_decision.py`, and the end-to-end test.
