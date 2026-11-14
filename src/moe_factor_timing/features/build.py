"""Feature construction with an explicit no-look-ahead contract.

Definitions
-----------
At a decision date ``t`` (month-end) the feature row uses factor returns
through month ``t`` only. The target is the factor return in month ``t + 1``.
There is no gap and no future information: ``r_t`` is observed at the close of
month ``t``, and the position is held over month ``t + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data.schema import MarketData
from ..utils.io import ensure_dir, write_csv, write_json


@dataclass(frozen=True)
class FeatureSet:
    X: pd.DataFrame
    y: pd.DataFrame
    decision_dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    manifest: dict

    def save(self, directory: str | Path) -> None:
        out = ensure_dir(Path(directory))
        X_out = self.X.copy()
        X_out.index.name = "decision_date"
        y_out = self.y.copy()
        y_out.index.name = "decision_date"
        write_csv(out / "X.csv", X_out)
        write_csv(out / "y.csv", y_out)
        write_csv(
            out / "dates.csv",
            pd.DataFrame({"decision_date": self.decision_dates, "target_date": self.target_dates}),
            index=False,
        )
        write_json(out / "feature_manifest.json", self.manifest)


def build_features(market: MarketData, config: Config) -> FeatureSet:
    returns = market.factors
    assets = list(returns.columns)
    lags = list(config.features.return_lags)
    vol_window = config.features.vol_window

    columns: dict[str, pd.Series] = {}
    feature_meta: list[dict] = []

    for asset in assets:
        for lag in lags:
            name = f"ret_{asset}_lag{lag}"
            columns[name] = returns[asset].shift(lag)
            feature_meta.append(
                {"name": name, "category": "lagged_return", "asset": asset, "lag": lag}
            )
        vol_name = f"vol{vol_window}_{asset}"
        columns[vol_name] = returns[asset].rolling(vol_window, min_periods=vol_window).std()
        feature_meta.append(
            {
                "name": vol_name,
                "category": "realized_volatility",
                "asset": asset,
                "window": vol_window,
            }
        )

    X = pd.DataFrame(columns)
    y = returns.shift(-1).copy()

    combined = pd.concat([X.add_prefix("x_"), y.add_prefix("y_")], axis=1)
    if config.features.dropna:
        combined = combined.dropna()
    if combined.empty:
        raise ValueError("feature construction produced no complete rows")

    X = combined.loc[:, [c for c in combined.columns if c.startswith("x_")]].rename(
        columns=lambda c: c[2:]
    )
    y = combined.loc[:, [c for c in combined.columns if c.startswith("y_")]].rename(
        columns=lambda c: c[2:]
    )
    X = X.loc[:, [meta["name"] for meta in feature_meta]]
    y = y.loc[:, assets]

    decision_dates = pd.DatetimeIndex(X.index)
    target_dates = pd.DatetimeIndex(decision_dates + pd.offsets.MonthEnd(1))

    correlations: list[dict] = []
    max_abs = 0.0
    for feature in X.columns:
        for target in y.columns:
            corr = float(np.corrcoef(X[feature].to_numpy(), y[target].to_numpy())[0, 1])
            if np.isfinite(corr):
                max_abs = max(max_abs, abs(corr))
                if abs(corr) > 0.95:
                    correlations.append({"feature": feature, "target": target, "corr": corr})

    manifest = {
        "n_features": int(X.shape[1]),
        "n_targets": int(y.shape[1]),
        "n_rows": int(len(X)),
        "start": decision_dates.min().strftime("%Y-%m-%d"),
        "end": decision_dates.max().strftime("%Y-%m-%d"),
        "return_lags": lags,
        "vol_window": vol_window,
        "feature_names": list(X.columns),
        "target_names": list(y.columns),
        "features": feature_meta,
        "target_definition": (
            "y_t is the factor excess return realised in month t+1; X_t uses factor "
            "returns and realised volatility through month t. Positions are formed at "
            "the close of month t and held over month t+1."
        ),
        "max_abs_feature_target_correlation": float(max_abs),
        "high_correlation_warnings": correlations,
        "scaling": "models standardise features using training data only",
    }

    return FeatureSet(
        X=X.astype(float),
        y=y.astype(float),
        decision_dates=decision_dates,
        target_dates=target_dates,
        feature_names=tuple(X.columns),
        target_names=tuple(y.columns),
        manifest=manifest,
    )
