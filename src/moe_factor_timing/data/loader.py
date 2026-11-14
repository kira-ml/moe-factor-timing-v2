"""Load the frozen reference panel into the typed market-data contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config
from ..utils.io import read_json
from ..utils.paths import resolve_path
from .schema import MarketData


def load_market_data(config: Config) -> MarketData:
    reference_dir = resolve_path(config.data.reference_dir)
    factors_path = reference_dir / "factors.csv"
    metadata_path = reference_dir / "metadata.json"
    if not factors_path.exists():
        raise FileNotFoundError(
            f"reference panel not found at {factors_path}; run 'moe-factor-timing fetch-data'"
        )

    panel = pd.read_csv(factors_path, parse_dates=["date"], index_col="date")
    panel.index = pd.DatetimeIndex(panel.index)
    panel = panel.sort_index()

    required = [*config.data.factors, config.data.risk_free]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(f"reference panel is missing columns: {missing}")

    if config.data.start is not None:
        panel = panel.loc[pd.Timestamp(config.data.start) :]
    if config.data.end is not None:
        panel = panel.loc[: pd.Timestamp(config.data.end)]
    if panel.empty:
        raise ValueError("date filters produced an empty panel")

    factors = panel.loc[:, list(config.data.factors)].astype(float)
    risk_free = panel.loc[:, config.data.risk_free].astype(float)

    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    return MarketData(
        factors=factors,
        risk_free=risk_free,
        metadata={
            **metadata,
            "loaded_columns": list(panel.columns),
            "loaded_start": factors.index.min().strftime("%Y-%m-%d"),
            "loaded_end": factors.index.max().strftime("%Y-%m-%d"),
            "loaded_rows": int(len(factors)),
            "reference_dir": str(Path(reference_dir)),
        },
    )
