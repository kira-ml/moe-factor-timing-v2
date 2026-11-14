"""Data contracts shared by ingestion, validation, features, and backtests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Canonical column names. Hyphens in source headers are normalised to
# underscores so downstream code never has to quote column names.
MKT_RF = "MKT_RF"
SMB = "SMB"
HML = "HML"
RMW = "RMW"
CMA = "CMA"
MOM = "MOM"
RF = "RF"

FACTOR_COLUMNS: tuple[str, ...] = (MKT_RF, SMB, HML, RMW, CMA, MOM)


@dataclass(frozen=True)
class MarketData:
    """A validated, point-in-time monthly market panel.

    Attributes
    ----------
    factors:
        Excess returns of the investable factor strategies, in decimal units
        (0.01 == 1%). The index contains month-end decision dates.
    risk_free:
        Monthly risk-free return, in decimal units, aligned to ``factors``.
    metadata:
        Provenance and validation details copied into the run artifacts.
    """

    factors: pd.DataFrame
    risk_free: pd.Series
    metadata: dict

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(self.factors.columns)

    @property
    def n_assets(self) -> int:
        return int(self.factors.shape[1])

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.factors.index[0])

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.factors.index[-1])


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict:
        return {
            "passed": bool(self.passed),
            "checks": [
                {"name": c.name, "passed": bool(c.passed), "detail": c.detail} for c in self.checks
            ],
        }
