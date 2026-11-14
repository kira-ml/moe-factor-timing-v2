"""Cross-sectional information coefficient of saved forecasts.

For each decision month, correlate the predicted factor returns with the
realised next-month returns across the six factors. A model with decision
value should have a positive mean IC. This is a diagnostic, not a substitute
for the portfolio-level evaluation.

Usage:
    python scripts/forecast_ic.py --run-dir runs/reference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_factor_timing.utils.io import ensure_dir, write_csv  # noqa: E402


def _model_ic(model_dir: Path, y: pd.DataFrame) -> dict | None:
    forecast_path = model_dir / "forecast_mean.csv"
    if not forecast_path.exists():
        return None
    forecast = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
    y = y.loc[forecast.index]
    ics: list[float] = []
    hits: list[float] = []
    for date in forecast.index:
        predicted = forecast.loc[date].to_numpy(dtype=float)
        realized = y.loc[date].to_numpy(dtype=float)
        if np.std(predicted) <= 1e-12 or np.std(realized) <= 1e-12:
            continue
        ic = float(spearmanr(predicted, realized).statistic)
        if np.isfinite(ic):
            ics.append(ic)
            hits.append(float(np.mean(np.sign(predicted) == np.sign(realized))))
    if not ics:
        return None
    values = np.asarray(ics)
    return {
        "model": model_dir.name,
        "n_months": int(len(values)),
        "mean_ic": float(values.mean()),
        "std_ic": float(values.std(ddof=1)),
        "t_stat": float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values)))),
        "positive_ic_fraction": float(np.mean(values > 0)),
        "sign_hit_rate": float(np.mean(hits)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/reference")
    parser.add_argument("--output", default="reports/forecast_ic.csv")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    y = pd.read_csv(run_dir / "features" / "y.csv", index_col=0, parse_dates=True)
    rows = []
    for model_dir in sorted((run_dir / "models").iterdir()):
        result = _model_ic(model_dir, y)
        if result is not None:
            rows.append(result)
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    ensure_dir(output.parent)
    write_csv(output, frame, index=False)
    print(frame.round(4).to_string(index=False))
    print(json.dumps({"output": str(output), "rows": len(frame)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
