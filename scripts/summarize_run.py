"""Print a compact summary of a completed run.

Usage:
    python scripts/summarize_run.py --run-dir runs/reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/reference")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    metrics = _read(run_dir / "evaluation" / "metrics.csv")
    holdout = _read(run_dir / "evaluation" / "holdout_metrics.csv")
    bootstrap = _read(run_dir / "evaluation" / "bootstrap_holdout.csv")
    calibration = _read(run_dir / "evaluation" / "calibration.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    print(f"# Run summary: {run_dir.name}\n")
    print(f"- Git commit: `{manifest.get('git_commit')}`")
    print(f"- Git dirty: `{manifest.get('git_dirty')}`")
    print(f"- Config hash: `{manifest.get('config_hash')}`")
    print(f"- Archive fingerprint: `{manifest['archive_fingerprint']['aggregate_sha256']}`")
    print(
        f"- Data start/end: {manifest['data'].get('loaded_start')} to {manifest['data'].get('loaded_end')}"
    )
    print()

    columns = [
        "sharpe",
        "gross_sharpe",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "utility",
        "avg_turnover",
    ]
    print("## Full out-of-sample\n")
    print(metrics[columns].round(4).to_string())
    print("\n## Holdout\n")
    print(holdout[columns].round(4).to_string())
    print("\n## Holdout Sharpe difference vs benchmark\n")
    if not bootstrap.empty:
        print(
            bootstrap[["sharpe_model", "sharpe_benchmark", "diff", "ci_low", "ci_high", "p_value"]]
            .round(4)
            .to_string()
        )
    print("\n## Calibration\n")
    if not calibration.empty:
        print(
            calibration[
                ["mean_nll", "mean_pit", "pit_below_5pct", "pit_above_95pct", "ks_pooled_p"]
            ]
            .round(4)
            .to_string()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
