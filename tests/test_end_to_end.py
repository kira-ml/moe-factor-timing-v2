from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import synthetic_market

from moe_factor_timing.run.orchestrator import run_pipeline
from moe_factor_timing.utils.io import read_json, write_json
from moe_factor_timing.utils.paths import project_root


def _write_reference(tmp_path: Path, market) -> Path:
    reference = tmp_path / "reference"
    reference.mkdir()
    panel = pd.concat([market.factors, market.risk_free.rename("RF")], axis=1)
    panel.index.name = "date"
    panel.to_csv(reference / "factors.csv")
    write_json(
        reference / "metadata.json",
        {
            "source": "synthetic",
            "columns": list(panel.columns),
            "loaded_rows": len(panel),
            "start": str(panel.index.min().date()),
            "end": str(panel.index.max().date()),
        },
    )
    return reference


def _write_config(tmp_path: Path, reference: Path) -> Path:
    raw = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    raw["run"]["seed"] = 3
    raw["data"]["reference_dir"] = str(reference)
    raw["data"]["factors"] = ["F0", "F1", "F2"]
    raw["data"]["start"] = None
    raw["split"]["min_train"] = 24
    raw["split"]["holdout_start"] = "2005-01-31"
    raw["models"] = [
        {"name": "one_over_n", "kind": "allocator", "method": "equal"},
        {
            "name": "ridge",
            "kind": "forecast",
            "method": "ridge",
            "params": {"alpha": 10.0, "shrinkage": 0.2},
        },
        {
            "name": "moe",
            "kind": "forecast",
            "method": "moe",
            "params": {
                "n_experts": 2,
                "n_restarts": 1,
                "max_iter": 5,
                "gating_steps": 2,
            },
        },
    ]
    raw["evaluation"]["bootstrap"]["n_repetitions"] = 100
    raw["evaluation"]["deflated_sharpe"]["n_trials"] = 3
    raw["output"]["runs_dir"] = str(tmp_path / "runs")
    raw["output"]["reports_dir"] = str(tmp_path / "reports")
    raw["output"]["save_plots"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.slow
def test_end_to_end_and_determinism(tmp_path):
    market = synthetic_market(n_months=240, n_assets=3, seed=11)
    reference = _write_reference(tmp_path, market)
    config_path = _write_config(tmp_path, reference)

    first = run_pipeline(config_path, run_id="e2e_a")
    second = run_pipeline(config_path, run_id="e2e_b")

    for run_dir in (first, second):
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "report" / "report.md").exists()
        assert (run_dir / "evaluation" / "metrics.csv").exists()
        manifest = read_json(run_dir / "manifest.json")
        assert manifest["validation"]["passed"] is True
        assert manifest["artifacts"]
        expected_archive_files = 183 if (project_root() / "archive").exists() else 0
        assert manifest["archive_fingerprint"]["file_count"] == expected_archive_files
        assert "Verdict" in (run_dir / "report" / "report.md").read_text(encoding="utf-8")

    for model in ("one_over_n", "ridge", "moe"):
        weights_a = pd.read_csv(first / "models" / model / "weights.csv", index_col=0)
        weights_b = pd.read_csv(second / "models" / model / "weights.csv", index_col=0)
        np.testing.assert_allclose(weights_a.to_numpy(), weights_b.to_numpy(), atol=1e-12)

    metrics_a = pd.read_csv(first / "evaluation" / "metrics.csv", index_col=0)
    metrics_b = pd.read_csv(second / "evaluation" / "metrics.csv", index_col=0)
    np.testing.assert_allclose(
        metrics_a.select_dtypes(include=[np.number]).to_numpy(),
        metrics_b.select_dtypes(include=[np.number]).to_numpy(),
        atol=1e-10,
    )
