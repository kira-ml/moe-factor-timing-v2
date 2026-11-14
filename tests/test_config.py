from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from moe_factor_timing.config import ConfigError, ModelConfig, load_config
from moe_factor_timing.models.registry import build_forecast_model


def test_default_config_loads(default_config):
    assert default_config.data.factors == ("MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM")
    assert default_config.split.min_train == 120
    assert default_config.decision.constraints.fully_invested is True


def test_missing_section_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("run:\n  seed: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_duplicate_models_raise(tmp_path, default_config):
    raw = yaml.safe_load(default_config.source_path.read_text(encoding="utf-8"))
    raw["models"].append(dict(raw["models"][0]))
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(path)


def test_unknown_model_parameter_raises():
    spec = ModelConfig(
        name="ridge",
        kind="forecast",
        method="ridge",
        params={"alpha": 1.0, "not_a_parameter": 3},
    )
    with pytest.raises(ValueError, match="unknown parameters"):
        build_forecast_model(spec, seed=1)


def test_config_hash_changes_with_seed(default_config):
    changed = replace(default_config, run=replace(default_config.run, seed=99))
    assert changed.config_hash() != default_config.config_hash()


def test_invalid_max_weight_raises(tmp_path, default_config):
    raw = yaml.safe_load(default_config.source_path.read_text(encoding="utf-8"))
    raw["decision"]["constraints"]["max_weight"] = 2.0
    path = tmp_path / "bad_weight.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="max_weight"):
        load_config(path)
