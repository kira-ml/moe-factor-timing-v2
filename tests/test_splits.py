from __future__ import annotations

import pandas as pd
import pytest

from moe_factor_timing.splits.walk_forward import check_split_plan, make_walk_forward_splits


def test_walk_forward_is_chronological(features, default_config):
    plan = make_walk_forward_splits(features.decision_dates, features.target_dates, default_config)
    assert plan.splits[0].test_idx == default_config.split.min_train
    assert plan.splits[-1].test_idx == len(features.decision_dates) - 1
    for split in plan.splits:
        assert split.train_start_idx == 0
        assert split.train_end_idx == split.test_idx - 1
        assert split.decision_date < split.target_date
    check_split_plan(plan)


def test_holdout_is_contiguous_suffix(features, default_config):
    plan = make_walk_forward_splits(features.decision_dates, features.target_dates, default_config)
    holdout = [split for split in plan.splits if split.is_holdout]
    pre = [split for split in plan.splits if not split.is_holdout]
    assert holdout
    assert pre
    assert pre[-1].test_idx < holdout[0].test_idx
    assert holdout[0].decision_date >= pd.Timestamp(default_config.split.holdout_start)


def test_too_few_observations_raises(features, default_config):
    with pytest.raises(ValueError, match="not enough observations"):
        make_walk_forward_splits(
            features.decision_dates[:50], features.target_dates[:50], default_config
        )


def test_test_step_is_respected(features, default_config):
    from dataclasses import replace

    config = replace(default_config, split=replace(default_config.split, test_step=2))
    plan = make_walk_forward_splits(features.decision_dates, features.target_dates, config)
    steps = [b.test_idx - a.test_idx for a, b in zip(plan.splits, plan.splits[1:], strict=False)]
    assert all(step == 2 for step in steps)
