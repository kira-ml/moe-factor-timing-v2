"""Chronological walk-forward splits.

No shuffling, no random splits, no overlap between the training block and the
evaluation month. The training window is expanding by default, which matches
the information set of a real-time allocator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ..config import Config


@dataclass(frozen=True)
class Split:
    train_start_idx: int
    train_end_idx: int
    test_idx: int
    decision_date: pd.Timestamp
    target_date: pd.Timestamp
    is_holdout: bool

    @property
    def n_train(self) -> int:
        return self.train_end_idx - self.train_start_idx + 1


@dataclass(frozen=True)
class SplitPlan:
    splits: tuple[Split, ...]
    min_train: int
    test_step: int
    holdout_start: pd.Timestamp | None

    def to_dict(self) -> dict:
        return {
            "n_splits": len(self.splits),
            "min_train": self.min_train,
            "test_step": self.test_step,
            "holdout_start": None if self.holdout_start is None else str(self.holdout_start.date()),
            "first_decision": str(self.splits[0].decision_date.date()),
            "last_decision": str(self.splits[-1].decision_date.date()),
            "n_holdout": sum(split.is_holdout for split in self.splits),
            "splits": [asdict(split) for split in self.splits],
        }


def make_walk_forward_splits(
    decision_dates: pd.DatetimeIndex,
    target_dates: pd.DatetimeIndex,
    config: Config,
) -> SplitPlan:
    if len(decision_dates) != len(target_dates):
        raise ValueError("decision_dates and target_dates must have equal length")
    if not decision_dates.is_monotonic_increasing:
        raise ValueError("decision_dates must be sorted")

    holdout_start = (
        pd.Timestamp(config.split.holdout_start) if config.split.holdout_start is not None else None
    )
    splits: list[Split] = []
    for test_idx in range(config.split.min_train, len(decision_dates), config.split.test_step):
        decision_date = pd.Timestamp(decision_dates[test_idx])
        splits.append(
            Split(
                train_start_idx=0,
                train_end_idx=test_idx - 1,
                test_idx=test_idx,
                decision_date=decision_date,
                target_date=pd.Timestamp(target_dates[test_idx]),
                is_holdout=bool(holdout_start is not None and decision_date >= holdout_start),
            )
        )
    if not splits:
        raise ValueError(
            f"not enough observations for min_train={config.split.min_train}: "
            f"n={len(decision_dates)}"
        )

    plan = SplitPlan(
        splits=tuple(splits),
        min_train=config.split.min_train,
        test_step=config.split.test_step,
        holdout_start=holdout_start,
    )
    check_split_plan(plan)
    return plan


def check_split_plan(plan: SplitPlan) -> None:
    previous_test = -1
    for split in plan.splits:
        if split.train_start_idx != 0:
            raise AssertionError("this benchmark uses an expanding window starting at row 0")
        if split.train_end_idx != split.test_idx - 1:
            raise AssertionError("training block must end immediately before the test row")
        if split.test_idx <= previous_test:
            raise AssertionError("test indices must increase")
        if split.decision_date >= split.target_date:
            raise AssertionError("decision date must precede target date")
        previous_test = split.test_idx
    if plan.holdout_start is not None:
        holdout = [split for split in plan.splits if split.is_holdout]
        pre = [split for split in plan.splits if not split.is_holdout]
        if holdout and pre and pre[-1].test_idx >= holdout[0].test_idx:
            raise AssertionError("holdout must be a contiguous suffix")
