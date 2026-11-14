"""Deterministic seed handling.

The benchmark deliberately avoids global RNG side effects where possible. This
module is the single place where global seeds are set, and every stochastic
model receives an explicit seed instead of reading the global RNG.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set the global seeds used by Python, NumPy, and hash randomization.

    Hash randomization cannot be changed after interpreter start; we set the
    environment variable for child processes and document the limitation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def rng(seed: int) -> np.random.Generator:
    """Return a private NumPy generator, avoiding global-state coupling."""
    return np.random.default_rng(seed)
