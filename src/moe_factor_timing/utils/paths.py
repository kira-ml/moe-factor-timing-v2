"""Project path helpers.

All relative paths in a config are resolved against the repository root, not
the current working directory. This makes the CLI, tests, and notebooks agree.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (the directory containing ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate
