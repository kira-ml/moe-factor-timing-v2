"""Run manifests: the audit trail for a result.

A manifest binds a run to its config hash, data snapshot, code commit, package
versions, and the SHA-256 of every artifact. It also records a deterministic
hash of the archived Version 1 tree, which proves the reference material was
not modified by the rebuild.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..utils.io import sha256_file, write_json
from ..utils.paths import project_root


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(project_root()),
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def archive_fingerprint() -> dict:
    """Aggregate hash of ``archive/`` without modifying it."""
    root = project_root() / "archive"
    if not root.exists():
        return {"file_count": 0, "aggregate_sha256": None}
    records = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        records.append(f"{path.relative_to(project_root()).as_posix()}:{sha256_file(path)}")
    aggregate = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    return {"file_count": len(records), "aggregate_sha256": aggregate}


def artifact_hashes(run_dir: Path) -> dict[str, dict]:
    hashes: dict[str, dict] = {}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path.name == "manifest.json":
            continue
        relative = path.relative_to(run_dir).as_posix()
        hashes[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return hashes


def write_manifest(
    run_dir: Path,
    config,
    data_metadata: dict,
    validation: dict,
    evaluation_meta: dict,
) -> Path:
    payload = {
        "run_id": run_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.config_hash(),
        "config_path": str(config.source_path),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "data": data_metadata,
        "validation": validation,
        "evaluation": evaluation_meta,
        "archive_fingerprint": archive_fingerprint(),
        "artifacts": artifact_hashes(run_dir),
    }
    path = run_dir / "manifest.json"
    write_json(path, payload)
    return path
