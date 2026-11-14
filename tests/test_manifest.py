from __future__ import annotations

import pytest

from moe_factor_timing.run.manifest import archive_fingerprint
from moe_factor_timing.utils.paths import project_root


def test_archive_fingerprint_is_stable():
    """The archived Version 1 tree must remain byte-for-byte unchanged."""
    if not (project_root() / "archive").exists():
        pytest.skip("archive is local-only and excluded from the public git tree")
    fingerprint = archive_fingerprint()
    assert fingerprint["file_count"] == 183
    assert (
        fingerprint["aggregate_sha256"]
        == "716ff99a5cc32c63971b668bb5dc6eb0909b6fce820fafa8d1f9c1fdb9d4a058"
    )
