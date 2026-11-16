"""Fail if the public git tree contains local-only or redistributed data.

The public release must never track the raw Ken French panel, the V1 archive,
local run artifacts, or the local environment file. This guard is intentionally
simple and runs in CI before tests.
"""

from __future__ import annotations

import subprocess
import sys

FORBIDDEN_EXACT = {
    ".env",
    "data/reference/factors.csv",
}
FORBIDDEN_PREFIXES = (
    "archive/",
    "runs/",
    "data/raw/",
    "data/cache/",
    "venv/",
    ".venv/",
)


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> int:
    try:
        files = tracked_files()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"public-tree guard: could not read git index: {exc}", file=sys.stderr)
        return 1

    violations = []
    for path in files:
        if path in FORBIDDEN_EXACT or path.startswith(FORBIDDEN_PREFIXES):
            violations.append(path)

    if violations:
        print("public-tree guard: forbidden tracked paths:", file=sys.stderr)
        for path in violations:
            print(f"  - {path}", file=sys.stderr)
        return 1

    print(f"public-tree guard: ok ({len(files)} tracked files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
