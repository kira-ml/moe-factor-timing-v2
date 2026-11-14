"""Download and freeze the Ken French reference panel.

Usage:
    python scripts/fetch_reference_data.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_factor_timing.data.french import fetch_reference_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/reference", help="output directory")
    parser.add_argument("--force", action="store_true", help="replace an existing snapshot")
    args = parser.parse_args()
    metadata = fetch_reference_data(args.output, force=args.force)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
