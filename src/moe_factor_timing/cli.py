"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import ConfigError, load_config
from .data.french import fetch_reference_data
from .data.loader import load_market_data
from .data.validate import validate_market_data
from .run.orchestrator import PipelineError, run_pipeline
from .utils.paths import resolve_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe-factor-timing",
        description="A reproducible benchmark for regime-aware factor allocation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute a full research run")
    run_parser.add_argument("--config", required=True, help="path to a YAML config")
    run_parser.add_argument("--run-id", default=None, help="override the run directory name")
    run_parser.add_argument(
        "--publish",
        action="store_true",
        help="copy the report and figures into the configured reports directory",
    )

    validate_parser = subparsers.add_parser("validate", help="validate the reference data only")
    validate_parser.add_argument("--config", required=True, help="path to a YAML config")

    fetch_parser = subparsers.add_parser(
        "fetch-data", help="download the Ken French reference panel"
    )
    fetch_parser.add_argument("--output", default="data/reference", help="output directory")
    fetch_parser.add_argument("--force", action="store_true", help="replace an existing snapshot")

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    run_dir = run_pipeline(args.config, run_id=args.run_id, publish=args.publish)
    print(f"run complete: {run_dir}")
    print(f"report: {run_dir / 'report' / 'report.md'}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    market = load_market_data(config)
    report = validate_market_data(market, config)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


def _cmd_fetch(args: argparse.Namespace) -> int:
    output = resolve_path(args.output)
    metadata = fetch_reference_data(output, force=args.force)
    print(json.dumps(metadata, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "fetch-data":
            return _cmd_fetch(args)
        parser.error(f"unknown command: {args.command}")
    except (ConfigError, FileNotFoundError, FileExistsError, PipelineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 2
