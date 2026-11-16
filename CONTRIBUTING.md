# Contributing

This repository is a research benchmark. Contributions are welcome when they preserve the evidence contract and the public-tree rules.

## Development setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
python -m pip install -e ".[dev]" -c constraints.txt
python scripts/fetch_reference_data.py
```

The raw Ken French panel is downloaded locally and is never committed.

## Checks

```bash
python scripts/check_public_tree.py
python -m ruff check src tests scripts
python -m pytest tests/ -q
```

The public-tree guard fails if `archive/`, `runs/`, `data/reference/factors.csv`, `.env`, or a local virtual environment is tracked by git.

## Pull requests

- Keep the reference configuration (`configs/default.yaml`) unchanged unless the pull request is explicitly about a new benchmark version.
- Add a test for any new model, accounting rule, or evaluation metric.
- Do not select hyperparameters or models on the 2006–2026 holdout.
- Do not add raw third-party data. Use the downloader and provenance metadata instead.
- Report gross and net performance, turnover, costs, and calibration when changing the strategy layer.
- Update the documentation when a contract changes.

## Adding a model

Read [`docs/14_extension_guide.md`](docs/14_extension_guide.md). A model contribution should include:

1. the model class implementing `ForecastModel` or `Allocator`;
2. a registry branch with strict parameter validation;
3. a smoke configuration;
4. unit tests for shape, determinism, and no-future-data behaviour;
5. the exact command that reproduces the comparison.

## Research claims

A new result is not a claim until it survives the evaluation protocol. Report uncertainty, power, and the benchmark comparison. A negative or inconclusive result is a valid contribution.

## Data and licensing

The original code and documentation are MIT-licensed. Do not redistribute the raw Ken French panel or the V1 archive. See [`docs/12_legal_ethical_security_review.md`](docs/12_legal_ethical_security_review.md).
