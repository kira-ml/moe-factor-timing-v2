# Third-Party Notices

This repository's original code and documentation are released under the MIT License (see `LICENSE`). The following third-party components are used at runtime or in the publication build. They are not vendored except where noted.

## Python dependencies

| Package | License | Copyright / project |
|---|---|---|
| NumPy | BSD 3-Clause | NumPy Developers |
| pandas | BSD 3-Clause | AQR Capital Management, Lambda Foundry, PyData Development Team |
| SciPy | BSD 3-Clause | SciPy Developers |
| scikit-learn | BSD 3-Clause | scikit-learn developers |
| matplotlib | PSF-based / BSD-compatible | Matplotlib Development Team |
| PyYAML | MIT | Kirill Simonov and contributors |
| pytest | MIT | Holger Krekel and contributors |
| Ruff | MIT | Charles Marsh and contributors |

These packages are installed by the user via `pip`; their source is not redistributed in this repository. Their license texts are available in the installed package metadata and upstream repositories.

## Fonts

The one-page publication embeds subsets of **Source Serif 4** and **IBM Plex Sans / Mono**.

- Source Serif 4: Copyright (c) 2014-2023 Adobe, SIL Open Font License 1.1.
- IBM Plex Sans / Mono: Copyright (c) 2017 IBM Corp., SIL Open Font License 1.1.

The full license texts are in `publication/FONT_LICENSE.txt` and `publication/fonts/OFL-*.txt`. The fonts are embedded for document rendering; they are not sold or distributed as standalone font software.

## Build tool

The PDF is rendered with headless Google Chrome, which is a local build tool and is not distributed with this repository.

## Data

The Ken French factor panel and the archived Version 1 data are **not redistributed** in the public git tree. They are downloaded or retained locally. See `NOTICE` and `docs/12_legal_ethical_security_review.md`.

## Trademarks

Fama-French, CRSP, Yahoo Finance, FRED, CBOE, Google Chrome, and LinkedIn are trademarks or service marks of their respective owners. Their use here is descriptive and does not imply endorsement.
