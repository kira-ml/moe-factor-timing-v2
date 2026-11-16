# 12 - Legal, Ethical, and Security Review

**Status:** Publication gate
**Date:** 2026-11-14; updated 2026-11-16
**Scope:** the local repository, the one-page publication, and the public GitHub link.
**Reviewer:** AI-assisted compliance audit. **This is not legal advice.** Items marked "professional review" require qualified counsel.

> **Update (2026-11-16).** The public release now lives in a fresh repository, [`kira-ml/moe-factor-timing-v2`](https://github.com/kira-ml/moe-factor-timing-v2), with clean history. The legacy repository is private. The public-history residual identified in the 2026-11-14 review is resolved for the public link. The local development repository still contains the old history.

---

## 1. Executive summary

| Area | Status | Finding |
|---|---|---|
| Original code license | **Confirmed** | MIT; original code and documentation owned by the author. |
| Python dependencies | **Confirmed permissive** | BSD-3, MIT, PSF-compatible. No GPL/AGPL runtime dependency. |
| Fonts in the PDF | **Resolved** | Switched from Microsoft system fonts to Source Serif 4 + IBM Plex Sans/Mono (SIL OFL 1.1). PDF embeds open-license subsets. |
| Secrets and credentials | **Confirmed clean in git** | `.env` is untracked and was never committed; no API keys, tokens, or private keys found in the current tree or history. |
| Personal data | **Confirmed** | Aggregate market data only; no PII beyond the author's name. |
| Ken French data | **Unresolved** | Copyright notice, no explicit redistribution license. Raw CSV removed from the public tree; downloader provided. |
| V1 archive data | **Unresolved** | Contains third-party ETF/VIX/FRED data. Kept local-only; removed from the public tree. |
| Git history | **Resolved for the public link** | Fresh public repository `kira-ml/moe-factor-timing-v2` with clean history; legacy repository private. Local development history is not public. |
| Research claims | **Confirmed after red-team gate** | Negative/inconclusive result, confidence intervals, cost sensitivity, calibration, and limitations disclosed. |
| LinkedIn publication | **Approved** | The one-pager is safe as a research note; the public link points to a clean repository. |

**Gate decision:** The one-page research note is approved for publication as a negative/inconclusive research note. The public repository link is approved: it points to a fresh repository with clean history, 120 tracked files, no raw Ken French panel, no V1 archive, and no `runs/`. The legacy development repository is private. Raw-data **redistribution** remains unresolved; the public tree ships a downloader, not the data.

---

## 2. Data legality and privacy

### 2.1 Ken French Data Library

- **Source:** https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **Observed terms:** The page displays a copyright notice ("Copyright Eugene F. Fama and Kenneth R. French") but no explicit open-source, Creative Commons, or redistribution license.
- **Use:** The data is publicly downloadable and widely used in academic research.
- **Redistribution:** Not explicitly granted. Free access to use is not the same as permission to redistribute.
- **Action taken:** The raw `data/reference/factors.csv` is removed from the public git tree and kept local-only. The repository ships a downloader (`scripts/fetch_reference_data.py`) and provenance metadata (`data/reference/metadata.json`).
- **Status:** **Unresolved for redistribution.** Professional review is recommended if the raw panel is to be redistributed.
- **Residual:** The raw CSV is absent from the public tree and public history. The legacy development repository is private.

### 2.2 V1 archive

- **Contents:** yfinance ETF price files (SPY, IWD, MTUM, QUAL, USMV), CBOE VIX data via Yahoo, FRED macro data, derived returns, logs, PDFs, and old code.
- **Concerns:** Yahoo Finance terms, CBOE index data terms, FRED attribution; these are not explicitly cleared for redistribution in this repository.
- **Instruction:** The archive must be preserved without modification.
- **Action taken:** The archive remains on disk, byte-identical and fingerprinted, but is removed from the public git tree and ignored. Its local fingerprint is checked by `tests/test_manifest.py` when present.
- **Status:** **Unresolved for public disclosure.** Professional review or removal from the public history is required.
- **Residual:** The archive is absent from the public tree and public history. The legacy development repository is private.

### 2.3 FRED

- US federal macroeconomic data is generally public domain, but FRED's terms may require attribution and may restrict redistribution of the FRED-branded service. The current project no longer uses FRED in the main analysis; the V1 archive contains a derived FRED file. Cite the source and do not imply endorsement.

### 2.4 Personal data

- No personal data, client data, account data, or confidential information is used.
- The only personal information in the publication is the author's name, provided at the author's request.
- No cookies, tracking, or user-level data are collected by the repository or the PDF.

---

## 3. Methodology and compliance

- **Research design:** chronological walk-forward, pre-committed 2006-2026 holdout, 1/N benchmark, drift-aware costs, 10 bps base cost with 5/20/30 bps sensitivity.
- **Statistical reporting:** block-bootstrap CIs, HAC standard errors, PIT/KS calibration, MDE, deflated Sharpe, subperiod stability.
- **Falsification:** red-team gate in `docs/11_red_team_validation.md`; the holdout gross/IC claims were rejected as unconfirmed.
- **No investment advice:** the report and one-pager state that this is research, not advice, and that backtests do not imply live performance.
- **No misleading performance claim:** the one-pager leads with the negative/inconclusive result and explicitly lists what the evidence does not support.
- **Retrospective panel:** the Ken French panel is reconstructed, not vintage; this is disclosed.
- **AI assistance:** the code and documents were prepared with AI assistance. Platform or jurisdiction-specific AI-disclosure rules should be checked before posting; the research claims remain the author's responsibility.

---

## 4. Intellectual property and open-source compliance

| Component | License | Compliance |
|---|---|---|
| Original code/docs | MIT | `LICENSE` included. |
| NumPy | BSD-3-Clause | Permissive; attribution in `THIRD_PARTY_NOTICES.md`. |
| pandas | BSD-3-Clause | Permissive; attribution included. |
| SciPy | BSD-3-Clause | Permissive; attribution included. |
| scikit-learn | BSD-3-Clause | Permissive; attribution included. |
| matplotlib | PSF-based / BSD-compatible | Permissive; attribution included. |
| PyYAML | MIT | Permissive; attribution included. |
| pytest | MIT | Development-only; attribution included. |
| Ruff | MIT | Development-only; attribution included. |
| Source Serif 4 / IBM Plex | SIL OFL 1.1 | Permissive; full texts in `publication/FONT_LICENSE.txt` and `publication/fonts/`. |
| Headless Chrome | Proprietary build tool | Not distributed; used locally to render the PDF. |
| Ken French data | Copyright; no explicit license | **Not redistributed in public tree; unresolved.** |
| V1 archive data | Third-party terms | **Not redistributed in public tree; unresolved.** |

No GPL/AGPL runtime dependency was found. The dependencies are installed by the user, not vendored. The PDF embeds only Source Serif 4 / IBM Plex subsets, not Microsoft system fonts.

---

## 5. Security and information disclosure

### Confirmed

- `.env` is present locally but **untracked** and **not present in git history**. It contains a `FRED_API_KEY` name; the value was never printed or committed.
- `runs/` is git-ignored and not public.
- `archive/v1/logs/` is local-only and ignored.
- No AWS keys, GitHub tokens, private keys, passwords, or generic API secrets were found in the current tree or git history by pattern scan.
- No absolute local paths remain in the public report; the generated report was updated to use `runs/reference` and `configs/default.yaml`.
- The one-pager contains no credentials, internal paths, or confidential data.

### Residual

- The public repository has clean history and contains no raw Ken French panel or V1 archive. The legacy development repository still contains the old history, but it is private.
- The public repository may be indexed or cached by third parties; because it contains no raw third-party data, the residual is reputational rather than data-disclosure.
- If the FRED key was ever used in a public notebook or shared environment, rotate it as a precaution. No evidence of exposure was found in this repository.

---

## 6. Publication safety

| Check | Status |
|---|---|
| Numbers match saved artifacts | Confirmed; generated from `runs/reference` + `reports/red_team_checks.json`. |
| Negative/inconclusive result stated | Confirmed. |
| Holdout vs full-sample IC distinguished | Confirmed. |
| Cost sensitivity shown (5/10/20 bps) | Confirmed; 5 bps is labeled within noise. |
| Calibration failure disclosed | Confirmed; 10.9% tails, 0/6 KS. |
| Single-initialization risk disclosed | Confirmed. |
| Reconstructed panel disclosed | Confirmed. |
| No live-performance claim | Confirmed. |
| No investment advice | Confirmed; footer disclaimer. |
| Author name and GitHub link | Added at the author's request. |
| Self-promotional language | Avoided; no bio, effort claims, or adjectives about the work. |
| Fonts | Source Serif 4 / IBM Plex subsets; SIL OFL 1.1 license included. |
| Data redistribution | Raw data absent from the public tree and public history; legacy repository private. |

---

## 7. Public GitHub history

The 2026-11-14 review identified a residual history risk in the original public repository. Remediation was executed:

- The public release now lives in a **fresh repository**, [`kira-ml/moe-factor-timing-v2`](https://github.com/kira-ml/moe-factor-timing-v2), created from a clean orphan snapshot.
- The public tree contains **120 tracked files**, no `archive/`, no `data/reference/factors.csv`, and no `runs/`.
- The public git history is clean: the raw Ken French panel and the V1 archive were never committed to the new repository.
- The legacy repository `kira-ml/mixture-of-experts-factor-timing` was made **private**.
- The local development repository still contains the old history, but it is not public and must not be linked publicly.

**Residual:** raw-data redistribution terms remain unresolved; do not redistribute the raw panel. Third-party caches or indexes of the legacy public repository may persist, but the legacy repository is private and the public release contains no raw data or archive.

---

## 8. Required actions before publication

- [x] Remove raw Ken French CSV from the public tree; add downloader.
- [x] Remove V1 archive from the public tree; keep local-only and fingerprinted.
- [x] Switch PDF fonts to Source Serif 4 / IBM Plex; include font license.
- [x] Add full name and GitHub link.
- [x] Remove absolute local paths from the public report.
- [x] Add `NOTICE`, `THIRD_PARTY_NOTICES.md`, and this review.
- [x] Confirm no secrets in current tree or git history.
- [x] Confirm the claims passed the red-team gate.
- [x] Decide on history remediation: fresh public repository, legacy repository private.
- [x] Point the public link at the clean repository; the historical-data disclosure is no longer required for the public link.
- [ ] If the raw data must be redistributed, obtain permission or professional legal review.
- [ ] Check platform-specific AI-disclosure and financial-promotion rules for LinkedIn.

---

## 9. Publication decision

**One-pager:** approved as a negative/inconclusive research note. The name and repository link are included naturally, and the page is not self-promotional.

**Repository HEAD:** approved. The public repository contains no raw Ken French panel, no V1 archive, and no `runs/`; the downloader and provenance metadata are documented.

**Repository history:** **clean for the public repository.** The legacy development repository remains private. Raw-data redistribution remains unresolved, so the public repository must not add the raw panel or archive.

**Overall:** **Pass for the public release**, with the data-redistribution question still requiring professional review if the raw panel is ever to be redistributed. This is not legal advice.
