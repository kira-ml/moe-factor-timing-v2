# One-Page Publication: Design and Credibility Rationale

**Artifact:** `publication/one_pager.pdf` (A4, one page) and `publication/one_pager.png` (200 dpi)
**Source:** `runs/reference` + `reports/red_team_checks.json`
**Builder:** `scripts/build_onepager.py`

---

## 1. The communication problem

The result is a negative/inconclusive benchmark. A conventional research note would either bury that fact or over-explain the machinery. Both are weak:

- Burying the negative result looks like marketing with footnotes.
- Over-explaining the machinery looks defensive and crowds the page.

The design principle is therefore:

> **Show the evidence trail. Do not describe the effort.**

Every credibility signal on the page is an auditable artifact, a number with uncertainty, or a falsification criterion. There are no adjectives about the author, the project, or the work.

---

## 2. What the reader can verify at a glance

| Signal | Where it appears |
|---|---|
| Long sample and out-of-sample design | `759 monthly observations · 627 OOS decisions · 248-month holdout` in the method strip |
| Hard benchmark | `1/N` in the title deck, stat block, charts, and results |
| Realistic cost accounting | `10 bps base · 5/20 bps sensitivity · drift-aware turnover` |
| Statistical uncertainty | Confidence intervals in the stat block and charts; `MDE 0.62` |
| HAC and block bootstrap | `6-lag HAC · 1,000 block-bootstrap draws` in the method strip |
| Calibration testing | PIT histogram; `10.9% tails · 0/6 assets pass KS` |
| Falsification | "What it does not support" box; explicit negative result |
| Model-risk awareness | `one initialization; restart audit shows local-optimum sensitivity` |
| Decision-rule robustness | `λ 2/5/10 · caps 0.25/0.35/0.50 · covariance scales 1/2/4` in the report and red-team checks |
| Reproducibility | `Config 2a035de · data SHA-256 cc805ab0 · archive 716ff99a`; code/report link |
| Engineering validation | `45 tests · red-team gate` in the method strip; `tests/` and `docs/11_red_team_validation.md` |
| Adversarial review | `docs/11_red_team_validation.md`; publication gate |

The reader does not need to trust a claim about rigor. They can see the holdout, the benchmark, the cost sweep, the calibration test, the uncertainty, and the negative result.

---

## 3. What is deliberately absent

- No author biography, headshot, or "about me."
- No statements about time invested, effort, or personal capability.
- No "novel," "state-of-the-art," "comprehensive," "rigorous," or "flagship" adjectives.
- No engagement call-to-action or LinkedIn metrics.
- No claim of live performance, capacity, or generalizability beyond the tested universe.
- No "gross alpha" headline. The gross number is shown only with its confidence interval and the explanation that it is not confirmed in the holdout.
- No 5 bps cherry-pick. The 5 bps result is shown, labeled as within noise, and subordinated to the pre-committed 10 bps result.

The absence is part of the signal. A page that does not sell itself is easier to believe.

---

## 4. The one-page argument

1. **Hook:** "A Small Signal Is Not a Strategy."
2. **Question:** Can a probabilistic MoE turn a small cross-sectional signal into net factor-allocation value?
3. **Evidence:** 63 years, 627 OOS decisions, 248-month holdout, 1/N benchmark, 10 bps base cost.
4. **Result:** Full-sample 1/N wins; holdout inconclusive; the net sign flips between 5 and 10 bps; the MoE is miscalibrated and high-turnover.
5. **Falsification:** The holdout gross edge and holdout IC are not confirmed; the 5 bps result is within noise.
6. **Contribution:** Predictive signal, gross Sharpe, and net decision value are three different objects.
7. **Audit:** Config, data, archive hashes; code and red-team checks.

This is a research note, not a victory lap. The page earns attention by taking the result seriously enough to report what failed.

---

## 5. Visual system

- **Format:** A4 portrait, 210 × 297 mm, one page.
- **Palette:** warm ivory `#F7F5F0`, navy `#0B1F33`, slate `#3E4C59`, oxblood `#7A2633`, brass `#A3854F`, stone rule `#D5CFC3`.
- **Typography:** Source Serif 4 for the display title; IBM Plex Sans for body and labels; IBM Plex Mono for numeric hashes.
- **Hierarchy:** 29.5 pt title, 8.4 pt deck, 8.2 pt lead, 7.1 pt body, 5.9–6.8 pt captions and lists.
- **Charts:** custom matplotlib, transparent backgrounds, no chartjunk, direct labels, no 3D or decorative effects.
- **Structure:** masthead → abstract + at-a-glance → cost chart + IC/PIT → subperiod + evidence gate → method/footer.
- **Density:** high but readable; the page is designed to be zoomed on a phone, not skimmed as a poster.

The visual language is closer to an institutional research brief than a corporate infographic.

---

## 6. Data and numerical consistency

All numbers are generated from:

- `runs/reference/evaluation/metrics.csv`
- `runs/reference/evaluation/holdout_metrics.csv`
- `runs/reference/evaluation/subperiods.csv`
- `runs/reference/models/moe/pit.csv`
- `reports/red_team_checks.json`

The builder does not hard-code the main results. It formats the saved artifacts. If the run changes, the page changes; if the red-team checks change, the page changes. The raw Ken French panel is downloaded locally and is not redistributed; see `docs/12_legal_ethical_security_review.md`.

---

## 7. Regenerate

```bash
python scripts/red_team_checks.py --run-dir runs/reference
python scripts/build_onepager.py
```

Outputs:

- `publication/one_pager.html`
- `publication/one_pager.pdf`
- `publication/one_pager.png`

The build validates that the PDF is exactly one page and that the PNG is 1653 × 2339 at 200 dpi.

---

## 9. Publication and legal status

- The one-page note is a research communication, not a promotional document.
- The author's full name and the GitHub link appear as attribution and reproducibility pointers, not as marketing.
- The PDF uses Source Serif 4 / IBM Plex font subsets under the SIL Open Font License 1.1; see `FONT_LICENSE.txt`.
- The raw Ken French panel and the V1 archive are **not redistributed** in the public tree; see `../NOTICE` and `../docs/12_legal_ethical_security_review.md`.
- The public repository history still contains earlier data snapshots; a fully clean public release requires a fresh repository or a history rewrite.

---

## 10. Publication gate

The one-pager is approved only under the gate in `docs/11_red_team_validation.md`:

- It reports a negative/inconclusive benchmark, not a positive alpha claim.
- It distinguishes full-sample IC from holdout IC.
- It shows the cost-sensitivity sign flip.
- It discloses the single initialization and reconstructed panel.
- It does not claim live performance or generalization beyond the tested universe.

If a future version changes the thesis, it must pass the same gate again.
