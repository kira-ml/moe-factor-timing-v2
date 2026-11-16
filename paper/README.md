# Research Paper

**A Small Signal Is Not a Strategy: A Reproducible Benchmark for Probabilistic Regime-Aware Factor Allocation**

**Author:** Ken Ira Lacson Talingting  
**Version:** 2.0.0 · Revision 1 (post-publication audit) · November 2026  
**DOI:** [10.5281/zenodo.24280434](https://doi.org/10.5281/zenodo.24280434) — archives v2.0.0  
**PDF:** [`moe_factor_timing_benchmark.pdf`](moe_factor_timing_benchmark.pdf) — Revision 1

---

## Abstract

We ask whether a probabilistic mixture-of-experts (MoE) model can allocate across six US equity factor portfolios better than naive equal weight after realistic costs. Using a 759-month raw panel (627 out-of-sample decisions) and a 2006–2026 holdout pre-specified before the V2 reference run, with prior exposure to V1, we estimate a three-expert linear-Gaussian MoE and map its predictive distribution into a constrained mean-variance portfolio. The full-sample rank IC is 0.057 (HAC t = 2.97), but it is a one-month effect: under a one-month information lag it falls to 0.009 (t = 0.50). The holdout rank IC is 0.040 (t = 1.35); the Pearson IC is 0.063 (t = 2.11), so the holdout signal is metric-dependent. The reference decision rule is near-linear: at risk aversion 2, the covariance term contributes 3.8% of the mean absolute forecast gradient. The pre-specified H2 is rejected: static mean-variance beats the MoE in both samples (0.874 vs 0.790 full; 0.509 vs 0.479 holdout). The six-factor 1/N has a higher holdout net Sharpe than the MoE (0.505 vs 0.479; difference −0.026, 95% CI [−0.470, 0.389], inconclusive), but a four-factor 1/N has a lower holdout Sharpe (0.408), and a market-only portfolio has 0.652. The full-sample 1/N difference is −0.269 (95% CI [−0.518, 0.006], p = 0.058), borderline rather than significant. Turnover is 40.6% one-way per month; the allocation-layer cost drag is 0.97% per year at 10 bps per side. The predictive distribution is miscalibrated (10.9% in each 5% tail; 0/6 KS). The result is a reproducible negative/inconclusive benchmark about this pipeline, information set, and benchmark, not a claim about probabilistic regime-aware allocation in general; the 242 checks are internal recomputation, not independent replication.

**Keywords:** factor timing; mixture of experts; regime models; portfolio allocation; transaction costs; reproducible research.  
**JEL:** G11; G17; C53; C58.

---

## Citation

```bibtex
@misc{talingting2026moefactor,
  author       = {Talingting, Ken Ira Lacson},
  title        = {moe-factor-timing v2: A reproducible benchmark for probabilistic regime-aware factor allocation},
  year         = {2026},
  version      = {2.0.0},
  howpublished = {GitHub},
  url          = {https://github.com/kira-ml/moe-factor-timing-v2},
  doi          = {10.5281/zenodo.24280434},
  note         = {Research benchmark; not investment advice; Revision 1 (post-publication audit)}
}
```

The repository also includes [`CITATION.cff`](../CITATION.cff).

---

## Reproducibility

The paper is generated from the verified reference run; the run artifacts are not redistributed. The core claims are reproducible with:

```bash
python -m pip install -e ".[dev]" -c constraints.txt
python scripts/fetch_reference_data.py
python -m moe_factor_timing run --config configs/default.yaml --run-id reference --publish
```

The internal validation record is in [`docs/13_validation_report.md`](../docs/13_validation_report.md). A fresh run reproduced 41 of 41 compared artifacts identically, and 242 independent recomputation checks passed. The post-publication audit diagnostics are reproducible with:

```bash
python scripts/paper_audit_checks.py --run-dir runs/reference
python scripts/lagged_information_check.py
python -m moe_factor_timing run --config configs/restart3.yaml --run-id restart3
```

The v2.0.0 PDF is archived at the DOI. The PDF in this repository is Revision 1; it corrects the full-sample comparison, clarifies the validation scope, and adds block-permutation, signal-delay, and gating diagnostics. Bit-for-bit reproduction requires the 2026-09-30 data snapshot (SHA-256 `cc805ab0…`); the public downloader fetches the current Ken French files, which may differ.
