# Research Paper

**A Small Signal Is Not a Strategy: A Reproducible Benchmark for Probabilistic Regime-Aware Factor Allocation**

**Author:** Ken Ira Lacson Talingting  
**Version:** 2.0.0 · November 2026  
**DOI:** [10.5281/zenodo.24280434](https://doi.org/10.5281/zenodo.24280434)  
**PDF:** [`moe_factor_timing_benchmark.pdf`](moe_factor_timing_benchmark.pdf)

---

## Abstract

We ask whether a probabilistic mixture-of-experts (MoE) model can allocate across six US equity factor portfolios better than naive equal weight after realistic costs. Using 759 monthly observations from 1963 to 2026, 30 causal features, and a pre-committed 2006–2026 holdout, we estimate a three-expert linear-Gaussian MoE by expectation–maximization and map its predictive distribution into a constrained mean-variance portfolio with drift-aware turnover costs. The model produces a small full-sample cross-sectional information coefficient of 0.057 (HAC t = 2.97), but the holdout IC is 0.040 (t = 1.35). Equal weight earns a full-sample net Sharpe of 1.059 versus 0.790 for the MoE and wins all three subperiods. The holdout net Sharpe difference is −0.026 (95% CI [−0.470, 0.389]), which is inconclusive rather than evidence of equivalence. The MoE’s holdout gross Sharpe is 0.622 versus 0.511, but the difference is not statistically confirmed, and turnover of 40.6% per month produces a 0.97% annual cost drag at 10 bps. The predictive distribution is miscalibrated: 10.9% of probability integral transform values fall in each 5% tail, and none of the six assets passes a 5% Kolmogorov–Smirnov test. The contribution is a reproducible benchmark and a caution: predictive signal, gross Sharpe, and net decision value are three different objects.

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
  note         = {Research benchmark; not investment advice}
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

The independent validation record is in [`docs/13_validation_report.md`](../docs/13_validation_report.md). A fresh run reproduced 41 of 41 compared artifacts identically, and 242 independent checks passed.

The PDF is the archival version of the paper. The source build is maintained separately; contact the author through the repository for the source if needed.
