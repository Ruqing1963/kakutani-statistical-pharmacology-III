# Statistical Pharmacology via Kakutani Dichotomy III

**Spectral Kakutani–Feldman–Hájek Criticality of Correlated Conformational Modes: Mode Amplitude Perturbation, Iso-Spectral Givens Rotation, and Tail-Increment Finite-Size Elimination**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23015709.svg)](https://doi.org/10.5281/zenodo.23015709)

Zhengyi Chen<sup>1</sup>, Ruqing Chen<sup>2</sup>

<sup>1</sup> Guangxi Key Laboratory of Drug Discovery and Optimization, School of Pharmacy, Guilin Medical University, Guilin 541199, P. R. China — chenzhengyi@glmc.edu.cn
<sup>2</sup> GUT Geoservice Inc., Montreal, Québec, Canada — ruqing@hotmail.com

Paper III of the series *Statistical Pharmacology via Kakutani Dichotomy*. Paper I (DOI 10.5281/zenodo.23005921) treated product ensembles; Paper II (DOI 10.5281/zenodo.23012217) proved the commuting–rotation decomposition `D_cov = D_cov^comm + D_cov^rot` of the correlated Kakutani index. This paper asks when each part diverges if a ligand's effect on the collective modes decays along the mode index, and introduces the doubling tail-increment estimator that removes the finite-size bias of the scaling exponent near the critical point.

## Status

Experiment 03 and the paper (`paper/paper03_spectral_kakutani_criticality.pdf`) are complete. Main results: unified three-regime Euler–Maclaurin asymptotics and the spectral Feldman–Hájek dichotomy at `alpha_c = 1/2` for mode amplitudes and for mode orientations separately (Theorem 3.1); constant-term annihilation by the doubling tail increment (Theorem 4.1); the refined naive-slope bias formula, exact to four decimals (Theorem 4.2); the geodesic parity theorem explaining the residual `+0.0028` shift of the Euclidean mechanism through the curvature of the coordinate `delta = e^eta - 1` on the SPD cone (Theorem 4.3); the joint `(alpha_lam, alpha_theta)` phase diagram (Theorem 5.1) with the finite-N two-power-law cross-over formula that reproduces all 361 measured grid exponents to 4e-4 (Corollary 5.3). The analytic companion `code/paper03_compute_tables.py` writes `results/paper03_tables_summary.txt`.

## Experiment 03: results at a glance

Two spectral perturbation mechanisms, both normalised to the leading coefficient `A = 0.0100` of Paper I:

- **Mechanism A (commuting amplitude perturbation)**: `lambda_i^B = lambda_i^A (1 + 0.4 i^{-alpha_lam})`, common eigenbasis, `D_cov = D_cov^comm`. Per-mode contribution `d_i = (1/2) ln[(1 + x/2)/sqrt(1 + x)] = x^2/16 - x^3/16 + 7x^4/128 + ...`, `x = 0.4 i^{-alpha_lam}` (coefficients verified numerically to `O(x^5)`).
- **Mechanism B (pure mode rotation)**: all eigenvalues unchanged (`D_cov^comm = 0` exactly); the k-th adjacent mode pair with anisotropy ratio `r = 3` is rotated by a Givens angle `theta_k = sqrt(0.06) k^{-alpha_theta}`; per-pair contribution `d_k = (1/2) ln[1 + (r-1)^2 sin^2(theta_k)/(4r)] = theta^2/6 - theta^4/12 + ...`.

Feldman–Hájek dichotomy: the ensembles are equivalent iff `alpha > 1/2`, for each mechanism separately. At `alpha = 1/2` the slope of `D_cov` on `ln N` is `0.00995` (A) and `0.01000` (B) against the theoretical `0.0100`.

**Doubling tail-increment estimator.** `D_tail(N) = D_cov(2N) - D_cov(N)` cancels the Euler–Maclaurin constant `A zeta(2 alpha)` whose pole at `2 alpha = 1` biases the naive regression of `ln D_cov` on `ln N`. On the window `[10^3, 10^5]`:

| alpha | 1 − 2α | naive γ (A) | Paper I prediction | doubling tail γ (A) | doubling tail γ (B) |
|---|---|---|---|---|---|
| 0.25 | +0.50 | 0.5201 | 0.5037 | +0.5092 | +0.5002 |
| 0.45 | +0.10 | 0.1729 | 0.1601 | +0.1028 | +0.1001 |
| 0.50 | 0.00 | 0.1130 | 0.1022 | +0.0020 | +0.0001 |
| 0.55 | −0.10 | 0.0687 | 0.0603 | −0.0986 | −0.0999 |
| 0.75 | −0.50 | 0.0054 | 0.0039 | −0.4996 | −0.4999 |

At `alpha = 0.45` the bias falls from `+0.073` (naive) to `+0.003` (A, the remaining residual is the sub-leading `N^{1-3alpha}` term of the odd expansion, reproduced by the analytic doubling-tail prediction to `0.002`) and from `+0.062` to `+0.0001` (B): reduction factors 26 and 1005. `gamma_tail^+ = max(gamma_tail, 0)` is the sharp phase curve `max(1 - 2alpha, 0)`.

**Joint phase diagram.** On the 19 × 19 grid `(alpha_lam, alpha_theta)` in `[0.05, 0.95]^2` with both mechanisms acting on 2 × 2 blocks (exact block determinants), the measured `gamma_tail^+` reproduces `max(1 - 2alpha_lam, 1 - 2alpha_theta, 0)` with 0 phase-classification mismatches out of 361 points: the ensembles are singular iff `min(alpha_lam, alpha_theta) <= 1/2`.

## Repository layout

```
.
├── code/
│   ├── exp03_spectral_kakutani_criticality.py    # mechanisms A/B, benchmarks, exponent scan, phase diagram, figure
│   └── paper03_compute_tables.py                 # series coefficients, EM predictions, limits, geodesic variant, tables
├── data/
│   ├── exp03_results_benchmark.csv               # D_cov^comm, D_cov^rot, D_tail, Pi_N^(2); N = 10^1..10^5; 4 alphas
│   ├── exp03_results_scaling_comparison.csv      # 19 alphas: naive, Paper I prediction, refined, doubling tail (A and B)
│   └── exp03_results_phase_diagram.csv           # 361 grid points: D_cov, D_comm, D_rot, gamma_tail, phase labels
├── figures/fig03_spectral_kakutani_criticality.{pdf,png}
├── paper/
│   ├── paper03_spectral_kakutani_criticality.tex # LaTeX source (amsart)
│   └── paper03_spectral_kakutani_criticality.pdf # compiled paper
├── results/
│   ├── paper03_tables_summary.txt                # text summary of every analytic constant and comparison
│   ├── paper03_tables.json, paper03_table4.csv   # machine-readable versions
│   ├── paper03_table6_crossover.csv              # Corollary 5.3: cross-over predictions on all 361 grid points
│   └── paper03_table{1..5}_rows.tex              # table bodies pasted into the paper
├── LICENSE, requirements.txt, .gitignore
└── README.md
```

## Reproducing

```bash
pip install -r requirements.txt
python code/exp03_spectral_kakutani_criticality.py     # prints OVERALL ACCEPTANCE: PASS
python code/paper03_compute_tables.py                  # analytic companion, writes results/
cd paper && pdflatex paper03_spectral_kakutani_criticality.tex && pdflatex paper03_spectral_kakutani_criticality.tex
```

Runs in about a minute (the phase diagram evaluates 361 exact block profiles of length 2 × 10^5). Python 3.12, NumPy 1.26, Matplotlib 3.8, mpmath 1.3.

## Citation

```bibtex
@article{ChenChen2026KakutaniIII,
  author  = {Chen, Zhengyi and Chen, Ruqing},
  title   = {Spectral Kakutani--Feldman--H{\'a}jek Criticality of Correlated Conformational Modes:
             Mode Amplitude Perturbation, Iso-Spectral Givens Rotation, and Tail-Increment Finite-Size Elimination},
  year    = {2026},
  doi     = {10.5281/zenodo.23015709},
  url     = {https://doi.org/10.5281/zenodo.23015709}
}
```

Repository: https://github.com/Ruqing1963/kakutani-statistical-pharmacology-III

## Funding

This project was supported by the National Natural Science Foundation of China (No. 22464010), the Guangxi Natural Science Foundation of China (No. 2025GXNSFAA069294), and the 2025 Bagui Youth Top Talent Project.

## License

Code under the MIT License; paper text, figures and data under CC BY 4.0 (see `LICENSE`).
