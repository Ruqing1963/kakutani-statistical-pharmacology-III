#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exp03_spectral_kakutani_criticality.py
======================================
Statistical Pharmacology via Kakutani Dichotomy, Paper III -- Experiment 03.
Spectral Kakutani-Feldman-Hajek criticality of covariance perturbations that decay
along the collective-mode index, and the doubling tail-increment estimator that
removes the finite-size bias of the scaling exponent near alpha = 1/2.

Setting (Paper II, Theorem 2.3)
-------------------------------
D_cov(N) = D_cov^comm(N) + D_cov^rot(N), where D_cov^comm = (1/2) sum_i ln cosh(Delta lambda_i / 2)
depends on the eigenvalue log-ratios only and D_cov^rot >= 0 vanishes iff the two
covariances share an ordered eigenbasis.

Mechanism A (commuting mode-amplitude perturbation)
    common eigenbasis, lambda_i^B = lambda_i^A (1 + c_lam i^(-alpha_lam)),  c_lam = 0.40
    exact per-mode contribution, x_i = c_lam i^(-alpha_lam):
        d_i = (1/2) ln[ (1 + x_i/2) / sqrt(1 + x_i) ]
            = x_i^2/16 - x_i^3/16 + 7 x_i^4/128 + O(x_i^5)
    leading coefficient c_lam^2/16 = 0.0100, the same prefactor A = c^2/(4p(1-p)) = 0.0100 as Paper I.
    (The brief quoted -c^3/32 and 5c^4/256 for the next two coefficients; the direct
    expansion gives -c^3/16 and 7c^4/128, which the script verifies numerically.)

Mechanism B (pure non-commuting mode rotation)
    Lambda_A = Lambda_B (all amplitudes unchanged, D_cov^comm = 0 exactly); the k-th
    adjacent pair of modes (2k-1, 2k), with anisotropy ratio r = lambda_{2k-1}/lambda_{2k} = 3,
    is rotated by the Givens angle theta_k = c_theta k^(-alpha_theta), c_theta = sqrt(0.06).
    exact per-pair contribution:
        d_k = (1/2) ln[ 1 + (r-1)^2 sin^2(theta_k) / (4r) ]
            = (beta/2) theta_k^2 - (beta/6 + beta^2/4) theta_k^4 + O(theta_k^6),  beta = (r-1)^2/(4r) = 1/3,
    leading coefficient (r-1)^2 c_theta^2 / (8r) = 0.06/6 = 0.0100.
    Here N counts rotated pairs (the ensemble has dimension 2N).

Feldman-Hajek dichotomy: the infinite Gaussian ensembles are equivalent iff D_cov(N) stays
bounded, i.e. iff alpha > alpha_c = 1/2, for each mechanism separately.

Doubling tail-increment estimator
    D_tail(N) = D_cov(2N) - D_cov(N) = sum_{N < i <= 2N} d_i
              = A (2^(1-2alpha) - 1)/(1-2alpha) N^(1-2alpha) (1 + O(N^-alpha))       [A = 0.01]
    cancels the Euler-Maclaurin constant A zeta(2alpha) whose pole at 2alpha = 1 biases the
    naive regression of ln D_cov on ln N (Paper I, Theorem 4.1 / Corollary 4.2);
    gamma_tail(alpha) := slope of ln D_tail on ln N  ->  1 - 2alpha on the whole range,
    gamma_tail^+ = max(gamma_tail, 0) = max(1 - 2alpha, 0) is the sharp phase curve.

Outputs (data/ and figures/ inside the repository layout, otherwise next to the script)
    exp03_results_benchmark.csv            mechanisms A and B, N = 10^1..10^5, alpha in {0.25, 0.5, 0.75, 1}
    exp03_results_scaling_comparison.csv   alpha = 0.05..0.95 (19 points): gamma_naive, gamma_pred (Paper I),
                                           refined gamma_pred, gamma_tail, gamma_tail^+ for both mechanisms
    exp03_results_phase_diagram.csv        19 x 19 grid (alpha_lam, alpha_theta): joint Feldman-Hajek phase diagram
    fig03_spectral_kakutani_criticality.{pdf,png}
"""

from __future__ import annotations

import csv
import os
import sys

import mpmath as mp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
C_LAM = 0.40                      # amplitude perturbation, c_lam^2/16 = 0.01
R_ANISO = 3.0                     # eigenvalue ratio inside a rotated pair
BETA = (R_ANISO - 1) ** 2 / (4 * R_ANISO)   # = 1/3
C_THETA = np.sqrt(0.06)           # rotation amplitude, beta c_theta^2 / 2 = 0.01
A_LEAD = 0.01                     # common leading coefficient of both mechanisms
N_MAX = 10 ** 5
ALPHAS_BENCH = (0.25, 0.50, 0.75, 1.00)
ALPHAS_SCAN = np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 2)
N_FIT_LO, N_FIT_HI = 10 ** 3, 10 ** 5
N_STAR = 10 ** 4                  # logarithmic centre of the regression window
ALPHA_C = 0.5
GRID = np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 2)   # phase-diagram axes

# acceptance tolerances
TOL_CRIT_SLOPE = 0.03             # relative, slope of D vs ln N at alpha = 1/2
TOL_TAIL_A = 0.02                 # |gamma_tail - (1 - 2alpha)| for mechanism A: the odd x^3 term of d_i gives a
                                  #   relative correction O(c N^-alpha) to D_tail, largest at small alpha
TOL_TAIL_A_PRED = 0.002           # |gamma_tail - refined Euler-Maclaurin prediction| for mechanism A
TOL_TAIL_B = 0.002                # |gamma_tail - (1 - 2alpha)| for mechanism B (even expansion in theta)
TOL_PHASE = 0.04                  # |gamma_tail^+ - max(1-2a_lam, 1-2a_theta, 0)| on the joint grid
MIN_NAIVE_BIAS_045 = 0.03         # the artefact must be visible at alpha = 0.45

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _repo_dir(name: str) -> str:
    d = os.path.join(ROOT, name)
    return d if os.path.isdir(d) else HERE


CSV_BENCH = os.path.join(_repo_dir("data"), "exp03_results_benchmark.csv")
CSV_SCAN = os.path.join(_repo_dir("data"), "exp03_results_scaling_comparison.csv")
CSV_PHASE = os.path.join(_repo_dir("data"), "exp03_results_phase_diagram.csv")
FIG_PATH = os.path.join(_repo_dir("figures"), "fig03_spectral_kakutani_criticality.png")
FIG_PATH_PDF = os.path.splitext(FIG_PATH)[0] + ".pdf"


# ----------------------------------------------------------------------------
# Per-mode contributions (exact) and their asymptotic expansions
# ----------------------------------------------------------------------------
def d_comm_exact(x: np.ndarray) -> np.ndarray:
    """(1/2) ln[(1 + x/2)/sqrt(1 + x)] = (1/2) ln cosh(Delta lambda / 2), Delta lambda = ln(1 + x)."""
    return 0.5 * (np.log1p(0.5 * x) - 0.5 * np.log1p(x))


def d_comm_series(x: np.ndarray, order: int = 4) -> np.ndarray:
    out = x ** 2 / 16
    if order >= 3:
        out = out - x ** 3 / 16
    if order >= 4:
        out = out + 7 * x ** 4 / 128
    return out


def d_rot_exact(theta: np.ndarray, beta: float = BETA) -> np.ndarray:
    """(1/2) ln[1 + beta sin^2 theta] for a rotated 2x2 block with beta = (r-1)^2/(4r)."""
    return 0.5 * np.log1p(beta * np.sin(theta) ** 2)


def d_rot_series(theta: np.ndarray, beta: float = BETA, order: int = 4) -> np.ndarray:
    out = 0.5 * beta * theta ** 2
    if order >= 4:
        out = out - (beta / 6 + beta ** 2 / 4) * theta ** 4
    return out


def profile(mechanism: str, alpha: float, n_max: int = 2 * N_MAX) -> np.ndarray:
    """Cumulative D_cov(N) for N = 1..n_max under mechanism 'A' (amplitude) or 'B' (rotation)."""
    i = np.arange(1, n_max + 1, dtype=np.float64)
    if mechanism == "A":
        return np.cumsum(d_comm_exact(C_LAM * i ** (-alpha)))
    if mechanism == "B":
        return np.cumsum(d_rot_exact(C_THETA * i ** (-alpha)))
    raise ValueError(mechanism)


def block_exact_cov(a: np.ndarray, b: np.ndarray, xa: np.ndarray, xb: np.ndarray, theta: np.ndarray):
    """Exact D_cov, D_comm, D_rot for 2x2 blocks: C_A = diag(a, b),
       C_B = R(theta) diag(a(1+xa), b(1+xb)) R(theta)^T, vectorised over blocks."""
    la, lb = a * (1 + xa), b * (1 + xb)
    c, s = np.cos(theta), np.sin(theta)
    # C_B entries
    b11 = la * c * c + lb * s * s
    b22 = la * s * s + lb * c * c
    b12 = (la - lb) * c * s
    # S = (C_A + C_B)/2
    s11, s22, s12 = 0.5 * (a + b11), 0.5 * (b + b22), 0.5 * b12
    logdet_s = np.log(s11 * s22 - s12 * s12)
    logdet_a = np.log(a * b)
    logdet_b = np.log(la * lb)
    d_cov = 0.5 * logdet_s - 0.25 * (logdet_a + logdet_b)
    # commuting part from the ordered spectra (a >= b and la >= lb by construction, r = 3)
    d_comm = d_comm_exact(xa) + d_comm_exact(xb)
    return d_cov, d_comm, d_cov - d_comm


# ----------------------------------------------------------------------------
# Regression helpers
# ----------------------------------------------------------------------------
def _grid(n_lo: int, n_hi: int, n_pts: int = 60) -> np.ndarray:
    return np.unique(np.logspace(np.log10(n_lo), np.log10(n_hi), n_pts).astype(int))


def gamma_naive(D: np.ndarray) -> float:
    ns = _grid(N_FIT_LO, N_FIT_HI)
    return float(np.polyfit(np.log(ns.astype(float)), np.log(D[ns - 1]), 1)[0])


def gamma_tail(D: np.ndarray) -> float:
    ns = _grid(N_FIT_LO, N_FIT_HI)
    dt = D[2 * ns - 1] - D[ns - 1]
    return float(np.polyfit(np.log(ns.astype(float)), np.log(dt), 1)[0])


def critical_slope(D: np.ndarray) -> float:
    ns = _grid(N_FIT_LO, N_FIT_HI)
    return float(np.polyfit(np.log(ns.astype(float)), D[ns - 1], 1)[0])


def gamma_pred_paper1(alpha: float, n_star: int = N_STAR) -> float:
    """Paper I, Theorem 4.1: local exponent of A[N^(1-2a)/(1-2a) + zeta(2a)] at N = n_star."""
    if abs(alpha - 0.5) < 1e-12:
        return float(1.0 / (np.log(n_star) + float(mp.euler)))
    z = float(mp.zeta(2 * alpha))
    e = 1 - 2 * alpha
    return e / (1 + e * z * n_star ** (-e))


def _expansion_terms(mechanism: str):
    if mechanism == "A":
        return [(2, C_LAM ** 2 / 16), (3, -C_LAM ** 3 / 16), (4, 7 * C_LAM ** 4 / 128)]
    return [(2, 0.5 * BETA * C_THETA ** 2), (4, -(BETA / 6 + BETA ** 2 / 4) * C_THETA ** 4)]


def gamma_pred_refined(mechanism: str, alpha: float, n_star: int = N_STAR) -> float:
    """Local exponent at n_star of the Euler-Maclaurin form with the first three (A) or two (B)
       terms of the per-mode expansion: sum_k a_k c^k [N^(1-k a)/(1-k a) + zeta(k a)]."""
    num, den = 0.0, 0.0
    for k, ak in _expansion_terms(mechanism):
        s = k * alpha
        if abs(s - 1) < 1e-12:
            den += ak * (np.log(n_star) + float(mp.euler))
            num += ak
        else:
            den += ak * (n_star ** (1 - s) / (1 - s) + float(mp.zeta(s)))
            num += ak * n_star ** (1 - s)
    return float(num / den)


def gamma_tail_pred(mechanism: str, alpha: float) -> float:
    """Least-squares exponent, on the regression grid, of the analytic doubling tail
       sum_k a_k c^k [(2N)^(1-k a) - N^(1-k a)]/(1-k a): the zeta constants cancel exactly, so the
       only deviation from 1 - 2 alpha is the sub-leading power N^(1-3a) (A) or N^(1-4a) (B)."""
    ns = _grid(N_FIT_LO, N_FIT_HI).astype(float)
    dt = np.zeros_like(ns)
    for k, ak in _expansion_terms(mechanism):
        s = k * alpha
        if abs(s - 1) < 1e-12:
            dt += ak * np.log(2.0)
        else:
            dt += ak * ((2 * ns) ** (1 - s) - ns ** (1 - s)) / (1 - s)
    return float(np.polyfit(np.log(ns), np.log(dt), 1)[0])


# ----------------------------------------------------------------------------
# Figure style
# ----------------------------------------------------------------------------
SURFACE, INK, INK_2, MUTED, GRID_C, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {0.25: "#2a78d6", 0.50: "#eb6834", 0.75: "#1baf7a", 1.00: "#eda100"}
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 9, "axes.unicode_minus": False,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlelocation": "left", "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED, "grid.color": GRID_C, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "legend.labelcolor": INK_2, "mathtext.fontset": "dejavusans",
})


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.6)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    ok_all = True
    print("=" * 78)
    print("Kakutani statistical pharmacology - Experiment 03 - spectral criticality and doubling tail")
    print(f"mechanism A: c_lam = {C_LAM} (c^2/16 = {C_LAM ** 2 / 16:.4f});  "
          f"mechanism B: r = {R_ANISO}, c_theta = {C_THETA:.7f} (beta c^2/2 = {0.5 * BETA * C_THETA ** 2:.4f})")
    print("=" * 78)

    # ---------- 0. Expansion check ----------
    x = np.array([0.4, 0.2, 0.1, 0.05])
    err3 = np.abs(d_comm_exact(x) - d_comm_series(x, 3))
    err4 = np.abs(d_comm_exact(x) - d_comm_series(x, 4))
    print("\n[0] Per-mode expansions")
    print(f"  mechanism A at x = {x.tolist()}: |exact - series(3)| = {err3}")
    print(f"                                  |exact - series(4)| = {err4}  (ratio to x^5: {err4 / x ** 5})")
    t = np.array([0.24, 0.12, 0.06])
    errt = np.abs(d_rot_exact(t) - d_rot_series(t))
    print(f"  mechanism B at theta = {t.tolist()}: |exact - series(4)| = {errt}  (ratio to theta^6: {errt / t ** 6})")
    ok_all &= bool(np.all(err4 < 0.5 * x ** 5)) and bool(np.all(errt < 0.2 * t ** 6))

    # ---------- 1. Benchmarks ----------
    print("\n[1] Benchmarks: D_cov^comm, D_cov^rot, D_tail = D(2N) - D(N), Pi_N^(2) = exp(-D_cov)")
    decades = [10 ** k for k in range(1, 6)]
    bench_rows = []
    profiles = {}
    for mech in ("A", "B"):
        for a in ALPHAS_BENCH:
            D = profile(mech, a)
            profiles[(mech, a)] = D
            for n in decades:
                d_n = float(D[n - 1])
                d_tail = float(D[2 * n - 1] - D[n - 1])
                bench_rows.append({"mechanism": mech, "alpha": a, "N": n,
                                   "D_cov_comm": d_n if mech == "A" else 0.0,
                                   "D_cov_rot": d_n if mech == "B" else 0.0,
                                   "D_cov": d_n, "D_tail": d_tail, "Pi2": float(np.exp(-d_n))})
    for mech in ("A", "B"):
        label = "A (amplitude, D_cov = D_cov^comm)" if mech == "A" else "B (rotation, D_cov = D_cov^rot)"
        print(f"\n  mechanism {label}")
        print(f"  {'alpha':>6} {'N':>7} {'D_cov':>12} {'D_tail':>12} {'Pi2':>10}")
        for r in bench_rows:
            if r["mechanism"] == mech:
                print(f"  {r['alpha']:>6.2f} {r['N']:>7} {r['D_cov']:>12.6g} {r['D_tail']:>12.6g} {r['Pi2']:>10.6f}")
    with open(CSV_BENCH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys())); w.writeheader(); w.writerows(bench_rows)

    # critical slopes and Feldman-Hajek check
    print("\n  Acceptance: slope of D_cov vs ln N at alpha = 1/2 (theory A = 0.0100)")
    for mech in ("A", "B"):
        sl = critical_slope(profiles[(mech, 0.5)])
        rel = abs(sl - A_LEAD) / A_LEAD
        passed = rel <= TOL_CRIT_SLOPE
        ok_all &= passed
        print(f"    mechanism {mech}: slope = {sl:.5f}, rel.err {rel:.2%}  {'PASS' if passed else 'FAIL'}")
    print("  Feldman-Hajek check: D_cov(1e5) - D_cov(1e4) (divergent for alpha <= 1/2, saturating for alpha > 1/2)")
    for mech in ("A", "B"):
        for a in ALPHAS_BENCH:
            D = profiles[(mech, a)]
            inc = float(D[10 ** 5 - 1] - D[10 ** 4 - 1])
            tail_lim = float(D[2 * 10 ** 5 - 1] - D[10 ** 5 - 1])
            print(f"    {mech} alpha={a:.2f}: increment {inc:.3e}, D_tail(1e5) = {tail_lim:.3e}"
                  f"  -> {'singular (D_tail does not vanish)' if a <= 0.5 else 'equivalent (D_tail -> 0)'}")

    # ---------- 2. Scaling comparison ----------
    print("\n[2] Scaling exponents over alpha in [0.05, 0.95]: naive vs Paper I prediction vs doubling tail")
    scan_rows = []
    print(f"  {'alpha':>6} {'1-2a':>7} | {'naive_A':>8} {'pred_I':>8} {'pred_ref':>8} {'tail_A':>8} {'tailpred':>8} | "
          f"{'naive_B':>8} {'tail_B':>8} | {'tail+_A':>8}")
    for a in ALPHAS_SCAN:
        a = float(a)
        DA, DB = profile("A", a), profile("B", a)
        gnA, gtA = gamma_naive(DA), gamma_tail(DA)
        gnB, gtB = gamma_naive(DB), gamma_tail(DB)
        gp = gamma_pred_paper1(a)
        gprA, gprB = gamma_pred_refined("A", a), gamma_pred_refined("B", a)
        gtpA, gtpB = gamma_tail_pred("A", a), gamma_tail_pred("B", a)
        row = {"alpha": a, "gamma_th": 1 - 2 * a, "gamma_naive_A": gnA, "gamma_pred_paper1": gp,
               "gamma_pred_refined_A": gprA, "gamma_tail_A": gtA, "gamma_tail_pred_A": gtpA,
               "gamma_tail_plus_A": max(gtA, 0.0),
               "gamma_naive_B": gnB, "gamma_pred_refined_B": gprB, "gamma_tail_B": gtB, "gamma_tail_pred_B": gtpB,
               "gamma_tail_plus_B": max(gtB, 0.0),
               "bias_naive_A": gnA - max(1 - 2 * a, 0.0), "bias_tail_A": gtA - (1 - 2 * a),
               "resid_tail_A_vs_pred": gtA - gtpA,
               "bias_naive_B": gnB - max(1 - 2 * a, 0.0), "bias_tail_B": gtB - (1 - 2 * a),
               "resid_tail_B_vs_pred": gtB - gtpB}
        scan_rows.append(row)
        print(f"  {a:>6.2f} {1 - 2 * a:>+7.2f} | {gnA:>8.4f} {gp:>8.4f} {gprA:>8.4f} {gtA:>+8.4f} {gtpA:>+8.4f} | "
              f"{gnB:>8.4f} {gtB:>+8.4f} | {max(gtA, 0):>8.4f}")
    with open(CSV_SCAN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys())); w.writeheader(); w.writerows(scan_rows)

    r045 = next(r for r in scan_rows if abs(r["alpha"] - 0.45) < 1e-9)
    maxA = max(abs(r["bias_tail_A"]) for r in scan_rows)
    maxB = max(abs(r["bias_tail_B"]) for r in scan_rows)
    maxAp = max(abs(r["resid_tail_A_vs_pred"]) for r in scan_rows)
    maxBp = max(abs(r["resid_tail_B_vs_pred"]) for r in scan_rows)
    passed = maxA <= TOL_TAIL_A and maxB <= TOL_TAIL_B and maxAp <= TOL_TAIL_A_PRED
    ok_all &= passed
    print(f"\n  Acceptance: max |gamma_tail - (1-2alpha)| over the scan: A {maxA:.4f} (tol {TOL_TAIL_A}; the residual is the"
          f" N^(1-3alpha) term), B {maxB:.4f} (tol {TOL_TAIL_B})")
    print(f"              max |gamma_tail - analytic doubling-tail prediction|: A {maxAp:.4f} (tol {TOL_TAIL_A_PRED}), "
          f"B {maxBp:.4f}  {'PASS' if passed else 'FAIL'}")
    print(f"  Acceptance: naive bias at alpha = 0.45: A {r045['bias_naive_A']:+.4f}, B {r045['bias_naive_B']:+.4f}"
          f" (must exceed {MIN_NAIVE_BIAS_045}); doubling-tail bias: A {r045['bias_tail_A']:+.4f}, "
          f"B {r045['bias_tail_B']:+.4f}  {'PASS' if r045['bias_naive_A'] > MIN_NAIVE_BIAS_045 else 'FAIL'}")
    ok_all &= r045["bias_naive_A"] > MIN_NAIVE_BIAS_045
    print(f"  bias reduction factor at alpha = 0.45: A x{r045['bias_naive_A'] / max(abs(r045['bias_tail_A']), 1e-12):.0f}, "
          f"B x{r045['bias_naive_B'] / max(abs(r045['bias_tail_B']), 1e-12):.0f}")

    # ---------- 3. Joint phase diagram ----------
    print("\n[3] Joint (alpha_lam, alpha_theta) phase diagram on a 19 x 19 grid")
    ns = _grid(N_FIT_LO, N_FIT_HI)
    k = np.arange(1, 2 * N_MAX + 1, dtype=np.float64)
    a_vec = np.full_like(k, R_ANISO)
    b_vec = np.ones_like(k)
    phase_rows = []
    gamma_grid = np.zeros((GRID.size, GRID.size))
    mismatches = 0
    for ia, al in enumerate(GRID):
        xa = C_LAM * (2 * k - 1) ** (-float(al))    # amplitude perturbation of mode 2k-1
        xb = C_LAM * (2 * k) ** (-float(al))        # and of mode 2k
        for it, at in enumerate(GRID):
            theta = C_THETA * k ** (-float(at))
            d_cov, d_comm, d_rot = block_exact_cov(a_vec, b_vec, xa, xb, theta)
            D = np.cumsum(d_cov)
            dt = D[2 * ns - 1] - D[ns - 1]
            g_tail = float(np.polyfit(np.log(ns.astype(float)), np.log(dt), 1)[0])
            g_plus = max(g_tail, 0.0)
            singular_num = g_tail > -0.05          # D_tail does not decay -> D_cov diverges
            singular_th = min(al, at) <= ALPHA_C + 1e-9
            mismatches += int(singular_num != singular_th)
            gamma_grid[it, ia] = g_plus
            Dc, Dm, Dr = float(np.cumsum(d_comm)[N_MAX - 1]), float(D[N_MAX - 1]), float(np.cumsum(d_rot)[N_MAX - 1])
            phase_rows.append({"alpha_lam": float(al), "alpha_theta": float(at), "D_cov_1e5": Dm,
                               "D_comm_1e5": Dc, "D_rot_1e5": Dr, "gamma_tail": g_tail, "gamma_tail_plus": g_plus,
                               "gamma_theory": max(1 - 2 * float(al), 1 - 2 * float(at), 0.0),
                               "phase_numeric": "singular" if singular_num else "equivalent",
                               "phase_theory": "singular" if singular_th else "equivalent"})
    with open(CSV_PHASE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(phase_rows[0].keys())); w.writeheader(); w.writerows(phase_rows)
    max_dev = max(abs(r["gamma_tail_plus"] - r["gamma_theory"]) for r in phase_rows)
    print(f"  grid points: {len(phase_rows)}; phase classification mismatches: {mismatches}; "
          f"max |gamma_tail^+ - max(1-2a_lam, 1-2a_theta, 0)| = {max_dev:.4f}")
    passed = mismatches == 0 and max_dev < TOL_PHASE
    ok_all &= passed
    print(f"  Acceptance: joint phase diagram (0 mismatches, max deviation < {TOL_PHASE})  {'PASS' if passed else 'FAIL'}")

    # ---------- 4. Figure ----------
    print("\n[4] Plotting ->", FIG_PATH)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), dpi=200)
    (axA, axB), (axC, axD) = axes
    ns_plot = np.unique(np.logspace(1, 5, 300).astype(int))
    i_all = np.arange(1, N_MAX + 1, dtype=np.float64)

    # (A) mechanism A: exact vs asymptotic expansion
    for a in ALPHAS_BENCH:
        D = profiles[("A", a)]
        axA.plot(ns_plot, D[ns_plot - 1], color=SERIES[a], lw=1.6, label=rf"$\alpha_\lambda={a:.2f}$")
        Dser = np.cumsum(d_comm_series(C_LAM * i_all ** (-a), 4))
        axA.plot(ns_plot, Dser[ns_plot - 1], color=INK_2, lw=0.9, ls="--")
        axA.annotate(rf"$\alpha_\lambda={a:.2f}$", xy=(ns_plot[-1], D[ns_plot[-1] - 1]), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=8, color=INK_2)
    axA.plot([], [], color=INK_2, lw=0.9, ls="--", label="4-term expansion")
    axA.set_xscale("log"); axA.set_yscale("log"); axA.set_xlim(10, 10 ** 5 * 2.6)
    axA.set_xlabel(r"$N$ (modes)"); axA.set_ylabel(r"$D_{\mathrm{cov}}^{\mathrm{comm}}(N)$")
    axA.set_title(r"A  Mechanism A: amplitude perturbation $\lambda_i^B=\lambda_i^A(1+0.4\,i^{-\alpha_\lambda})$")
    axA.legend(loc="upper left")
    tidy(axA)

    # (B) mechanism B
    for a in ALPHAS_BENCH:
        D = profiles[("B", a)]
        axB.plot(ns_plot, D[ns_plot - 1], color=SERIES[a], lw=1.6, label=rf"$\alpha_\theta={a:.2f}$")
        Dser = np.cumsum(d_rot_series(C_THETA * i_all ** (-a), order=4))
        axB.plot(ns_plot, Dser[ns_plot - 1], color=INK_2, lw=0.9, ls="--")
        axB.annotate(rf"$\alpha_\theta={a:.2f}$", xy=(ns_plot[-1], D[ns_plot[-1] - 1]), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=8, color=INK_2)
    axB.plot([], [], color=INK_2, lw=0.9, ls="--", label="2-term expansion")
    axB.set_xscale("log"); axB.set_yscale("log"); axB.set_xlim(10, 10 ** 5 * 2.6)
    axB.set_xlabel(r"$N$ (rotated mode pairs)"); axB.set_ylabel(r"$D_{\mathrm{cov}}^{\mathrm{rot}}(N)$")
    axB.set_title(r"B  Mechanism B: pure rotation $\theta_k=\sqrt{0.06}\,k^{-\alpha_\theta}$, $r=3$, $\Lambda_A=\Lambda_B$")
    axB.legend(loc="upper left")
    tidy(axB)
    for ax in (axA, axB):
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6)); ax.xaxis.set_minor_formatter(NullFormatter())

    # (C) naive vs tail exponents
    al = np.array([r["alpha"] for r in scan_rows])
    th = np.maximum(1 - 2 * al, 0)
    axC.plot(al, th, color=INK_2, lw=1.0, ls="--", label=r"theory $\max(1-2\alpha,0)$")
    axC.plot(al, [r["gamma_naive_A"] for r in scan_rows], color=ORANGE, lw=1.4, marker="o", ms=4.5, mec=SURFACE,
             label=r"naive $\gamma$ (regression of $\ln D_{\mathrm{cov}}$), A")
    axC.plot(al, [r["gamma_naive_B"] for r in scan_rows], color=ORANGE, lw=0, marker="s", ms=4.5, mfc="none",
             label=r"naive $\gamma$, B")
    axC.plot(al, [r["gamma_tail_plus_A"] for r in scan_rows], color=BLUE, lw=1.6, marker="o", ms=4.5, mec=SURFACE,
             label=r"doubling tail $\gamma_{\mathrm{tail}}^+$, A")
    axC.plot(al, [r["gamma_tail_plus_B"] for r in scan_rows], color=BLUE, lw=0, marker="s", ms=4.5, mfc="none",
             label=r"doubling tail $\gamma_{\mathrm{tail}}^+$, B")
    axC.axvline(ALPHA_C, color=AXIS, lw=0.8)
    axC.text(ALPHA_C + 0.01, 0.62, r"$\alpha_c=1/2$", color=INK_2, fontsize=8, va="top")
    axC.annotate(rf"bias at $\alpha=0.45$: naive {r045['bias_naive_A']:+.3f}, tail {r045['bias_tail_A']:+.4f}",
                 xy=(0.45, r045["gamma_naive_A"]), xytext=(0.52, 0.42), textcoords="data", fontsize=8, color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    axC.set_xlim(0, 1.0); axC.set_ylim(-0.05, 1.0)
    axC.set_xlabel(r"decay exponent $\alpha$ ($\alpha_\lambda$ for A, $\alpha_\theta$ for B)")
    axC.set_ylabel(r"scaling exponent $\gamma$")
    axC.set_title("C  Naive exponent artefact vs doubling-tail phase curve")
    axC.legend(loc="upper right")
    tidy(axC)

    # (D) joint phase diagram
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
    extent = (GRID[0] - 0.025, GRID[-1] + 0.025, GRID[0] - 0.025, GRID[-1] + 0.025)
    axD.grid(False)
    im = axD.imshow(gamma_grid, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=0.9, aspect="auto")
    axD.axvline(ALPHA_C, color=SURFACE, lw=2.2); axD.axhline(ALPHA_C, color=SURFACE, lw=2.2)
    axD.axvline(ALPHA_C, color=INK, lw=0.9, ls="--"); axD.axhline(ALPHA_C, color=INK, lw=0.9, ls="--")
    axD.text(0.27, 0.27, r"$P_A\perp P_B$" "\n" r"(both diverge)", ha="center", va="center", fontsize=8, color=SURFACE)
    axD.text(0.73, 0.27, r"$P_A\perp P_B$" "\n" r"(rotation)", ha="center", va="center", fontsize=8, color=SURFACE)
    axD.text(0.27, 0.73, r"$P_A\perp P_B$" "\n" r"(amplitude)", ha="center", va="center", fontsize=8, color=SURFACE)
    axD.text(0.73, 0.73, r"$P_A\sim P_B$" "\n" r"(equivalent)", ha="center", va="center", fontsize=8, color=INK)
    cb = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.02)
    cb.set_label(r"$\gamma_{\mathrm{tail}}^+=\max(1-2\alpha_\lambda,\,1-2\alpha_\theta,\,0)$", color=INK_2)
    cb.ax.tick_params(colors=MUTED)
    axD.set_xlabel(r"amplitude exponent $\alpha_\lambda$"); axD.set_ylabel(r"rotation exponent $\alpha_\theta$")
    axD.set_title(rf"D  Joint Feldman-Hajek phase diagram ({len(phase_rows)} grid points)")
    tidy(axD)

    fig.tight_layout()
    fig.savefig(FIG_PATH); fig.savefig(FIG_PATH_PDF)
    plt.close(fig)

    print("\nOutput files:")
    for pth in (CSV_BENCH, CSV_SCAN, CSV_PHASE, FIG_PATH, FIG_PATH_PDF):
        print("  ", pth)
    print("\nOVERALL ACCEPTANCE:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
