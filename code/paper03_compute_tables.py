#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paper03_compute_tables.py
=========================
Analytic companion of Paper III. Reads the three CSV files written by
exp03_spectral_kakutani_criticality.py, adds the geodesic variant of mechanism A
(lambda_i^B = lambda_i^A exp(c i^-alpha)), evaluates every Euler-Maclaurin constant,
limit and finite-size correction quoted in the paper, and writes

  results/paper03_tables_summary.txt   human-readable summary of all numbers
  results/paper03_tables.json          the same, machine-readable
  results/paper03_table{1..5}_rows.tex table bodies pasted verbatim into the paper
  results/paper03_table4.csv           three-way comparison (Euclidean A, geodesic A, rotation B)

Coefficient closed forms
  Euclidean A : d(delta) = (1/2) ln[(1+delta/2)/sqrt(1+delta)] = sum_{k>=2} a_k delta^k,
                a_k = (-1)^k (1 - 2^{1-k}) / (4k)  (a_2 = 1/16, a_3 = -1/16, a_4 = 7/128, a_5 = -3/64, ...)
  geodesic A  : d(eta) = (1/2) ln cosh(eta/2) = eta^2/16 - eta^4/384 + eta^6/5760 - ...  (even)
  rotation B  : d(theta) = (1/2) ln[1 + kappa sin^2 theta], kappa = (r-1)^2/(4r) (even in theta)
"""
from __future__ import annotations

import csv
import json
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from exp03_spectral_kakutani_criticality import (  # noqa: E402
    C_LAM, C_THETA, R_ANISO, BETA, N_FIT_LO, N_FIT_HI, N_STAR, ALPHAS_BENCH, ALPHAS_SCAN, GRID,
    d_comm_exact, d_rot_exact, profile, _grid, gamma_naive, gamma_tail, gamma_pred_paper1, gamma_pred_refined,
)

mp.mp.dps = 30


def _repo_dir(name: str) -> str:
    d = os.path.join(ROOT, name)
    return d if os.path.isdir(d) else HERE


DATA_DIR, RES_DIR = _repo_dir("data"), _repo_dir("results")
KMAX = 40          # terms of the per-mode series used in the Euler-Maclaurin predictions


# ---------------------------------------------------------------------------
# Per-mode series coefficients
# ---------------------------------------------------------------------------
def coeffs_euclid(kmax: int = KMAX):
    """a_k for k = 2..kmax of d(delta) = (1/2) ln[(1+delta/2)/sqrt(1+delta)]."""
    return {k: (-1) ** k * (1 - mp.mpf(2) ** (1 - k)) / (4 * k) for k in range(2, kmax + 1)}


def coeffs_geo(kmax: int = 24):
    t = mp.taylor(lambda u: mp.log(mp.cosh(u / 2)) / 2, 0, kmax)
    return {k: t[k] for k in range(2, kmax + 1) if abs(t[k]) > mp.mpf(10) ** -25}


def coeffs_rot(kmax: int = 24):
    kap = mp.mpf(BETA)
    t = mp.taylor(lambda u: mp.log(1 + kap * mp.sin(u) ** 2) / 2, 0, kmax)
    return {k: t[k] for k in range(2, kmax + 1) if abs(t[k]) > mp.mpf(10) ** -25}


def d_geo_exact(eta: np.ndarray) -> np.ndarray:
    """(1/2) ln cosh(eta/2), evaluated with the even Taylor series for |eta/2| < 0.05 where the
    direct formula loses precision (cosh(x) rounds to 1 + x^2/2 with relative error ~ eps/x^2)."""
    x = 0.5 * np.asarray(eta, dtype=np.float64)
    x2 = x * x
    small = x2 / 2 - x2 ** 2 / 12 + x2 ** 3 / 45 - 17 * x2 ** 4 / 2520
    large = np.log(np.cosh(np.where(np.abs(x) < 0.05, 0.05, x)))
    return 0.5 * np.where(np.abs(x) < 0.05, small, large)


def profile_geo(alpha: float, n_max: int = 2 * 10 ** 5) -> np.ndarray:
    i = np.arange(1, n_max + 1, dtype=np.float64)
    return np.cumsum(d_geo_exact(C_LAM * i ** (-alpha)))


# ---------------------------------------------------------------------------
# Euler-Maclaurin machinery
# ---------------------------------------------------------------------------
def H_em(N, s):
    """Euler-Maclaurin form of H_N(s) = sum_{i<=N} i^-s (valid for s > 0, analytic continuation of zeta)."""
    N = mp.mpf(N); s = mp.mpf(s)
    if abs(s - 1) < mp.mpf(10) ** -12:
        return mp.log(N) + mp.euler + 1 / (2 * N) - 1 / (12 * N ** 2)
    return N ** (1 - s) / (1 - s) + mp.zeta(s) + N ** (-s) / 2 - s * N ** (-s - 1) / 12


def D_pred(mech: str, alpha: float, N) -> mp.mpf:
    """Multi-term Euler-Maclaurin prediction of D(N) for mechanism 'A' (Euclidean), 'G' (geodesic), 'B'."""
    if mech == "A":
        co, c = coeffs_euclid(), mp.mpf(C_LAM)
    elif mech == "G":
        co, c = coeffs_geo(), mp.mpf(C_LAM)
    else:
        co, c = coeffs_rot(), mp.mpf(C_THETA)
    return mp.fsum(ak * c ** k * H_em(N, k * alpha) for k, ak in co.items())


def D_inf(mech: str, alpha: float) -> mp.mpf:
    """Limit sum_{k} a_k c^k zeta(k alpha) for alpha > 1/2 (all k alpha > 1)."""
    if mech == "A":
        co, c = coeffs_euclid(), mp.mpf(C_LAM)
    elif mech == "G":
        co, c = coeffs_geo(), mp.mpf(C_LAM)
    else:
        co, c = coeffs_rot(), mp.mpf(C_THETA)
    return mp.fsum(ak * c ** k * mp.zeta(k * alpha) for k, ak in co.items())


def D_tail_pred(mech: str, alpha: float, ns: np.ndarray) -> np.ndarray:
    """Analytic doubling tail sum_k a_k c^k [(2N)^(1-ka) - N^(1-ka)]/(1-ka) (zeta constants cancel)."""
    if mech == "A":
        co, c = coeffs_euclid(), mp.mpf(C_LAM)
    elif mech == "G":
        co, c = coeffs_geo(), mp.mpf(C_LAM)
    else:
        co, c = coeffs_rot(), mp.mpf(C_THETA)
    out = []
    for N in ns:
        N = mp.mpf(int(N))
        tot = mp.mpf(0)
        for k, ak in co.items():
            s = k * alpha
            if abs(s - 1) < 1e-12:
                tot += ak * c ** k * mp.log(2)
            else:
                tot += ak * c ** k * ((2 * N) ** (1 - s) - N ** (1 - s)) / (1 - s)
        out.append(float(tot))
    return np.array(out)


def gamma_of(D_vals: np.ndarray, ns: np.ndarray) -> float:
    return float(np.polyfit(np.log(ns.astype(float)), np.log(D_vals), 1)[0])


def delta_gamma_formula(alpha: float, c: float, N: float) -> float:
    """Leading finite-size shift of the Euclidean doubling-tail exponent, Theorem 4.3:
       alpha c (1-2a)/(1-3a) (2^(1-3a)-1)/(2^(1-2a)-1) N^-a."""
    K2 = (2 ** (1 - 2 * alpha) - 1) / (1 - 2 * alpha) if abs(1 - 2 * alpha) > 1e-12 else np.log(2)
    K3 = (2 ** (1 - 3 * alpha) - 1) / (1 - 3 * alpha) if abs(1 - 3 * alpha) > 1e-12 else np.log(2)
    return alpha * c * (K3 / K2) * N ** (-alpha)


# ---------------------------------------------------------------------------
def main():
    out = {"c_lam": C_LAM, "c_theta": float(C_THETA), "r": R_ANISO, "kappa": BETA}
    lines = []
    P = lines.append
    P("Paper III - analytic companion summary (paper03_compute_tables.py)")
    P("=" * 78)

    # ---- coefficients ----
    ca, cg, cr = coeffs_euclid(8), coeffs_geo(12), coeffs_rot(12)
    out["coeff_euclid"] = {k: float(v) for k, v in ca.items()}
    out["coeff_geo"] = {k: float(v) for k, v in cg.items()}
    out["coeff_rot"] = {k: float(v) for k, v in cr.items()}
    P("\n[coefficients] Euclidean A: " + ", ".join(f"a_{k} = {mp.nstr(v, 10)} ({mp.identify(v)})" for k, v in ca.items()))
    P("[coefficients] geodesic A : " + ", ".join(f"g_{k} = {mp.nstr(v, 10)} ({mp.identify(v)})" for k, v in cg.items()))
    P("[coefficients] rotation B : " + ", ".join(f"b_{k} = {mp.nstr(v, 10)}" for k, v in cr.items()))
    P(f"   check: kappa/2 = {BETA / 2:.10f}, -kappa(2+3kappa)/12 = {-BETA * (2 + 3 * BETA) / 12:.10f}")
    # numerical check of the pull-back identity at delta = 0.4
    d = mp.mpf("0.4")
    eta = mp.log(1 + d)
    lhs = mp.log(mp.cosh(eta / 2)) / 2
    rhs = mp.log((1 + d / 2) / mp.sqrt(1 + d)) / 2
    P(f"   pull-back identity at delta = 0.4: (1/2)ln cosh(eta/2) = {mp.nstr(lhs, 15)}, "
      f"(1/2)ln[(1+d/2)/sqrt(1+d)] = {mp.nstr(rhs, 15)}, diff = {mp.nstr(lhs - rhs, 3)}")

    # ---- data ----
    bench = list(csv.DictReader(open(os.path.join(DATA_DIR, "exp03_results_benchmark.csv"), encoding="utf-8")))
    scan = list(csv.DictReader(open(os.path.join(DATA_DIR, "exp03_results_scaling_comparison.csv"), encoding="utf-8")))
    phase = list(csv.DictReader(open(os.path.join(DATA_DIR, "exp03_results_phase_diagram.csv"), encoding="utf-8")))

    # ---- Tables 1 and 2: benchmarks vs multi-term Euler-Maclaurin predictions ----
    P("\n[Tables 1-2] benchmark vs Euler-Maclaurin prediction (K = 40 series terms)")
    t12 = {"A": [], "B": []}
    for r in bench:
        mech, a, N = r["mechanism"], float(r["alpha"]), int(r["N"])
        Dm = float(r["D_cov"]); pred = float(D_pred(mech, a, N))
        dinf = float(D_inf(mech, a)) if a > 0.5 else float("nan")
        t12[mech].append({"alpha": a, "N": N, "D_cov": Dm, "D_pred": pred, "diff": Dm - pred,
                          "D_tail": float(r["D_tail"]), "Pi2": float(r["Pi2"]), "D_inf": dinf})
        P(f"   {mech} alpha={a:.2f} N={N:>6}: D={Dm:.8f} pred={pred:.8f} diff={Dm - pred:+.1e} D_tail={float(r['D_tail']):.6g}"
          + (f" D_inf={dinf:.8f}" if a > 0.5 else ""))
    out["table12"] = t12
    # limits and critical constants
    lim = {}
    for mech in ("A", "G", "B"):
        for a in (0.75, 1.0):
            lim[f"{mech}_{a}"] = float(D_inf(mech, a))
    out["D_inf"] = lim
    P("\n[limits] D_inf(alpha) = sum_k a_k c^k zeta(k alpha):")
    for k, v in lim.items():
        P(f"   {k}: {v:.10f}")
    # critical constants: D(N) = A ln N + C_crit + o(1) at alpha = 1/2, C_crit = A gamma_EM + sum_{k>=3} a_k c^k zeta(k/2)
    for mech in ("A", "G", "B"):
        co = coeffs_euclid() if mech == "A" else (coeffs_geo() if mech == "G" else coeffs_rot())
        c = mp.mpf(C_LAM) if mech != "B" else mp.mpf(C_THETA)
        Acoef = co[2] * c ** 2
        Ccrit = Acoef * mp.euler + mp.fsum(ak * c ** k * mp.zeta(k * mp.mpf("0.5")) for k, ak in co.items() if k >= 3)
        out[f"critical_{mech}"] = {"A": float(Acoef), "C_crit": float(Ccrit)}
        P(f"[critical] {mech}: A = {float(Acoef):.10f}, C_crit = A*gamma_EM + sum_(k>=3) a_k c^k zeta(k/2) = {float(Ccrit):.10f}")
    # measured critical slopes
    for mech in ("A", "B"):
        D = profile(mech, 0.5)
        ns = _grid(N_FIT_LO, N_FIT_HI)
        sl, ic = np.polyfit(np.log(ns.astype(float)), D[ns - 1], 1)
        out[f"critical_{mech}"]["slope_measured"] = float(sl); out[f"critical_{mech}"]["intercept_measured"] = float(ic)
        P(f"[critical] {mech}: measured slope {sl:.6f}, intercept {ic:.6f}")
    Dg = profile_geo(0.5); ns = _grid(N_FIT_LO, N_FIT_HI)
    sl, ic = np.polyfit(np.log(ns.astype(float)), Dg[ns - 1], 1)
    out["critical_G"]["slope_measured"] = float(sl); out["critical_G"]["intercept_measured"] = float(ic)
    P(f"[critical] G: measured slope {sl:.6f}, intercept {ic:.6f}")

    # ---- Table 3: scaling comparison (from CSV) + window-regression refined predictions ----
    P("\n[Table 3] scaling exponents: naive vs predictions vs doubling tail")
    t3 = []
    for r in scan:
        a = float(r["alpha"])
        # refined naive prediction regressed over the window (not only at N*)
        ns = _grid(N_FIT_LO, N_FIT_HI)
        DpA = np.array([float(D_pred("A", a, int(n))) for n in ns])
        g_ref_fit = gamma_of(DpA, ns)
        t3.append({"alpha": a, "gamma_th": float(r["gamma_th"]), "gamma_naive_A": float(r["gamma_naive_A"]),
                   "gamma_pred_paper1": float(r["gamma_pred_paper1"]), "gamma_pred_refined_A": float(r["gamma_pred_refined_A"]),
                   "gamma_pred_refined_fit_A": g_ref_fit, "gamma_tail_A": float(r["gamma_tail_A"]),
                   "gamma_tail_pred_A": float(r["gamma_tail_pred_A"]), "gamma_naive_B": float(r["gamma_naive_B"]),
                   "gamma_tail_B": float(r["gamma_tail_B"])})
        P(f"   a={a:.2f}: naive_A={float(r['gamma_naive_A']):.4f} predI={float(r['gamma_pred_paper1']):.4f} "
          f"ref(N*)={float(r['gamma_pred_refined_A']):.4f} ref(fit)={g_ref_fit:.4f} tail_A={float(r['gamma_tail_A']):+.4f} "
          f"tailpred_A={float(r['gamma_tail_pred_A']):+.4f} naive_B={float(r['gamma_naive_B']):.4f} tail_B={float(r['gamma_tail_B']):+.4f}")
    out["table3"] = t3

    # ---- Table 4: three-way comparison Euclidean A / geodesic A / rotation B ----
    P("\n[Table 4] doubling-tail exponent: Euclidean A vs geodesic A vs rotation B")
    t4 = []
    ns = _grid(N_FIT_LO, N_FIT_HI)
    for r in scan:
        a = float(r["alpha"])
        Dg = profile_geo(a)
        gtG = gamma_tail(Dg); gnG = gamma_naive(Dg)
        gtG_pred = gamma_of(D_tail_pred("G", a, ns), ns)
        dg_formula = delta_gamma_formula(a, C_LAM, N_STAR) if abs(1 - 3 * a) > 1e-9 else float("nan")
        t4.append({"alpha": a, "gamma_th": 1 - 2 * a,
                   "tail_A": float(r["gamma_tail_A"]), "bias_A": float(r["gamma_tail_A"]) - (1 - 2 * a),
                   "tail_G": gtG, "bias_G": gtG - (1 - 2 * a), "tail_G_pred": gtG_pred,
                   "tail_B": float(r["gamma_tail_B"]), "bias_B": float(r["gamma_tail_B"]) - (1 - 2 * a),
                   "delta_gamma_formula_Nstar": dg_formula, "naive_G": gnG})
        P(f"   a={a:.2f}: tail A={float(r['gamma_tail_A']):+.4f} (bias {float(r['gamma_tail_A']) - (1 - 2 * a):+.4f}, "
          f"formula {dg_formula:+.4f}) | geo {gtG:+.4f} (bias {gtG - (1 - 2 * a):+.4f}, pred {gtG_pred:+.4f}) | "
          f"B {float(r['gamma_tail_B']):+.4f} (bias {float(r['gamma_tail_B']) - (1 - 2 * a):+.4f})")
    out["table4"] = t4
    with open(os.path.join(RES_DIR, "paper03_table4.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(t4[0].keys())); w.writeheader(); w.writerows(t4)
    r045 = next(r for r in t4 if abs(r["alpha"] - 0.45) < 1e-9)
    # Theorem 4.3 at alpha = 0.45: formula at N*, window-averaged 3-term and full predictions
    a = 0.45
    def tail_pred_terms(kmax):
        co = coeffs_euclid(kmax); c = mp.mpf(C_LAM)
        vals = []
        for N in ns:
            N = mp.mpf(int(N)); tot = mp.mpf(0)
            for k, ak in co.items():
                s = k * a
                tot += ak * c ** k * ((2 * N) ** (1 - s) - N ** (1 - s)) / (1 - s)
            vals.append(float(tot))
        return gamma_of(np.array(vals), ns)
    out["theorem43_alpha045"] = {
        "delta_gamma_formula_Nstar": delta_gamma_formula(a, C_LAM, N_STAR),
        "delta_gamma_formula_window_mean": float(np.mean([delta_gamma_formula(a, C_LAM, float(n)) for n in ns])),
        "gamma_tail_pred_2terms": tail_pred_terms(2), "gamma_tail_pred_3terms": tail_pred_terms(3),
        "gamma_tail_pred_4terms": tail_pred_terms(4), "gamma_tail_pred_40terms": tail_pred_terms(40),
        "gamma_tail_measured_A": r045["tail_A"], "gamma_tail_measured_G": r045["tail_G"], "gamma_tail_measured_B": r045["tail_B"],
    }
    P("\n[Theorem 4.3 at alpha = 0.45, c = 0.4]")
    for k, v in out["theorem43_alpha045"].items():
        P(f"   {k}: {v:+.5f}")

    # ---- Table 5: phase-diagram subgrid ----
    P("\n[Table 5] joint phase diagram: measured gamma_tail^+ on the sub-grid alpha in {0.05, 0.15, ..., 0.95}")
    sub = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    look = {(round(float(r["alpha_lam"]), 2), round(float(r["alpha_theta"]), 2)): r for r in phase}
    t5 = []
    for at in reversed(sub):
        row = {"alpha_theta": at}
        for al in sub:
            row[f"{al:.2f}"] = float(look[(al, at)]["gamma_tail_plus"])
        t5.append(row)
        P(f"   a_theta={at:.2f}: " + " ".join(f"{row[f'{al:.2f}']:.3f}" for al in sub))
    mism = sum(1 for r in phase if r["phase_numeric"] != r["phase_theory"])
    maxdev = max(abs(float(r["gamma_tail_plus"]) - float(r["gamma_theory"])) for r in phase)
    n_sing = sum(1 for r in phase if r["phase_theory"] == "singular")
    out["phase"] = {"points": len(phase), "mismatches": mism, "max_dev": maxdev, "singular_points": n_sing,
                    "equivalent_points": len(phase) - n_sing}
    P(f"   points {len(phase)}, singular {n_sing}, equivalent {len(phase) - n_sing}, mismatches {mism}, max |gamma+ - theory| {maxdev:.4f}")
    # deviations by region
    regs = {"both": [], "amp": [], "rot": [], "equiv": []}
    for r in phase:
        al, at = float(r["alpha_lam"]), float(r["alpha_theta"])
        dev = abs(float(r["gamma_tail_plus"]) - float(r["gamma_theory"]))
        key = "both" if (al <= 0.5 and at <= 0.5) else ("amp" if al <= 0.5 else ("rot" if at <= 0.5 else "equiv"))
        regs[key].append(dev)
    out["phase"]["max_dev_by_region"] = {k: max(v) for k, v in regs.items()}
    P("   max deviation by region: " + ", ".join(f"{k} {max(v):.4f}" for k, v in regs.items()))

    # ---- LaTeX rows ----
    def sci(x):
        if x == 0: return "0"
        m, e = f"{x:.1e}".split("e"); return rf"${m}\times10^{{{int(e)}}}$"
    for mech, fname in (("A", "paper03_table1_rows.tex"), ("B", "paper03_table2_rows.tex")):
        with open(os.path.join(RES_DIR, fname), "w", encoding="utf-8") as f:
            for r in t12[mech]:
                dinf = f"{r['D_inf']:.7f}" if r["alpha"] > 0.5 else "--"
                f.write(f"{r['alpha']:.2f} & $10^{{{int(round(np.log10(r['N'])))}}}$ & {r['D_cov']:.7f} & {r['D_pred']:.7f} & "
                        f"{sci(r['diff'])} & {r['D_tail']:.6g} & {r['Pi2']:.6f} & {dinf} \\\\\n")
    with open(os.path.join(RES_DIR, "paper03_table3_rows.tex"), "w", encoding="utf-8") as f:
        for r in t3:
            f.write(f"{r['alpha']:.2f} & {r['gamma_th']:+.2f} & {r['gamma_naive_A']:.4f} & {r['gamma_pred_paper1']:.4f} & "
                    f"{r['gamma_pred_refined_A']:.4f} & {r['gamma_pred_refined_fit_A']:.4f} & {r['gamma_tail_A']:+.4f} & "
                    f"{r['gamma_tail_pred_A']:+.4f} & {r['gamma_naive_B']:.4f} & {r['gamma_tail_B']:+.4f} \\\\\n")
    with open(os.path.join(RES_DIR, "paper03_table4_rows.tex"), "w", encoding="utf-8") as f:
        for r in t4:
            fm = f"{r['delta_gamma_formula_Nstar']:+.4f}" if not np.isnan(r["delta_gamma_formula_Nstar"]) else "--"
            f.write(f"{r['alpha']:.2f} & {r['gamma_th']:+.2f} & {r['tail_A']:+.4f} & {r['bias_A']:+.4f} & {fm} & "
                    f"{r['tail_G']:+.4f} & {r['bias_G']:+.4f} & {r['tail_B']:+.4f} & {r['bias_B']:+.4f} \\\\\n")
    with open(os.path.join(RES_DIR, "paper03_table5_rows.tex"), "w", encoding="utf-8") as f:
        for row in t5:
            f.write(f"{row['alpha_theta']:.2f} & " + " & ".join(f"{row[f'{al:.2f}']:.3f}" for al in sub) + " \\\\\n")

    # ---- Corollary 5.3: finite-N two-power-law cross-over on the joint grid ----
    P("\n[Corollary 5.3] two-power-law cross-over of the joint doubling-tail exponent (361 grid points)")
    A_lead = float(coeffs_euclid()[2] * mp.mpf(C_LAM) ** 2)          # = 0.01
    ns = _grid(N_FIT_LO, N_FIT_HI)
    ca_full, cr_full = coeffs_euclid(), coeffs_rot()

    def weight(gamma, factor):
        """W = factor * A * (2^gamma - 1)/gamma, with the limit A ln 2 at gamma = 0."""
        return factor * A_lead * (np.log(2.0) if abs(gamma) < 1e-12 else (2.0 ** gamma - 1.0) / gamma)

    def tail_comm_pred(al, N):
        """Analytic amplitude tail of blocks N+1..2N = modes 2N+1..4N (Theorem 4.1 at 2N), all series terms."""
        N = mp.mpf(int(N)); tot = mp.mpf(0); c = mp.mpf(C_LAM)
        for k, ak in ca_full.items():
            s = k * al
            tot += ak * c ** k * (mp.log(2) if abs(s - 1) < 1e-12 else ((4 * N) ** (1 - s) - (2 * N) ** (1 - s)) / (1 - s))
        return float(tot)

    def tail_rot_pred(at, N):
        N = mp.mpf(int(N)); tot = mp.mpf(0); c = mp.mpf(C_THETA)
        for k, bk in cr_full.items():
            s = k * at
            tot += bk * c ** k * (mp.log(2) if abs(s - 1) < 1e-12 else ((2 * N) ** (1 - s) - N ** (1 - s)) / (1 - s))
        return float(tot)

    cross_rows = []
    quad_dev_lead = {"both": [], "amp": [], "rot": [], "equiv": []}
    quad_dev_fit = {"both": [], "amp": [], "rot": [], "equiv": []}
    for r in phase:
        al, at = float(r["alpha_lam"]), float(r["alpha_theta"])
        g_meas = float(r["gamma_tail"])
        gl, gt = 1 - 2 * al, 1 - 2 * at
        Wl, Wt = weight(gl, 2.0 ** gl), weight(gt, 1.0)
        # leading closed form at N*: dominant part = larger exponent
        if gl >= gt:
            g1, g2, W1, W2 = gl, gt, Wl, Wt
        else:
            g1, g2, W1, W2 = gt, gl, Wt, Wl
        g_lead = g1 - (g1 - g2) / (1.0 + (W1 / W2) * N_STAR ** (g1 - g2))
        # full analytic prediction regressed over the window
        T = np.array([tail_comm_pred(al, n) + tail_rot_pred(at, n) for n in ns])
        g_fit = gamma_of(T, ns)
        key = "both" if (al <= 0.5 and at <= 0.5) else ("amp" if al <= 0.5 else ("rot" if at <= 0.5 else "equiv"))
        quad_dev_lead[key].append(abs(g_lead - g_meas)); quad_dev_fit[key].append(abs(g_fit - g_meas))
        cross_rows.append({"alpha_lam": al, "alpha_theta": at, "gamma_tail_measured": g_meas,
                           "gamma_max": max(gl, gt), "W_lam": Wl, "W_theta": Wt,
                           "gamma_lead_Nstar": g_lead, "gamma_fit_analytic": g_fit,
                           "dev_lead": g_lead - g_meas, "dev_fit": g_fit - g_meas, "quadrant": key})
    with open(os.path.join(RES_DIR, "paper03_table6_crossover.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cross_rows[0].keys())); w.writeheader(); w.writerows(cross_rows)
    out["corollary53"] = {
        "max_abs_dev_lead_by_quadrant": {k: max(v) for k, v in quad_dev_lead.items()},
        "max_abs_dev_fit_by_quadrant": {k: max(v) for k, v in quad_dev_fit.items()},
        "max_abs_dev_lead_all": max(abs(r["dev_lead"]) for r in cross_rows),
        "max_abs_dev_fit_all": max(abs(r["dev_fit"]) for r in cross_rows),
        "rms_dev_fit_all": float(np.sqrt(np.mean([r["dev_fit"] ** 2 for r in cross_rows]))),
    }
    P("   max |gamma_lead(N*) - measured| by quadrant: " + ", ".join(f"{k} {max(v):.4f}" for k, v in quad_dev_lead.items()))
    P("   max |gamma_fit(analytic) - measured| by quadrant: " + ", ".join(f"{k} {max(v):.4f}" for k, v in quad_dev_fit.items()))
    P(f"   over all 361 points: max |lead - meas| = {out['corollary53']['max_abs_dev_lead_all']:.4f}, "
      f"max |fit - meas| = {out['corollary53']['max_abs_dev_fit_all']:.4f}, rms |fit - meas| = {out['corollary53']['rms_dev_fit_all']:.5f}")
    feats = [(0.55, 0.45), (0.45, 0.55), (0.35, 0.45), (0.45, 0.35), (0.45, 0.45), (0.25, 0.25)]
    out["corollary53"]["featured"] = []
    for al, at in feats:
        r = next(x for x in cross_rows if abs(x["alpha_lam"] - al) < 1e-9 and abs(x["alpha_theta"] - at) < 1e-9)
        out["corollary53"]["featured"].append(r)
        gl, gt = 1 - 2 * al, 1 - 2 * at
        P(f"   (a_lam, a_theta) = ({al:.2f}, {at:.2f}): gamma_lam = {gl:+.2f}, gamma_theta = {gt:+.2f}, "
          f"W_lam = {r['W_lam']:.6f}, W_theta = {r['W_theta']:.6f}, N*^|dg| = {N_STAR ** abs(gl - gt):.3f}, "
          f"lead(N*) = {r['gamma_lead_Nstar']:+.4f}, fit = {r['gamma_fit_analytic']:+.4f}, measured = {r['gamma_tail_measured']:+.4f}")
    with open(os.path.join(RES_DIR, "paper03_table6_rows.tex"), "w", encoding="utf-8") as f:
        for r in out["corollary53"]["featured"]:
            f.write(f"({r['alpha_lam']:.2f}, {r['alpha_theta']:.2f}) & {1 - 2 * r['alpha_lam']:+.2f} & {1 - 2 * r['alpha_theta']:+.2f} & "
                    f"{r['W_lam']:.5f} & {r['W_theta']:.5f} & {r['gamma_lead_Nstar']:+.4f} & {r['gamma_fit_analytic']:+.4f} & "
                    f"{r['gamma_tail_measured']:+.4f} \\\\\n")

    with open(os.path.join(RES_DIR, "paper03_tables.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    with open(os.path.join(RES_DIR, "paper03_tables_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwritten:", RES_DIR)


if __name__ == "__main__":
    main()
