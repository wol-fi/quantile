"""Evaluate the implied-volatility integral, Eq. (25) of the paper, without root finding.

This is the experiment of Appendix D.2. Requirements: numpy, scipy, matplotlib,
mpmath. Writes results.csv, results.json and the plots beside this script.

Usage:
    python verify_implied_volatility.py           # full experiment
    python verify_implied_volatility.py --smoke   # four quick cases, prints only
    python verify_implied_volatility.py --skip-mp # skip the 35- and 50-digit checks
"""

from __future__ import annotations

import argparse
import cmath
import csv
import json
import math
from pathlib import Path
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import IntegrationWarning, quad
from scipy.special import ndtr

OUT = Path(__file__).resolve().parent
KS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
VS = (0.1, 0.2, 0.4, 0.8, 1.5, 3.0)
# Fixed before any price or reference volatility is supplied to the integral.
MESH = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4,
        0.6, 0.8, 0.9, 0.95, 0.98, 0.995, 1.0)


def bs_price_mp(k, v):
    """Forward-normalized Black--Scholes call; erfc avoids tail subtraction."""
    k, v = mp.mpf(k), mp.mpf(v)
    d1 = v / 2 - k / v
    d2 = -v / 2 - k / v
    return (mp.erfc(-d1 / mp.sqrt(2))
            - mp.exp(k) * mp.erfc(-d2 / mp.sqrt(2))) / 2


def reference_price(k: float, v: float) -> float:
    with mp.workdps(100):
        return float(bs_price_mp(str(k), str(v)))


def contour_pair(t: float, k: float) -> tuple[complex, complex]:
    """nu(t) and nu'(t), as defined in Section 4; principal square root."""
    x = t / (1 - t)
    den = k * k + x + x * x / 4
    kap = complex(x, x * x / den)
    kap_x = complex(1, (2 * k * k * x + x * x) / (den * den))
    nu = cmath.sqrt(kap)
    nu_t = kap_x / (2 * (1 - t)**2 * nu)
    return nu, nu_t


def complex_ig_cdf(nu: complex, k: float) -> complex:
    """P_IG(4/nu^2; mean=2/k, shape=1), in its normal-CDF formula."""
    return complex(ndtr(k / nu - nu / 2)
                   + math.exp(k) * ndtr(-k / nu - nu / 2))


def stable_log(nu: complex, k: float, c: float) -> complex:
    """Exactly Log(1-P_IG/(1-c)), evaluated without subtracting near-equals.

    Near the right endpoint use log1p(-P_IG/(1-c)). Else use the algebraically
    identical (b-c)/(1-c), where b is the normalized Black--Scholes call price at
    total volatility nu, avoiding 1-P_IG when P_IG rounds to 1.
    Zero imaginary part from underflow is approached from the upper half-plane.
    """
    ig = complex_ig_cdf(nu, k)
    if abs(ig) < 0.125 * (1 - c):
        return complex(np.log1p(-ig / (1 - c)))
    b = complex(ndtr(nu / 2 - k / nu)
                - math.exp(k) * ndtr(-nu / 2 - k / nu))
    arg = (b - c) / (1 - c)
    if arg.imag < -1e-12 * max(abs(arg), 1e-300):
        raise ArithmeticError(f"Contour left the upper CDF half-plane: {arg}")
    return cmath.log(complex(arg.real, max(0.0, arg.imag)))


def recover_volatility(k: float, c: float, maturity: float = 1.0,
                       tol: float = 2e-12) -> dict:
    """Direct quadrature of Eq. (25); no inversion and no volatility input.

    The secondary substitution t=s^2 removes the t^(-1/2) lower-end behavior.
    Both integrations have finite interval (0,1); no tail is truncated.
    """
    if not (k > 0 and 0 < c < 1 and maturity > 0):
        raise ValueError("Require k>0, 0<c<1 and maturity>0")

    def integrand(s):
        if s == 0:
            return math.pi
        if s == 1:
            return 0.0
        nu, nu_t = contour_pair(s * s, k)
        return (nu_t * stable_log(nu, k, c)).imag * 2 * s

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        value, err = quad(integrand, 0, 1, points=MESH[1:-1],
                          epsabs=tol, epsrel=tol, limit=800)
    scale = math.pi * math.sqrt(maturity)
    return {"sigma_integral": value / scale, "quad_error_estimate": err / scale,
            "warnings": [str(w.message) for w in caught]}


def high_precision_check(k: str, v: str, dps: int) -> dict:
    """Independent arbitrary-precision IG evaluation and quadrature."""
    start = time.perf_counter()
    with mp.workdps(dps):
        kk, vv = mp.mpf(k), mp.mpf(v)
        c = bs_price_mp(kk, vv)
        def integrand(s):
            if s == 0:
                return mp.pi
            if s == 1:
                return mp.mpf(0)
            t = s * s
            x = t / (1 - t)
            d = kk**2 + x + x**2 / 4
            kap = x + 1j * x**2 / d
            kap_x = 1 + 1j * (2 * kk**2 * x + x**2) / d**2
            nu = mp.sqrt(kap)
            # Evaluate the IG CDF in conventional mean/shape parameters.
            z, mean = 4 / kap, 2 / kk
            a = mp.sqrt(1 / z) * (z / mean - 1)
            b = -mp.sqrt(1 / z) * (z / mean + 1)
            ig = (mp.erfc(-a / mp.sqrt(2))
                  + mp.exp(2 / mean) * mp.erfc(-b / mp.sqrt(2))) / 2
            arg = 1 - ig / (1 - c)
            # The exact argument lies above the real axis, with argument pi in the
            # limit t -> 0. Rounding can flip the sign of its tiny imaginary part,
            # which must not move the logarithm to the lower branch (-pi).
            if mp.im(arg) < -mp.mpf(10)**(6 - dps) * abs(arg):
                raise ArithmeticError(f"Contour left the upper CDF half-plane: {arg}")
            if mp.im(arg) <= 0:
                logarithm = mp.log(mp.mpc(mp.re(arg), 0))
            else:
                logarithm = mp.log(arg)
            nu_s = kap_x * s / ((1 - t)**2 * nu)
            return mp.im(nu_s * logarithm)
        result = mp.quad(integrand, [mp.mpf(str(s)) for s in MESH],
                         method="gauss-legendre", maxdegree=8) / mp.pi
        record = {"k": k, "v": v, "dps": dps,
                  "recovered_total_volatility": mp.nstr(result, dps),
                  "absolute_error": mp.nstr(abs(result - vv), 8),
                  "relative_error": mp.nstr(abs(result - vv) / vv, 8),
                  "seconds": time.perf_counter() - start}
    return record


def make_plot(grid: list[dict], curves: list[dict], paper: bool = False):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    if paper:
        fig, (ax, err_ax) = plt.subplots(2, 1, figsize=(6.6, 8.2),
                                       gridspec_kw={"height_ratios": [1.1, 1]})
    else:
        fig, (ax, err_ax) = plt.subplots(1, 2, figsize=(13.5, 5.1),
                                       gridspec_kw={"width_ratios": [1.25, 1]})
    colors = ("#2463a6", "#008675", "#cf7f14", "#9a448b")
    dense = np.linspace(0.035, 2.05, 800)
    for color, k in zip(colors, (0.01, 0.1, 0.5, 1.0)):
        prices = ndtr(dense / 2 - k / dense) - np.exp(k) * ndtr(-dense / 2 - k / dense)
        ax.plot(prices, dense, color=color, lw=1.8, label=fr"$\mathsf{{k}}={k:g}$")
        pts = [r for r in curves if r["k"] == k]
        ax.scatter([r["c"] for r in pts], [r["sigma_integral"] for r in pts],
                   s=22, facecolors="white", edgecolors=color, linewidths=1.1, zorder=3)
    ax.set(xlabel=r"Forward-normalized call price $\mathsf{c}$",
           ylabel=r"Implied volatility $\sigma_{\mathrm{BS}}$ ($\mathsf{T}=1$)",
           xlim=(-0.008, 0.70), ylim=(0, 2.08), title="Direct evaluation of the finite-interval formula")
    ax.grid(alpha=0.16)
    ax.legend(title="Log-moneyness", loc="lower right", frameon=False)
    ax.text(0.02, 0.97, "Lines: known Black-Scholes prices and volatilities\n"
            "Circles: contour integral, with only (k, c, T) as inputs",
            transform=ax.transAxes, va="top", fontsize=9.3)

    errors = np.array([[next(r["relative_error"] for r in grid
                             if r["k"] == k and r["v_true"] == v)
                        for v in VS] for k in KS])
    logs = np.log10(np.maximum(errors, 1e-16))
    color_max = max(-14.0, math.ceil(float(np.max(logs))))
    im = err_ax.imshow(logs, origin="lower", aspect="auto", cmap="viridis_r",
                       vmin=-16, vmax=color_max)
    err_ax.set(xticks=range(len(VS)), xticklabels=[str(v) for v in VS],
               yticks=range(len(KS)), yticklabels=[str(k) for k in KS],
               xlabel=r"Known total volatility $\sigma_{\mathrm{BS}}\sqrt{\mathsf{T}}$",
               ylabel=r"Log-moneyness $\mathsf{k}$",
               title="Relative recovery error: 42 parameter combinations")
    for i in range(len(KS)):
        for j in range(len(VS)):
            err_ax.text(j, i, f"{errors[i,j]:.0e}", ha="center", va="center",
                        fontsize=8.5, color="black" if logs[i,j] < (-16 + color_max) / 2 else "white")
    fig.colorbar(im, ax=err_ax, label="log10(relative error)", fraction=0.047, pad=0.04)
    fig.suptitle("Numerical verification of the Black-Scholes contour inverse",
                 fontsize=12 if paper else 15, y=1.01)
    fig.tight_layout()
    stem = "implied_volatility_verification_paper" if paper else "implied_volatility_verification"
    fig.savefig(OUT / (stem + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(OUT / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-mp", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        for k, v in ((0.01, 0.2), (0.5, 0.4), (2.0, 0.1), (0.1, 3.0)):
            c = reference_price(k, v)
            result = recover_volatility(k, c)
            print(json.dumps({"k": k, "v_true": v, "c": c, **result,
                              "error": result["sigma_integral"] - v}), flush=True)
        return

    start = time.perf_counter()
    grid, curves = [], []
    for i, k in enumerate(KS):
        for j, v in enumerate(VS):
            maturity = (0.25, 1.0, 4.0)[(i + j) % 3]
            sigma = v / math.sqrt(maturity)
            c = reference_price(k, v)
            result = recover_volatility(k, c, maturity)
            recovered = result["sigma_integral"]
            with mp.workdps(100):
                repriced = float(bs_price_mp(str(k), mp.mpf(recovered) * mp.sqrt(maturity)))
            grid.append({"k": k, "v_true": v, "T": maturity, "c": c,
                         "sigma_true": sigma, **result,
                         "absolute_error": abs(recovered - sigma),
                         "relative_error": abs(recovered - sigma) / sigma,
                         "repricing_error": abs(repriced - c)})
        print(f"Verified grid row k={k:g}", flush=True)
    for k in (0.01, 0.1, 0.5, 1.0):
        for v in np.linspace(0.1, 2.0, 17):
            c = reference_price(k, float(v))
            curves.append({"k": k, "v_true": float(v), "c": c,
                           **recover_volatility(k, c)})
        print(f"Computed plot curve k={k:g}", flush=True)

    high_precision = []
    if not args.skip_mp:
        for k, v in (("0.05", "0.4"), ("0.5", "0.8"), ("1", "1.5")):
            for dps in (35, 50):
                record = high_precision_check(k, v, dps)
                high_precision.append(record)
                print("High precision " + json.dumps(record), flush=True)
                if mp.mpf(record["relative_error"]) > mp.mpf(10)**(10 - dps):
                    raise AssertionError(f"High-precision check failed: {record}")

    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0]))
        writer.writeheader()
        for row in grid:
            writer.writerow({**row, "warnings": " | ".join(row["warnings"])})
    make_plot(grid, curves)
    make_plot(grid, curves, paper=True)
    summary = {"formula": "Eq. (25), nu(t)=sqrt(kappa(t/(1-t)))",
               "method": "Adaptive finite-interval quadrature after t=s^2; no root finding",
               "mesh_independent_of_target": True, "epsabs_epsrel": 2e-12,
               "grid_count": len(grid), "plot_integral_count": len(curves),
               "k_range": [min(KS), max(KS)], "v_range": [min(VS), max(VS)],
               "T_values": [0.25, 1, 4],
               "c_range": [min(r["c"] for r in grid), max(r["c"] for r in grid)],
               "max_absolute_sigma_error": max(r["absolute_error"] for r in grid),
               "max_relative_sigma_error": max(r["relative_error"] for r in grid),
               "max_absolute_repricing_error": max(r["repricing_error"] for r in grid),
               "quadrature_warnings": [r for r in grid + curves if r["warnings"]],
               "high_precision": high_precision, "grid": grid, "plot_points": curves,
               "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                            "matplotlib": matplotlib.__version__, "mpmath": mp.__version__},
               "seconds": time.perf_counter() - start}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("grid", "plot_points", "high_precision", "quadrature_warnings")},
                     indent=2), flush=True)
    if summary["max_relative_sigma_error"] > 1e-8:
        raise AssertionError("Recovery error exceeded the verification threshold 1e-8")


if __name__ == "__main__":
    main()
