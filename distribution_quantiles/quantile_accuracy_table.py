"""Accuracy of the contour-integral quantiles (Appendix D.1 of the paper).

Rows are working laws, columns probability levels. Each entry is the relative
error of the residue-form quantile integral, evaluated with the paper's contour
height, against a 30-digit reference quantile from Newton's method on the real
CDF. The integral uses a composite 20-point Gauss--Legendre rule in 24-digit
arithmetic. Cells lie in a transformed variable and are no wider than the peak
width h(x) there, except in the far Student tails beyond all tested quantiles;
the mesh is fixed per row and the same for every p. GIG, GH and VG have no
closed-form complex CDF; for them P(kappa) is accumulated along the contour with
the Gauss--Legendre integration matrix.

Usage:
    python quantile_accuracy_table.py          # all rows, about 12 min on 8 cores
    python quantile_accuracy_table.py 4 10 11  # only these rows (0-based, see ROWS)

Requirements: scipy, mpmath. Writes results.json and results.csv beside this
script. A partial run keeps the other rows of an existing results.json.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mpmath as mp
import scipy
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import kv

OUT = Path(__file__).resolve().parent
P_GRID = (0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99)
DPS, REF_DPS, NODES, CHUNK = 24, 30, 20, 150

ROWS = [  # (family, row label, working law, parameters)
    ("GB", "Beta, shapes below one", "beta", (0.3, 0.6)),
    ("GB", "Beta, shapes above one", "beta", (2.5, 3.0)),
    ("GG and GEV", "Gamma, shape below one", "gamma", (0.4,)),
    ("GG and GEV", "Exponential (GEV)", "gamma", (1.0,)),
    ("GIG", "Inverse Gaussian", "gig", (-0.5, 1.0)),
    ("GH", "Hyperbolic", "gh", (1.0, 1.5, 0.5)),
    ("GH", "Normal inverse Gaussian", "gh", (-0.5, 2.0, 1.0)),
    ("Normal and log-normal", "Normal", "normal", ()),
    ("Student's t", "a < 1", "t", (0.5,)),
    ("Student's t", "a = 5", "t", (5.0,)),
    ("VG", "Density unbounded at zero", "vg", (-0.3, 0.15)),
    ("VG", "Density bounded at zero", "vg", (1.5, 0.2)),
]


# ---------------------------------------------------------------- quadrature rule and mesh

_RULE = None


def rule():
    """Gauss--Legendre nodes T, weights W and integration matrix S on [-1, 1].

    S[i][j] integrates the j-th Lagrange basis polynomial from -1 to T[i].
    """
    global _RULE
    if _RULE is None:
        n = NODES
        with mp.workdps(DPS + 15):
            def legendre(t):
                v = [mp.mpf(1), t]
                for j in range(2, n + 1):
                    v.append(((2 * j - 1) * t * v[-1] - (j - 1) * v[-2]) / j)
                return v

            T, W = [], []
            for k in range(1, n + 1):
                t = mp.cos(mp.pi * (k - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
                for _ in range(60):
                    v = legendre(t)
                    step = v[n] / (n * (t * v[n] - v[n - 1]) / (t * t - 1))
                    t -= step
                    if abs(step) < mp.mpf(10) ** -(DPS + 12):
                        break
                v = legendre(t)
                dp = n * (t * v[n] - v[n - 1]) / (t * t - 1)
                T.append(t)
                W.append(2 / ((1 - t * t) * dp * dp))
            V, A = mp.matrix(n, n), mp.matrix(n, n)
            for i, t in enumerate(T):
                v = legendre(t)
                for k in range(n):
                    V[i, k] = v[k]
                    A[i, k] = t + 1 if k == 0 else (v[k + 1] - v[k - 1]) / (2 * k + 1)
            S = A * V ** -1
            _RULE = (T, W, [[S[i, j] for j in range(n)] for i in range(n)])
    return _RULE


def make_cells(lo, hi, width):
    """Consecutive cells no wider than the local peak width; fixed for all p."""
    cells, s = [], lo
    while s < hi:
        w = min(width(s), width(min(s + width(s), hi)))
        e = min(s + w, hi)
        cells.append((s, e))
        s = e
    return cells


def graded_cells(first, hi, growth=1.2, cap=1.0):
    """Cells on [0, hi] growing geometrically from width `first` up to `cap`."""
    cells, s, w = [], 0.0, first
    while s < hi:
        e = min(s + w, hi)
        cells.append((s, e))
        s, w = e, min(w * growth, cap)
    return cells


def singular_quad(f, a, x):
    """int_0^x f(y) dy for f ~ y^(2a) at 0; y = x u^m with m = 1/(2a+1) removes the singularity."""
    m = 1 / (2 * mp.mpf(a) + 1) if a < 0 else mp.mpf(1)
    return mp.quad(lambda u: f(x * u ** m) * x * m * u ** (m - 1), [0, 1])


# ---------------------------------------------------------------- working laws and contours

def build(kind, params):
    """Return segments [(cells, point)] with point(s) -> (z, dz/ds, g(z), P(z) or None)."""
    if kind == "beta":
        a, b = params
        A1, B1, Bab = abs(a - 1), abs(b - 1), mp.beta(a, b)
        ea, eb = mp.mpf(a) - 1, mp.mpf(b) - 1  # exact exponents (0.3 - 1 rounds in floats)

        def point(s):  # logistic variable x = 1/(1+e^-s); 1-x kept separately
            e = mp.exp(-s)
            x, xc = 1 / (1 + e), e / (1 + e)
            D = 1 + A1 * xc + B1 * x
            h = x * xc / D
            dh = ((xc - x) * D - x * xc * (B1 - A1)) / D ** 2
            z, zc = mp.mpc(x, h), mp.mpc(xc, -h)
            P = (mp.betainc(a, b, 0, z, regularized=True) if x <= 0.5
                 else 1 - mp.betainc(b, a, 0, zc, regularized=True))
            return z, mp.mpc(1, dh) * x * xc, z ** ea * zc ** eb / Bab, P

        width = lambda s: 1 / (1 + A1 / (1 + math.exp(s)) + B1 / (1 + math.exp(-s)))
        return {"segments": [(make_cells(-60 / (a + 1), 60 / b + 5, width), point)]}

    if kind == "gamma":
        (a,) = params
        A1, Ga, ea = abs(a - 1), mp.gamma(a), mp.mpf(a) - 1

        def point(s):  # x = e^s
            x = mp.exp(s)
            z = mp.mpc(x, x / (1 + A1 + x))
            return (z, mp.mpc(1, (1 + A1) / (1 + A1 + x) ** 2) * x,
                    z ** ea * mp.exp(-z) / Ga, mp.gammainc(a, 0, z, regularized=True))

        width = lambda s: 1 / (1 + A1 + math.exp(s))
        return {"segments": [(make_cells(-60 / (a + 1), math.log(80), width), point)]}

    if kind == "gig":
        a, b = params
        A1, K2, ea = abs(a - 1), 2 * mp.besselk(a, b), mp.mpf(a) - 1
        h = lambda x: x * x / (b + 2 * A1 * x + b * x * x)

        def point(s):  # x = e^s
            x = mp.exp(s)
            D = b + 2 * A1 * x + b * x * x
            z = mp.mpc(x, x * x / D)
            dz = mp.mpc(1, (2 * b * x + 2 * A1 * x * x) / D ** 2) * x
            return z, dz, z ** ea * mp.exp(-mp.mpf(b) / 2 * (z + 1 / z)) / K2, None

        width = lambda s: math.exp(s) / (b + 2 * A1 * math.exp(s) + b * math.exp(2 * s))
        x0 = b / 130  # exp(-65) below
        return {"segments": [(make_cells(math.log(x0), math.log(130 / b + 20), width), point)],
                "cumulative": True, "x0": x0, "h": h}

    if kind == "gh":
        a, b, d = params
        D = mp.mpf(1.5) + abs(a - 0.5) - a
        h = min(mp.mpf(1) / 2, mp.exp(-b * (1 - mp.sqrt(3) / 2)) * (mp.sqrt(3) / 2) ** D
                / (2 * abs(d) + 4 * b / mp.sqrt(3) + 8 * D / 3))
        r = mp.sqrt(mp.mpf(b) ** 2 - mp.mpf(d) ** 2)
        nu = mp.mpf(a) - mp.mpf(1) / 2
        C = r ** a / (mp.sqrt(2 * mp.pi) * mp.mpf(b) ** nu * mp.besselk(a, r))

        def point(x):
            z = mp.mpc(x, h)
            w = 1 + z * z
            return (z, mp.mpc(1),
                    C * mp.exp(d * z) * w ** (nu / 2) * mp.besselk(nu, b * mp.sqrt(w)), None)

        hf = float(h)
        lo, hi = -(62 / (b + d) + 5), 62 / (b - d) + 5
        return {"segments": [(make_cells(lo, hi, lambda s: hf), point)], "cumulative": True}

    if kind == "normal":
        def point(x):
            z = mp.mpc(x, 1 / (2 * (1 + x * x)))
            return (z, mp.mpc(1, -x / (1 + x * x) ** 2), mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi),
                    (1 + mp.erf(z / mp.sqrt(2))) / 2)

        return {"segments": [(make_cells(-11.0, 11.0, lambda s: 1 / (2 * (1 + s * s))), point)]}

    if kind == "t":
        (a,) = params
        am = mp.mpf(a)
        ca = mp.gamma((am + 1) / 2) / (mp.sqrt(am * mp.pi) * mp.gamma(am / 2))
        h = mp.sqrt(am) / (2 * (am + 1))
        X = 1200.0 if a < 1 else 30.0  # uniform cells of width h on [-X, X]
        U = math.log(10 ** (26 / a) / X)  # logarithmic tails out to x^-a = 1e-26

        def at(x, dz):
            z = mp.mpc(x, h)
            return (z, dz, ca * (1 + z * z / am) ** (-(am + 1) / 2),
                    mp.mpf(1) / 2 + ca * z * mp.hyp2f1(mp.mpf(1) / 2, (am + 1) / 2,
                                                         mp.mpf(3) / 2, -z * z / am))

        core = lambda x: at(x, mp.mpc(1))
        right = lambda u: at(X * mp.exp(u), mp.mpc(X * mp.exp(u)))
        left = lambda u: at(-X * mp.exp(u), mp.mpc(X * mp.exp(u)))
        hf = float(h)
        tail = graded_cells(hf / X, U)  # start at the core width, so peaks near X stay resolved
        return {"segments": [(tail, left), (make_cells(-X, X, lambda s: hf), core), (tail, right)],
                "observable": mp.atan}

    if kind == "vg":
        a, d, eps = params
        c = mp.sqrt(1 + mp.mpf(d) ** 2)
        K = mp.sqrt(mp.pi) * (2 * c) ** a * mp.gamma(mp.mpf(a) + mp.mpf(1) / 2)
        Dh, A, B = 8 * 2 ** abs(a), 1 + 2 * abs(a), float(c) + abs(d)
        h = lambda x: x / (Dh * (A + B * x))
        g = lambda z: mp.exp(eps * d * z) * z ** a * mp.besselk(a, c * z) / K

        def point(s):  # x = e^s
            x = mp.exp(s)
            z = mp.mpc(x, h(x))
            return z, mp.mpc(1, A / (Dh * (A + B * x) ** 2)) * x, g(z), None

        x0, x_end = 1e-14, 62 / (float(c) - eps * d) + 5
        width = lambda s: 1 / (Dh * (A + B * math.exp(s)))
        return {"segments": [(make_cells(math.log(x0), math.log(x_end), width), point)],
                "cumulative": True, "x0": x0, "h": h, "g": g}

    raise ValueError(kind)


def units():
    """Evaluation units: VG splits into its two half-axes at p0."""
    out = []
    for r, (_, _, kind, params) in enumerate(ROWS):
        if kind != "vg":
            out.append(((r, 0), kind, params, [mp.mpf(p) for p in P_GRID], list(P_GRID)))
            continue
        a, d = params
        c = mp.sqrt(1 + mp.mpf(d) ** 2)
        ah = mp.mpf(a) + mp.mpf(1) / 2
        p0 = mp.betainc(ah, ah, 0, (c - d) / (2 * c), regularized=True)
        for eps in (1, -1):
            side = [p for p in P_GRID if (p - p0) * eps > 0]
            if side:
                out.append(((r, eps), kind, (a, d, eps), [abs(p - p0) for p in side], side))
    return out


# ---------------------------------------------------------------- parallel pass: integrand values

def chunk_task(args):
    key, kind, params, seg, i0, i1, targets = args
    mp.mp.dps = DPS
    unit = build(kind, params)
    cells, point = unit["segments"][seg]
    T, W, _ = rule()
    if unit.get("cumulative"):  # return node values; P is accumulated in order afterwards
        data = []
        for sl, sr in cells[i0:i1]:
            sl, sr = mp.mpf(sl), mp.mpf(sr)
            half, mid = (sr - sl) / 2, (sr + sl) / 2
            nodes = []
            for t in T:
                z, dz, g, _ = point(mid + half * t)
                nodes.append((z, dz, g * dz))
            data.append((half, nodes))
        return key, seg, i0, data
    G = unit.get("observable")
    J = [mp.mpc(0)] * len(targets)
    for sl, sr in cells[i0:i1]:
        sl, sr = mp.mpf(sl), mp.mpf(sr)
        half, mid = (sr - sl) / 2, (sr + sl) / 2
        for t, w in zip(T, W):
            z, dz, g, P = point(mid + half * t)
            term = half * w * (z if G is None else G(z)) * g * dz
            for k, p in enumerate(targets):
                J[k] += term / (P - p)
    return key, seg, i0, J


def cumulative_pass(kind, params, targets, ps, parts):
    """Accumulate P(kappa) cell by cell and form the residue (and GIG log-form) sums."""
    unit = build(kind, params)
    T, W, S = rule()
    n, m = len(T), len(targets)
    P_run, J, L = mp.mpc(0), [mp.mpc(0)] * m, None
    if kind in ("vg", "gig"):  # start exactly where the mesh starts, x0 = exp(s0)
        x0 = mp.exp(mp.mpf(unit["segments"][0][0][0][0]))
    if kind == "vg":  # straight segment from 0 to kappa(x0); P is tiny there
        g = unit["g"]
        k0 = mp.mpc(x0, unit["h"](x0))
        with mp.workdps(DPS + 10):
            P_run = singular_quad(lambda t: g(t * k0) * k0, params[0], 1)
            M0 = singular_quad(lambda t: t * k0 * g(t * k0) * k0, params[0], 1)
        J = [-M0 / dp for dp in targets]
    if kind == "gig":  # log form on (0, x0), where P ~ exp(-65)
        k0 = mp.mpc(x0, unit["h"](x0))
        L = [k0 * (mp.log(mp.mpf(p) / (1 - mp.mpf(p))) + mp.j * mp.pi) for p in ps]
    for _, _, data in parts:
        for half, nodes in data:
            F = [Fv for _, _, Fv in nodes]
            Pn = [P_run + half * mp.fsum(S[i][j] * F[j] for j in range(n)) for i in range(n)]
            P_run += half * mp.fsum(W[j] * F[j] for j in range(n))
            for j, (z, dz, Fj) in enumerate(nodes):
                wz = half * W[j] * z * Fj
                for k in range(m):
                    J[k] += wz / (Pn[j] - targets[k])
                if L is not None:
                    wl = half * W[j] * dz
                    for k, p in enumerate(ps):
                        L[k] += wl * mp.log((Pn[j] - p) / (1 - mp.mpf(p)))
    return J, L, P_run


# ---------------------------------------------------------------- 30-digit references

def newton(F, f, p, x):
    x = mp.mpf(x)
    done = False
    for _ in range(25):
        step = (F(x) - p) / f(x)
        x -= step
        if done:
            return x
        done = abs(step) <= abs(x) * mp.mpf(10) ** (6 - REF_DPS)
    raise RuntimeError(f"Newton did not converge at p={p}")


def reference_task(r):
    mp.mp.dps = REF_DPS
    _, _, kind, params = ROWS[r]
    extra = {}
    if kind == "beta":
        a, b = params
        B = mp.beta(a, b)
        F = lambda x: mp.betainc(a, b, 0, x, regularized=True)
        f = lambda x: x ** (a - 1) * (1 - x) ** (b - 1) / B
        start = lambda p: stats.beta.ppf(p, a, b)
    elif kind == "gamma":
        (a,) = params
        Ga = mp.gamma(a)
        F = lambda x: mp.gammainc(a, 0, x, regularized=True)
        f = lambda x: x ** (a - 1) * mp.exp(-x) / Ga
        start = lambda p: stats.gamma.ppf(p, a)
    elif kind == "gig":
        a, b = params
        K2, ea = 2 * mp.besselk(a, b), mp.mpf(a) - 1
        f = lambda x: x ** ea * mp.exp(-mp.mpf(b) / 2 * (x + 1 / x)) / K2
        F = lambda x: mp.quad(f, [0, x])
        start = lambda p: stats.geninvgauss.ppf(p, a, b)
        extra["norm_err"] = float(abs(mp.quad(f, [0, 1, mp.inf]) - 1))
    elif kind == "gh":
        a, b, d = params
        rr = mp.sqrt(mp.mpf(b) ** 2 - mp.mpf(d) ** 2)
        nu = mp.mpf(a) - mp.mpf(1) / 2
        C = rr ** a / (mp.sqrt(2 * mp.pi) * mp.mpf(b) ** nu * mp.besselk(a, rr))
        f = lambda x: (C * mp.exp(d * x) * (1 + x * x) ** (nu / 2)
                       * mp.besselk(nu, b * mp.sqrt(1 + x * x)))
        F = lambda x: mp.quad(f, [-mp.inf, 0, x] if x > 0 else [-mp.inf, x])
        start = lambda p: stats.genhyperbolic.ppf(p, a, b, d)
        extra["norm_err"] = float(abs(mp.quad(f, [-mp.inf, 0, mp.inf]) - 1))
    elif kind == "normal":
        F, f, start = mp.ncdf, mp.npdf, stats.norm.ppf
    elif kind == "t":
        (a,) = params
        am = mp.mpf(a)
        ca = mp.gamma((am + 1) / 2) / (mp.sqrt(am * mp.pi) * mp.gamma(am / 2))
        F = lambda x: mp.mpf(1) / 2 + ca * x * mp.hyp2f1(mp.mpf(1) / 2, (am + 1) / 2,
                                                          mp.mpf(3) / 2, -x * x / am)
        f = lambda x: ca * (1 + x * x / am) ** (-(am + 1) / 2)
        start = lambda p: stats.t.ppf(p, a)
    elif kind == "vg":
        a, d = params
        c = mp.sqrt(1 + mp.mpf(d) ** 2)
        K = mp.sqrt(mp.pi) * (2 * c) ** a * mp.gamma(mp.mpf(a) + mp.mpf(1) / 2)
        f = lambda x: mp.exp(d * x) * abs(x) ** a * mp.besselk(a, c * abs(x)) / K
        fneg = lambda y: f(-y)
        # half-axis masses, with the singularity at 0 removed by substitution
        p0_num = singular_quad(fneg, a, 1) + mp.quad(fneg, [1, mp.inf])
        mass_pos = singular_quad(f, a, 1) + mp.quad(f, [1, mp.inf])
        F = lambda x: p0_num + singular_quad(f, a, x) if x > 0 else p0_num - singular_quad(fneg, a, -x)
        cf, Kf = float(c), float(K)
        fd = lambda x: math.exp(d * x) * abs(x) ** a * kv(a, cf * abs(x)) / Kf if x else 0.0
        p0d = quad(fd, -60, 0, limit=200)[0]
        cdfd = lambda x: (p0d + quad(fd, 0, x, limit=200)[0]) if x >= 0 else quad(fd, -60, x, limit=200)[0]
        start = lambda p: brentq(lambda x: cdfd(x) - p, -40, 40, xtol=1e-14)
        ah = mp.mpf(a) + mp.mpf(1) / 2
        p0_formula = mp.betainc(ah, ah, 0, (c - d) / (2 * c), regularized=True)
        extra["p0_formula_err"] = float(abs(p0_formula - p0_num))
        extra["norm_err"] = float(abs(p0_num + mass_pos - 1))
    else:
        raise ValueError(kind)
    refs = [newton(F, f, p, start(p)) for p in P_GRID]
    return r, refs, extra


# ---------------------------------------------------------------- main

def main(selected):
    """Evaluate the selected rows; other rows are kept from an existing results.json."""
    t0 = time.time()
    mp.mp.dps = DPS
    U = [u for u in units() if u[0][0] in selected]
    tasks, sizes = [], {}
    for key, kind, params, targets, _ in U:
        segs = build(kind, params)["segments"]
        sizes[key] = sum(len(cells) for cells, _ in segs)
        for seg, (cells, _) in enumerate(segs):
            for i0 in range(0, len(cells), CHUNK):
                tasks.append((key, kind, params, seg, i0, min(i0 + CHUNK, len(cells)), targets))
    print(f"{len(tasks)} chunks, {sum(sizes.values())} cells x {NODES} nodes", flush=True)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        ref_futures = [pool.submit(reference_task, r) for r in selected]
        chunks = list(pool.map(chunk_task, tasks))
        refs = {r: (q, extra) for r, q, extra in (fut.result() for fut in ref_futures)}
    print(f"parallel pass done after {time.time() - t0:.0f} s", flush=True)

    parts = {}
    for key, seg, i0, data in chunks:
        parts.setdefault(key, []).append((seg, i0, data))
    Q = {r: {} for r in selected}
    Q_log, mass = {}, {}
    for key, kind, params, targets, ps in U:
        r, sign = key
        ordered = sorted(parts[key], key=lambda t: (t[0], t[1]))
        if build(kind, params).get("cumulative"):
            J, L, P_end = cumulative_pass(kind, params, targets, ps, ordered)
            mass[key] = P_end
            if L is not None:
                Q_log[r] = {p: mp.im(Lk) / mp.pi for p, Lk in zip(ps, L)}
        else:
            J = [mp.fsum(part[2][k] for part in ordered) for k in range(len(targets))]
        for p, Jk in zip(ps, J):
            val = -mp.im(Jk) / mp.pi
            if kind == "t":
                val = mp.tan(val)
            if kind == "vg":
                val = sign * val
            Q[r][p] = val

    old = {}
    if len(selected) < len(ROWS) and (OUT / "results.json").exists():
        old = {row["label"]: row for row in json.loads((OUT / "results.json").read_text())["rows"]}
    results = []
    for r, (family, label, kind, params) in enumerate(ROWS):
        if r not in selected:
            if label in old:
                results.append(old[label])
            continue
        q_ref, extra = refs[r]
        errs = [float(abs(Q[r][p] - qr) / abs(qr)) for p, qr in zip(P_GRID, q_ref)]
        row = {"family": family, "label": label, "law": kind, "params": params,
               "rel_err": dict(zip(map(str, P_GRID), errs)),
               "q": {str(p): mp.nstr(Q[r][p], 22) for p in P_GRID},
               "q_ref": {str(p): mp.nstr(qr, 28) for p, qr in zip(P_GRID, q_ref)},
               "cells": sum(sizes[k] for k, *_ in U if k[0] == r), **extra}
        if r in Q_log:
            row["log_form_rel_err"] = max(float(abs(Q_log[r][p] - qr) / abs(qr))
                                          for p, qr in zip(P_GRID, q_ref))
        for key in mass:
            if key[0] == r:
                expected = 1 if kind != "vg" else None
                row.setdefault("final_mass", []).append(mp.nstr(mass[key], 25))
                if expected is not None:
                    row["mass_err"] = float(abs(mass[key] - expected))
        results.append(row)

    csv_rows = []
    for row in results:
        errs = [row["rel_err"][str(p)] for p in P_GRID]
        csv_rows += [(row["family"], row["label"], p, row["q"][str(p)], row["q_ref"][str(p)], e)
                     for p, e in zip(P_GRID, errs)]
        print(f"{row['label']:28s} " + " ".join(f"{e:8.1e}" for e in errs), flush=True)

    (OUT / "results.json").write_text(json.dumps(
        {"p_grid": P_GRID, "dps": DPS, "ref_dps": REF_DPS, "nodes_per_cell": NODES,
         "versions": {"python": sys.version.split()[0], "scipy": scipy.__version__,
                      "mpmath": mp.__version__},
         "seconds": round(time.time() - t0), "rows": results}, indent=2))
    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "row", "p", "contour_quantile", "reference_quantile", "rel_err"])
        w.writerows(csv_rows)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    # optional row indices (0-based, in ROWS order) recompute only those rows
    main([int(a) for a in sys.argv[1:]] or list(range(len(ROWS))))
