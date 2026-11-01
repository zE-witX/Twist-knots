"""State integrals of each twist knot, as executable integrands for kdb.series.saddle.

Every knot carries two integrals:
  * 'nz'     : the Neumann-Zagier / Andersen-Kashaev integral
               4_1: GGM (113)  e^{-2 pi i u^2} int Phi_b(v) Phi_b(u+v) e^{-pi i (v^2 + 4uv)} dv
               5_2: GGM (230)  int Phi_b(v) Phi_b(v+u) Phi_b(v-u) e^{-2 pi i v^2} dv
               6_1: 6_1/7_2 note (int61),  7_2: (int72)   (2-dimensional)
  * 'marino' : the new state integral I_{K_p}(z, w) of Marino 5.4 (any p), see kdb.series.twist.
Rescaling s = S/(2 pi b), u = U/(2 pi b), h = 2 pi i b^2:
  Phi_b(X/(2 pi b) + shifts) -> Dilog with -e^X = t0 e^{a.S + alpha h};  c_b shifts give t0 -> -t0, alpha = 1/2;
  e^{pi i P(s,u)} (P quadratic) -> exp(-P(S,U)/(2h)), i.e. e^{-pi i P} -> exp(+P/(2h));   e^{2 pi (b m - n/b).s} -> descendants (not used here).
Critical points are found by Newton's method on the exponentiated gradient (multistart); sectors are labelled
by the volume sum_terms c * D(t) (Bloch-Wigner), which is lift independent.
"""
from __future__ import annotations

import random
from fractions import Fraction
from typing import Dict, List, Tuple

from flint import acb, acb_mat, arb, ctx

from .saddle import Dilog, Integrand, Quadratic, expand, li_s


def _acb(v):
    if isinstance(v, acb):
        return v
    v = Fraction(v)
    return acb(v.numerator) / v.denominator


def nz_integrand(knot: str, x) -> Integrand:
    x = _acb(x)
    U = x.log()
    h2 = Fraction(1, 2)
    if knot == "4_1":
        # Phi_b(v) Phi_b(u+v) e^{-pi i (v^2 + 4uv)} -> exp(+(V^2 + 4UV)/(2h))
        return Integrand(1, [Dilog(1, (1,), acb(-1), label="Phi_b(v)"), Dilog(1, (1,), -x, label="Phi_b(u+v)")],
                         Quadratic([[acb(1)]], [2 * U], acb(0)), name="NZ 4_1 (GGM (113))")
    if knot == "5_2":
        # Phi_b(v) Phi_b(v+u) Phi_b(v-u) e^{-2 pi i v^2} -> exp(+V^2/h)
        return Integrand(1, [Dilog(1, (1,), acb(-1)), Dilog(1, (1,), -x), Dilog(1, (1,), -1 / x)],
                         Quadratic([[acb(2)]], [acb(0)], acb(0)), name="NZ 5_2 (GGM (230))")
    if knot == "6_1":
        # Phi_b(c_b - X) with X = s1 - s2 - 2u, s1 - s2 - u, s1, s2;  Phi_b(c_b - X): -e^{2 pi b (c_b - X)} = e^{h/2} e^{-X}
        # P = 2 s1 s2 - 2 s2^2 + 2 u s1 - 6 u s2
        dl = [Dilog(1, (-1, 1), x * x, h2), Dilog(1, (-1, 1), x, h2), Dilog(1, (-1, 0), acb(1), h2),
              Dilog(1, (0, -1), acb(1), h2)]
        Q = [[acb(0), acb(-1)], [acb(-1), acb(2)]]
        return Integrand(2, dl, Quadratic(Q, [-U, 3 * U], acb(0)), name="NZ 6_1 (int61)")
    if knot == "7_2":
        # Phi_b(c_b - s1 + 2u) Phi_b(c_b - s1 + u) Phi_b(c_b - s1) Phi_b(c_b - s2)
        # P = 4 c_b s1 - 2 s1^2 + 2 s1 s2 - s2^2/2 + 4 u s1 - 2 u s2
        # 4 c_b s1 -> pi i * 4 * i (b + 1/b)/2 * S1/(2 pi b) = -S1 - 2 pi i S1/h : linear 1/h term + O(1) prefactor e^{-S1}
        pi = acb.pi()
        dl = [Dilog(1, (-1, 0), x * x, h2), Dilog(1, (-1, 0), x, h2), Dilog(1, (-1, 0), acb(1), h2),
              Dilog(1, (0, -1), acb(1), h2)]
        Q = [[acb(2), acb(-1)], [acb(-1), acb(1) / 2]]
        L = [-2 * U - 2 * pi * acb(0, 1), U]
        return Integrand(2, dl, Quadratic(Q, L, acb(0)), prefactor=[(acb(1), (-1, 0))], name="NZ 7_2 (int72)")
    raise KeyError(f"no NZ integrand recorded for {knot}")


def _grad_exp(I: Integrand, Z):
    """exp of the gradient of S (branch free) and its Jacobian, for Newton."""
    d = I.d
    G = [acb(0)] * d
    H = [[acb(0)] * d for _ in range(d)]
    for t in I.dilogs:
        tt = _acb(t.t0)
        for i in range(d):
            if t.a[i]:
                tt *= (t.a[i] * Z[i]).exp()
        l1 = -(1 - tt).log()
        l0 = tt / (1 - tt)
        for i in range(d):
            if t.a[i]:
                G[i] += t.c * t.a[i] * l1
                for j in range(d):
                    if t.a[j]:
                        H[i][j] += t.c * t.a[i] * t.a[j] * l0
    for i in range(d):
        G[i] += _acb(I.quad.L[i])
        for j in range(d):
            G[i] += _acb(I.quad.Q[i][j]) * Z[j]
            H[i][j] += _acb(I.quad.Q[i][j])
    E = [g.exp() for g in G]
    F = [e - 1 for e in E]
    J = [[E[i] * H[i][j] for j in range(d)] for i in range(d)]
    return F, J


def critical_points(I: Integrand, n_start: int = 300, seed: int = 1, digits: int = 50) -> List[List[acb]]:
    """Distinct non-degenerate critical points (exponentiated gradient = 1), up to 2 pi i lattice shifts."""
    old = ctx.prec
    ctx.prec = int(digits * 3.4) + 32
    rng = random.Random(seed)
    sols: List[List[acb]] = []
    try:
        for _ in range(n_start):
            Z = [acb(rng.uniform(-2.5, 2.5), rng.uniform(-3.1, 3.1)) for _ in range(I.d)]
            ok = False
            for it in range(200):
                try:
                    F, J = _grad_exp(I, Z)
                    step = acb_mat(J).solve(acb_mat([[f] for f in F]))
                except Exception:
                    break
                Z = [(Z[i] - step[i, 0]).mid() for i in range(I.d)]
                if max(abs(step[i, 0]).mid() for i in range(I.d)) < arb(10) ** (-digits):
                    ok = True
                    break
                if max(abs(z.real).mid() for z in Z) > 30:
                    break
            if not ok:
                continue
            # discard degenerate / boundary points (t = 0 or 1 for some dilog)
            bad = False
            for t in I.dilogs:
                tt = _acb(t.t0)
                for i in range(I.d):
                    if t.a[i]:
                        tt *= (t.a[i] * Z[i]).exp()
                if abs(tt).mid() < 1e-20 or abs(1 - tt).mid() < 1e-20:
                    bad = True
            if bad:
                continue
            Y = [z.exp() for z in Z]
            if any(all(abs(Y[i] - s[i].exp()).mid() < 1e-20 for i in range(I.d)) for s in sols):
                continue
            sols.append(Z)
        return sols
    finally:
        ctx.prec = old


def volume(I: Integrand, Z) -> acb:
    """sum over dilog terms of c * D(t) (Bloch-Wigner); lift independent."""
    tot = arb(0)
    for t in I.dilogs:
        tt = _acb(t.t0)
        for i in range(I.d):
            if t.a[i]:
                tt *= (t.a[i] * Z[i]).exp()
        D = tt.polylog(2).imag + (1 - tt).arg() * abs(tt).log()
        tot += t.c * D
    return tot


def nz_series(knot: str, x, n: int, digits: int = 50):
    """[(Z, volume, SaddleResult)] for all critical points of the NZ integrand, sorted by volume."""
    old = ctx.prec
    ctx.prec = int(digits * 3.4) + 64 + 6 * n
    try:
        I = nz_integrand(knot, x)
        out = []
        for Z in critical_points(I, digits=digits):
            out.append((Z, volume(I, Z), expand(I, Z, n)))
        out.sort(key=lambda t: -float(t[1].mid()))
        return out
    finally:
        ctx.prec = old
