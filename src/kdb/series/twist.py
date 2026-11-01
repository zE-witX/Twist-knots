"""Non-trivial sectors of twist knots K_p from Marino's double contour integral (Marino 5.4, I_{K_p}(z,w)).

With Z = 2 pi b z, W = 2 pi b w, x = e^{2 pi b u} = e^U, A = e^Z, B = e^W, h = 2 pi i b^2, the integrand
    -Phi_b^{-1}(z - i/2b + u) Phi_b^{-1}(z - i/2b - u) Phi_b^{-1}(z - i/2b)
     Phi_b(z - w + ib/2 - i/2b) Phi_b(z + w + ib/2 - i/2b) e^{-2 pi i (p+1/2)(w + i/2b)^2}
     (e^{2 pi b (z+w)} - e^{2 pi b (z-w)}) tanh(pi z/b) tanh(pi w/b)
becomes, term by term (kdb.series.saddle conventions; -e^X = t0 e^{a.Z + alpha h}):
    Phi_b^{-1}(z - i/2b +- u):  c=-1, t0 = x^{+-1},  Li(x^{+-1} A)
    Phi_b^{-1}(z - i/2b):       c=-1, t0 = 1,        Li(A)
    Phi_b(z -+ w + ib/2 - i/2b): c=+1, t0 = 1, alpha = 1/2,  Li(A B^{-+1} e^{h/2})
    Gaussian:                   (1/h) (p+1/2)(W + i pi)^2
    prefactor:                  e^{Z+W} - e^{Z-W};  tanh factors -> constants (+-1).
Classical part S(Z,W) = -Li2(xA) - Li2(A/x) - Li2(A) + Li2(A/B) + Li2(AB) + (p+1/2)(W+i pi)^2, i.e. the
cyclotomic potential of the 6_1/7_2 notes up to sector-independent terms and 2 pi i Z-lattice shifts of W.
Critical equations (exponentiated): -B^{2p}(B - A) = 1 - AB,  (1-A)(1-xA)(1-A/x) = (1-AB)(1-A/B).
The map B -> 1/B preserves them; each sector appears twice and we keep one representative.

The fluctuation series a_n depend only on (A, B) (they involve Li_{s<=0} and the Hessian), so they are
branch independent; the action needs a choice of logarithms and is reported with principal branches,
mod 4 pi^2 (and in the cyclotomic normalisation W(A,B;x) of the notes).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence

from flint import acb, acb_poly, arb, ctx, fmpq, fmpq_poly

from .saddle import Dilog, Integrand, Quadratic, SaddleResult, expand, newton


def _acb(v):
    if isinstance(v, acb):
        return v
    v = Fraction(v)
    return acb(v.numerator) / v.denominator


def riley_degree(p: int) -> int:
    """Number of non-abelian parabolic sectors of K_p (degree of the Riley polynomial)."""
    if p == 0:
        return 0
    return 2 * p - 1 if p > 0 else 2 * (-p)


def integrand(p: int, x) -> Integrand:
    x = _acb(x)
    pi = acb.pi()
    i = acb(0, 1)
    half = Fraction(1, 2)
    dl = [
        Dilog(-1, (1, 0), x, label="Phi_b^-1(z-i/2b+u)"),
        Dilog(-1, (1, 0), 1 / x, label="Phi_b^-1(z-i/2b-u)"),
        Dilog(-1, (1, 0), acb(1), label="Phi_b^-1(z-i/2b)"),
        Dilog(+1, (1, -1), acb(1), half, label="Phi_b(z-w+ib/2-i/2b)"),
        Dilog(+1, (1, 1), acb(1), half, label="Phi_b(z+w+ib/2-i/2b)"),
    ]
    c = acb(2 * p + 1) / 2
    quad = Quadratic(Q=[[acb(0), acb(0)], [acb(0), 2 * c]], L=[acb(0), 2 * c * pi * i], c0=-c * pi * pi)
    pref = [(acb(1), (1, 1)), (acb(-1), (1, -1))]
    return Integrand(2, dl, quad, Fraction(0), pref, name=f"I_K{p}")


def _ND(p: int):
    """A = N(B)/D(B) from the w-equation."""
    if p > 0:
        N = acb_poly([1] + [0] * (2 * p) + [1])            # 1 + B^{2p+1}
        D = acb_poly([0, 1] + [0] * (2 * p - 2) + [1])     # B + B^{2p}
    else:
        q = -p
        N = acb_poly([0, 1] + [0] * (2 * q - 2) + [1])     # B + B^{2q}
        D = acb_poly([1] + [0] * (2 * q) + [1])            # 1 + B^{2q+1}
    return N, D


def critical_polynomial(p: int, x):
    """Exact polynomial in B (fmpq_poly, x rational) whose roots contain all critical B."""
    x = Fraction(x)
    xq = fmpq(x.numerator, x.denominator)
    if p > 0:
        N = fmpq_poly([1] + [0] * (2 * p) + [1])
        D = fmpq_poly([0, 1] + [0] * (2 * p - 2) + [1])
    else:
        q = -p
        N = fmpq_poly([0, 1] + [0] * (2 * q - 2) + [1])
        D = fmpq_poly([1] + [0] * (2 * q) + [1])
    Bp = fmpq_poly([0, 1])
    lhs = (D - N * xq) * (D * xq - N) * (D - N) * Bp
    rhs = D * xq * (D - N * Bp) * (Bp * D - N)
    return lhs - rhs


def critical_factors(p: int, x):
    """Irreducible factors of the critical polynomial other than B, B-1, B+1 (exact)."""
    P = critical_polynomial(p, x)
    _, facs = P.factor()
    out = []
    for f, e in facs:
        if f.degree() == 1:
            r = -f.coeffs()[0] / f.coeffs()[1]
            if r in (0, 1, -1):
                continue
        out.append(f)
    return out


@dataclass
class Sector:
    A: acb
    B: acb
    Z: List[acb]
    label: str = ""


def saddles(p: int, x, digits: int = 60) -> List[Sector]:
    """Non-abelian critical points (one per B <-> 1/B pair), refined by Newton."""
    old = ctx.prec
    ctx.prec = max(ctx.prec, int(digits * 3.4) + 64)
    try:
        xr = Fraction(x)
        roots = []
        for f in critical_factors(p, xr):
            fa = acb_poly([acb(int(c.p)) / int(c.q) for c in f.coeffs()])
            dfa = fa.derivative()
            for r in fa.roots():
                r = r.mid()
                for _ in range(200):  # Newton refinement of the isolating ball midpoint
                    step = (fa(r) / dfa(r)).mid()
                    r = (r - step).mid()
                    if abs(step).mid() < abs(r).mid() * arb(2) ** (-ctx.prec + 20):
                        break
                roots.append(r)
        x = _acb(xr)
        N, D = _ND(p)
        I = integrand(p, x)
        cands = []
        for B in roots:
            B = B.mid()
            if abs(B).mid() < 1e-20 or abs(B - 1).mid() < 1e-12 or abs(B + 1).mid() < 1e-12:
                continue
            Dv = D(B)
            if abs(Dv).mid() < 1e-25:
                continue
            A = (N(B) / Dv).mid()
            if abs(A).mid() < 1e-20 or abs(A - 1).mid() < 1e-12:
                continue
            r1 = abs(-(B ** (2 * p)) * (B - A) - (1 - A * B)).mid() if p >= 0 else \
                abs(-(B - A) - (1 - A * B) * B ** (-2 * p)).mid()
            r2 = abs((1 - A) * (1 - x * A) * (1 - A / x) - (1 - A * B) * (1 - A / B)).mid()
            if r1 > 1e-15 or r2 > 1e-15:
                continue
            cands.append((A, B))
        # pair B with 1/B
        reps = []
        used = [False] * len(cands)
        for i, (A, B) in enumerate(cands):
            if used[i]:
                continue
            used[i] = True
            for j in range(i + 1, len(cands)):
                if not used[j] and abs(cands[j][1] * B - 1).mid() < 1e-12 and abs(cands[j][0] - A).mid() < 1e-12:
                    used[j] = True
                    break
            if abs(B).mid() < 1:
                B = 1 / B
            reps.append((A, B))
        out = []
        for A, B in reps:
            # (A, B) solve the exponentiated critical equations to working precision; in log form the gradient
            # vanishes only modulo 2 pi i Z (choice of lifts), which does not affect the fluctuation series.
            out.append(Sector(A, B, [A.log(), B.log()]))
        if len(out) != riley_degree(p):
            raise RuntimeError(f"K_{p}: found {len(out)} sectors, expected {riley_degree(p)} (x={x})")
        return out
    finally:
        ctx.prec = old


def cyclotomic_action(p: int, x, A: acb, B: acb) -> acb:
    """W(A,B;x) of the 6_1/7_2 notes (principal logarithms):
    (p+1/2) log^2 B + i pi log B - Li2(A) + Li2(AB) + Li2(A/B) - pi^2/6 + Li2(x) + Li2(1/x) - Li2(xA) - Li2(A/x)."""
    x = _acb(x)
    pi = acb.pi()
    lB = B.log()
    L = lambda t: t.polylog(2)
    return ((acb(2 * p + 1) / 2) * lB * lB + acb(0, 1) * pi * lB - L(A) + L(A * B) + L(A / B) - pi * pi / 6
            + L(x) + L(1 / x) - L(x * A) - L(A / x))


def series(p: int, x, n: int, digits: int = 60):
    """List of (Sector, SaddleResult) with the fluctuation series to h^n."""
    old = ctx.prec
    ctx.prec = int(digits * 3.4) + 64 + 6 * n
    try:
        I = integrand(p, x)
        res = []
        for s in saddles(p, x, digits):
            r = expand(I, s.Z, n)
            res.append((s, r))
        return res
    finally:
        ctx.prec = old
