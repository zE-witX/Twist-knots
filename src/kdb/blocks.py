"""Holomorphic blocks of 4_1 as truncated q-series (GGMW (94)-(112)).

All functions take a *coefficient ring* ``R`` (see :class:`Ring`) that fixes how the Jacobi
variable x is represented: exactly as a rational function (``Ring.symbolic()``), or as an exact
rational number (``Ring.at(Fraction(2))``).  q-series are :class:`kdb.algebra.TSeries` in q
(for |q| < 1) or in p = 1/q (for |q| > 1), truncated at order N.

Plain blocks (GGMW (94)):
    C_m = sum_k (-1)^k q^{k(k+1)/2+km} / ((x^-1;q)_{k+1} (x;q)_{k+1})
    A_m = sum_k (-1)^k q^{k(k+1)/2+km} x^{k+m} / ((q;q)_k (x^2 q;q)_k),   B_m(x) = A_m(1/x)
Decorations (GGMW (107)-(109) with the squared theta, see CLAUDE.md 3.3):
    alpha = (q x^2;q)_inf / [(q x;q)_inf (x^-1;q)_inf]^2
    beta  = x (q x^-2;q)_inf / [(q x^-1;q)_inf (x;q)_inf]^2
For |q| > 1 we use (a;q)_inf := 1/(a/q;1/q)_inf and the inversion GGMW (98)-(99):
    C_m(x,1/p) = C_{-m}(x,p),  A_m(x,1/p) = B_{-m}(x,p),  B_m(x,1/p) = A_{-m}(x,p).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any

from .algebra import RatFun, TSeries


@dataclass(frozen=True)
class Ring:
    """Coefficient ring for x: symbolic (RatFun in x) or a rational number."""

    x: Any
    one: Any
    zero: Any
    name: str

    @staticmethod
    def symbolic() -> "Ring":
        return Ring(RatFun.x(), RatFun(1), RatFun(0), "symbolic")

    @staticmethod
    def at(value) -> "Ring":
        v = Fraction(value)
        return Ring(v, Fraction(1), Fraction(0), f"x={v}")

    def c(self, a):
        """Embed a constant (int / Fraction / ring element)."""
        if isinstance(a, (int, Fraction)):
            return self.one * a
        return a

    def xp(self, e: int):
        return self.x ** e if e >= 0 else (self.one / self.x) ** (-e)


def _S(R: Ring, coeffs, N, val=0):
    return TSeries(coeffs, N, val, R.zero)


def one(R: Ring, N):
    return _S(R, [R.one], N)


def poch(R: Ring, a, s: int, n: int, N: int) -> TSeries:
    """prod_{j=0}^{n-1} (1 - a q^{s+j}),  a in R, s >= 0."""
    r = one(R, N)
    for j in range(n):
        e = s + j
        if e > N:
            break  # factor is 1 + O(q^{N+1})
        f = [R.one] + [R.zero] * N
        f[e] = f[e] - a
        r = r * _S(R, f, N)
    return r


def poch_inf(R: Ring, a, s: int, N: int) -> TSeries:
    """(a q^s; q)_inf truncated."""
    return poch(R, a, s, N + 1 - s if s <= N else 0, N)


# ----------------------------------------------------------------------------------------------
# plain blocks, |q| < 1
# ----------------------------------------------------------------------------------------------
def _k_range(m: int, N: int):
    """k with k(k+1)/2 + k m <= N (exponents are >= 0 for m >= -1)."""
    k = 0
    while True:
        e = k * (k + 1) // 2 + k * m
        if e < 0:
            raise ValueError(f"block index m={m} gives negative q-exponents; use the q-difference relation")
        if e > N and k > max(0, -m):
            return
        if e <= N:
            yield k, e
        k += 1


def C_block(R: Ring, m: int, N: int) -> TSeries:
    xi = R.one / R.x
    s = _S(R, [], N)
    for k, e in _k_range(m, N):
        den = poch(R, xi, 0, k + 1, N) * poch(R, R.x, 0, k + 1, N)
        term = den.inverse() * R.c((-1) ** k)
        s = s + _shift(term, e, R, N)
    return s


def A_block(R: Ring, m: int, N: int, x=None) -> TSeries:
    x = R.x if x is None else x
    s = _S(R, [], N)
    for k, e in _k_range(m, N):
        den = poch(R, R.one, 1, k, N) * poch(R, x * x, 1, k, N)
        xp = x ** (k + m) if k + m >= 0 else (R.one / x) ** (-(k + m))
        term = den.inverse() * (xp * R.c((-1) ** k))
        s = s + _shift(term, e, R, N)
    return s


def B_block(R: Ring, m: int, N: int) -> TSeries:
    return A_block(R, m, N, x=R.one / R.x)


def _shift(s: TSeries, e: int, R: Ring, N: int) -> TSeries:
    return TSeries([R.zero] * e + s.c, N, 0, R.zero) if e > 0 else s.truncate(N)


# ----------------------------------------------------------------------------------------------
# decorations
# ----------------------------------------------------------------------------------------------
def decorations(R: Ring, N: int, region: str = "inside"):
    """(alpha, beta) as series in q ('inside', |q|<1) or in p = 1/q ('outside', |q|>1)."""
    x, xi = R.x, R.one / R.x
    if region == "inside":
        th_p = poch_inf(R, x, 1, N) * poch_inf(R, xi, 0, N)       # theta(-q^{1/2} x, q)
        th_m = poch_inf(R, xi, 1, N) * poch_inf(R, x, 0, N)       # theta(-q^{1/2} x^-1, q)
        alpha = poch_inf(R, x * x, 1, N) / (th_p * th_p)
        beta = poch_inf(R, xi * xi, 1, N) / (th_m * th_m) * x
    else:
        # (a;q)_inf = 1/(a p; p)_inf with p = 1/q
        alpha = (poch_inf(R, x, 0, N) ** 2 * poch_inf(R, xi, 1, N) ** 2) / poch_inf(R, x * x, 0, N)
        beta = (poch_inf(R, xi, 0, N) ** 2 * poch_inf(R, x, 1, N) ** 2) / poch_inf(R, xi * xi, 0, N) * x
    return alpha, beta


# ----------------------------------------------------------------------------------------------
# J matrices
# ----------------------------------------------------------------------------------------------
class Blocks:
    """Cache of blocks for a ring R and truncation N."""

    def __init__(self, R: Ring, N: int):
        self.R, self.N = R, N
        self._c = {}

    def _get(self, key, f):
        if key not in self._c:
            self._c[key] = f()
        return self._c[key]

    def C(self, m):
        return self._get(("C", m), lambda: C_block(self.R, m, self.N))

    def A(self, m):
        return self._get(("A", m), lambda: A_block(self.R, m, self.N))

    def B(self, m):
        return self._get(("B", m), lambda: B_block(self.R, m, self.N))

    def deco(self, region="inside"):
        return self._get(("deco", region), lambda: decorations(self.R, self.N, region))

    # decorated blocks in the region |q|<1 (series in q)
    def calA(self, m):
        return self._get(("calA", m), lambda: self.deco()[0] * self.A(m))

    def calB(self, m):
        return self._get(("calB", m), lambda: self.deco()[1] * self.B(m))

    def J(self, m, decorated=True, region="inside"):
        """3x3 matrix calJ_m(x,q) (decorated) or J_m; region 'outside' = |q|>1, series in p=1/q."""
        R, N = self.R, self.N
        z = _S(R, [], N)
        o = one(R, N)
        if region == "inside":
            C0, C1 = self.C(m), self.C(m + 1)
            A0, A1, B0, B1 = self.A(m), self.A(m + 1), self.B(m), self.B(m + 1)
        else:
            C0, C1 = self.C(-m), self.C(-m - 1)
            A0, A1 = self.B(-m), self.B(-m - 1)
            B0, B1 = self.A(-m), self.A(-m - 1)
        if decorated:
            al, be = self.deco(region)
            A0, A1, B0, B1 = al * A0, al * A1, be * B0, be * B1
        return [[o, C0, C1], [z, A0, A1], [z, B0, B1]]
