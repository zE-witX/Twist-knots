"""Generic formal saddle-point (Gaussian) expansion of state integrals built from Faddeev dilogarithms.

An integrand is described, after the rescaling Z = 2 pi b z (variables Z in C^d), by
    exp( sum_terms ) * g(Z),        h = 2 pi i b^2,
with terms of the following kinds (each a small dataclass below):

* ``Dilog(c, a, t0, alpha)``: c * log Phi_b( (a.Z + log t0' ...)/(2 pi b) ) in the small-b expansion
  (Andersen-Kashaev, quoted in Marino App. A, (as-fd)):
        log Phi_b(X/(2 pi b)) ~ sum_{k>=0} h^{2k-1} B_{2k}(1/2)/(2k)! Li_{2-2k}(-e^X),
  where the argument has been written so that  -e^X = t0 * exp(a.Z + alpha*h).
  (A shift i m b/2 of the Phi_b argument gives alpha = m/2; a shift by i/(2b) flips the sign of t0.)
* ``Quadratic(Q, L, c0)``: (1/h) (Z^T Q Z / 2 + L.Z + c0), from Gaussians e^{-pi i s^T Q s ...}.
* ``Const(c_h)``: an explicit factor exp(c_h * h) (e.g. regular factors q^{c}).
* prefactor g(Z) = sum_i c_i exp(b_i . Z): O(1) factors such as e^{2 pi b (z+w)} - e^{2 pi b (z-w)}.

At a non-degenerate critical point Z* of the classical part S(Z) (the 1/h terms), with Z = Z* + sqrt(h) zeta,
    integrand = exp(S(Z*)/h) * exp(E_0) * g(Z*) * exp(zeta^T H zeta / 2) * (1 + sum_{j>=1} p_j(zeta) h^{j/2})
and the formal Gaussian average gives  phi(h) = sum_n a_n h^n,  a_0 = 1,  with
    <zeta^alpha>  computed from the covariance Sigma = -H^{-1} (Isserlis/Wick).
The full one-saddle contribution is  (2 pi h)^{d/2} det(-H)^{-1/2} e^{S(Z*)/h} e^{E_0} g(Z*) phi(h)
(up to the measure normalisation of the integral, which is convention dependent and recorded separately).

Numerics: python-flint acb (ball radii are dropped after each step: working-precision semantics;
use a precision repeat to validate digits).  Multivariate polynomials in zeta use Kronecker substitution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import factorial
from typing import Dict, List, Sequence, Tuple

from flint import acb, acb_mat, acb_poly, ctx

from .gauss41 import bernoulli_half, li_minus, stirling_row


# ----------------------------------------------------------------------------------------------
# integrand description
# ----------------------------------------------------------------------------------------------
@dataclass
class Dilog:
    c: int                      # +1 for Phi_b in the numerator, -1 for the denominator
    a: Tuple[int, ...]          # linear form in Z
    t0: object                  # complex constant: -e^X = t0 * exp(a.Z + alpha h)
    alpha: Fraction = Fraction(0)
    label: str = ""


@dataclass
class Quadratic:
    Q: List[List[object]]       # symmetric matrix: (1/h) * Z^T Q Z / 2
    L: List[object]             # (1/h) * L.Z
    c0: object = 0              # (1/h) * c0


@dataclass
class Integrand:
    d: int
    dilogs: List[Dilog]
    quad: Quadratic
    const_h: Fraction = Fraction(0)                       # exp(const_h * h)
    prefactor: List[Tuple[object, Tuple[int, ...]]] = field(default_factory=lambda: [])  # sum c_i e^{b_i.Z}
    name: str = ""


def _acb(v) -> acb:
    if isinstance(v, acb):
        return v
    if isinstance(v, Fraction):
        return acb(v.numerator) / v.denominator
    if isinstance(v, complex):
        return acb(v.real, v.imag)
    return acb(v)


def _dot(a, Z):
    s = acb(0)
    for ai, zi in zip(a, Z):
        if ai:
            s += ai * zi
    return s


def li_s(s: int, t: acb) -> acb:
    """Li_s(t) for integer s <= 2 (s <= 0 rational, s = 1, 2 via flint)."""
    if s <= 0:
        return li_minus(-s, t)
    if s == 1:
        return -(1 - t).log()
    return t.polylog(s)


# ----------------------------------------------------------------------------------------------
# classical data
# ----------------------------------------------------------------------------------------------
def classical(I: Integrand, Z: Sequence[acb]):
    """S(Z), grad S, Hessian H at Z (S = coefficient of 1/h)."""
    d = I.d
    S = acb(0)
    G = [acb(0)] * d
    H = [[acb(0)] * d for _ in range(d)]
    for t in I.dilogs:
        tt = _acb(t.t0) * _dot(t.a, Z).exp()
        S += t.c * li_s(2, tt)
        l1 = li_s(1, tt)
        l0 = li_s(0, tt)
        for i in range(d):
            if t.a[i]:
                G[i] += t.c * t.a[i] * l1
                for j in range(d):
                    if t.a[j]:
                        H[i][j] += t.c * t.a[i] * t.a[j] * l0
    Q, L = I.quad.Q, I.quad.L
    for i in range(d):
        S += _acb(L[i]) * Z[i]
        G[i] += _acb(L[i])
        for j in range(d):
            S += _acb(Q[i][j]) * Z[i] * Z[j] / 2
            G[i] += _acb(Q[i][j]) * Z[j]
            H[i][j] += _acb(Q[i][j])
    S += _acb(I.quad.c0)
    return S, G, H


def newton(I: Integrand, Z0: Sequence[acb], digits: int = 60, maxit: int = 200) -> List[acb]:
    """Refine a critical point of S by Newton's method at the current precision."""
    Z = [_acb(z) for z in Z0]
    d = I.d
    tol = acb(10) ** (-digits)
    for _ in range(maxit):
        _, G, H = classical(I, Z)
        M = acb_mat(H)
        step = M.solve(acb_mat([[g] for g in G]))
        Z = [(Z[i] - step[i, 0]).mid() for i in range(d)]
        if max(abs(step[i, 0]).mid() for i in range(d)) < abs(tol).mid():
            break
    return Z


# ----------------------------------------------------------------------------------------------
# multivariate polynomials via Kronecker substitution
# ----------------------------------------------------------------------------------------------
class Kron:
    def __init__(self, d: int, maxdeg: int):
        self.d, self.D = d, maxdeg + 1

    def index(self, alpha: Sequence[int]) -> int:
        k, m = 0, 1
        for a in alpha:
            k += a * m
            m *= self.D
        return k

    def exps(self, k: int) -> Tuple[int, ...]:
        out = []
        for _ in range(self.d):
            out.append(k % self.D)
            k //= self.D
        return tuple(out)

    def linear_power(self, a: Sequence[int], m: int) -> Dict[Tuple[int, ...], int]:
        """(a.zeta)^m as {alpha: integer coefficient}."""
        poly = {tuple([0] * self.d): 1}
        for _ in range(m):
            new = {}
            for al, c in poly.items():
                for i, ai in enumerate(a):
                    if ai:
                        b = list(al)
                        b[i] += 1
                        b = tuple(b)
                        new[b] = new.get(b, 0) + c * ai
            poly = new
        return poly

    def to_poly(self, terms: Dict[Tuple[int, ...], acb]) -> acb_poly:
        if not terms:
            return acb_poly([0])
        n = max(self.index(al) for al in terms) + 1
        cs = [acb(0)] * n
        for al, c in terms.items():
            cs[self.index(al)] += c
        return acb_poly(cs)


def moments(Sigma: List[List[acb]], maxdeg: int):
    """E[zeta^alpha] for a centred Gaussian with covariance Sigma, all |alpha| <= maxdeg (even)."""
    d = len(Sigma)
    memo = {tuple([0] * d): acb(1)}

    def E(al):
        if al in memo:
            return memo[al]
        if sum(al) % 2:
            memo[al] = acb(0)
            return memo[al]
        i = next(k for k, a in enumerate(al) if a)
        beta = list(al)
        beta[i] -= 1
        acc = acb(0)
        for j in range(d):
            if beta[j]:
                g = list(beta)
                g[j] -= 1
                acc += Sigma[i][j] * beta[j] * E(tuple(g))
        memo[al] = acc.mid()
        return memo[al]

    return E


# ----------------------------------------------------------------------------------------------
# the expansion
# ----------------------------------------------------------------------------------------------
@dataclass
class SaddleResult:
    Z: List[acb]
    S: acb                 # classical action S(Z*) (coefficient of 1/h)
    H: List[List[acb]]     # Hessian of S at Z*
    detH: acb
    E0: acb                # O(h^0) constant of the exponent
    g: acb                 # prefactor at Z*
    a: List[acb]           # a_0..a_n, a_0 = 1  (series in h)


def expand(I: Integrand, Z: Sequence[acb], n: int) -> SaddleResult:
    """Formal Gaussian expansion to h^n at the critical point Z (already refined)."""
    d = I.d
    Z = [_acb(z) for z in Z]
    S, G, H = classical(I, Z)
    maxdeg = 6 * n + 2
    K = Kron(d, maxdeg)
    # p_j terms: dict j -> {alpha: coeff}
    P: Dict[int, Dict[Tuple[int, ...], acb]] = {j: {} for j in range(1, 2 * n + 1)}
    E0 = acb(0)

    def add(j, al, c):
        dct = P[j]
        dct[al] = dct.get(al, acb(0)) + c

    zero_al = tuple([0] * d)
    for t in I.dilogs:
        tt = _acb(t.t0) * _dot(t.a, Z).exp()
        alpha = _acb(t.alpha)
        # power of eps = sqrt(h): 4k - 2 + 2j + m ; Li_{2-2k-j-m}
        for k in range(0, n + 2):
            bk = bernoulli_half(2 * k) / factorial(2 * k)
            bk = acb(bk.numerator) / bk.denominator
            for jj in range(0, n + 2):
                for m in range(0, 2 * n + 3):
                    pw = 4 * k - 2 + 2 * jj + m
                    if pw > 2 * n:
                        break
                    if pw < 0 or (pw == -1):
                        continue
                    if k == 0 and jj == 0 and m <= 2:
                        continue  # classical, gradient (vanishes), Hessian (Gaussian weight)
                    coef = t.c * bk * alpha ** jj / factorial(jj) / factorial(m) * li_s(2 - 2 * k - jj - m, tt)
                    if pw == 0:
                        if m == 0:
                            E0 += coef
                        else:
                            raise AssertionError("unexpected eps^0 term")
                        continue
                    for al, ci in K.linear_power(t.a, m).items():
                        add(pw, al, coef * ci)
    # quadratic polynomial contributes only to S, gradient and Hessian.
    if I.const_h:
        add(2, zero_al, _acb(Fraction(I.const_h)))
    # prefactor ratio g(Z* + eps zeta)/g(Z*) = sum_m eps^m Gm(zeta)
    g0 = acb(0)
    Gm: Dict[int, Dict[Tuple[int, ...], acb]] = {m: {} for m in range(1, 2 * n + 1)}
    for c, b in I.prefactor:
        w = _acb(c) * _dot(b, Z).exp()
        g0 += w
        for m in range(1, 2 * n + 1):
            for al, ci in K.linear_power(b, m).items():
                Gm[m][al] = Gm[m].get(al, acb(0)) + w * ci / factorial(m)
    if not I.prefactor:
        g0 = acb(1)
    for m in Gm:
        for al in Gm[m]:
            Gm[m][al] = Gm[m][al] / g0
    # exponentiate
    ps = [K.to_poly({al: c.mid() for al, c in P[j].items()}) for j in range(1, 2 * n + 1)]
    es = [acb_poly([1])]
    for j in range(1, 2 * n + 1):
        acc = acb_poly([0])
        for k in range(1, j + 1):
            acc += ps[k - 1] * es[j - k] * k
        es.append(acb_poly([c.mid() for c in (acc * (acb(1) / j)).coeffs()]))
    if I.prefactor:
        gs = [acb_poly([1])] + [K.to_poly(Gm[m]) for m in range(1, 2 * n + 1)]
        es2 = []
        for j in range(0, 2 * n + 1):
            acc = acb_poly([0])
            for m in range(0, j + 1):
                acc += gs[m] * es[j - m]
            es2.append(acb_poly([c.mid() for c in acc.coeffs()]))
        es = es2
    # Gaussian average
    Hm = acb_mat(H)
    Sig = -(Hm.inv())
    Sigma = [[Sig[i, j].mid() for j in range(d)] for i in range(d)]
    E = moments(Sigma, maxdeg)
    a = []
    for k in range(n + 1):
        cs = es[2 * k].coeffs()
        acc = acb(0)
        for idx, c in enumerate(cs):
            if c == 0:
                continue
            acc += c * E(K.exps(idx))
        a.append(acc.mid())
    return SaddleResult(Z=Z, S=S, H=H, detH=Hm.det(), E0=E0, g=g0, a=a)


def log_series(a: Sequence[acb]) -> List[acb]:
    """Coefficients of log(sum a_n h^n) (a_0 = 1)."""
    n = len(a) - 1
    L = [acb(0)] * (n + 1)
    for k in range(1, n + 1):
        acc = k * a[k]
        for j in range(1, k):
            acc -= j * L[j] * a[k - j]
        L[k] = acc / k
    return L
