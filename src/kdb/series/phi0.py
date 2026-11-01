"""The trivial-connection series phi_0(x, h) of twist knots K_p.

Routes (porting guides ``phi0_colored_jones`` and ``phi0_state_integral``):

(i)  colored Jones / Habiro (MMR).  Masbaum's cyclotomic coefficients (Marino 5.4, CLAUDE.md 3.0)
         C^p_n(q) = -q^n sum_k (-1)^k q^{p k(k+1) + k(k-1)/2} (q^{2k+1}-1) (q;q)_n / ((q;q)_{n+k+1} (q;q)_{n-k})
     and J = sum_n C^p_n(q) (xq;q)_n (q/x;q)_n.  With eps = 2 - x - 1/x, s_j = 2cosh(jh) - 2,
     (xq;q)_n (q/x;q)_n = q^{n(n+1)/2} prod_{j=1}^n (eps + s_j), and
         phi_0(x;h) = sum_m P_m(eps)/Delta^{2m+1} h^m,   Delta = 1 - p eps = 1 + p z^2,  z = x^{1/2}-x^{-1/2}.
     * ``mmr_eps``      : bivariate (h, eps) expansion, P_m read off and verified (fast, exact);
     * ``mmr_notebook`` : the notebook algorithm (Dq coefficient extraction, Lagrange interpolation in n,
                          resummation in t = log x); used as a cross-check at low order.
(ii) residue (Mellin-Barnes) route of the state integral: Marino 3.4 at x = 1 (``marino_x1``) and the
     fast x-dependent routine ``Phi0x`` of asymp-x.nb (``phi0x_residue``), 4_1 only.
(iii) formal solution of an inhomogeneous q-difference equation (``qdiff_formal``).
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import comb, factorial
from typing import Dict, List, Sequence, Tuple

from flint import fmpq, fmpq_poly, fmpz_poly

# ----------------------------------------------------------------------------------------------
# Masbaum coefficients
# ----------------------------------------------------------------------------------------------
_q = fmpz_poly([0, 1])


@lru_cache(maxsize=None)
def _qpoch(k: int) -> fmpz_poly:
    r = fmpz_poly([1])
    for j in range(1, k + 1):
        r *= (1 - _q ** j)
    return r


@lru_cache(maxsize=None)
def masbaum_C(p: int, n: int) -> Tuple[fmpz_poly, int]:
    """C^p_n(q) as (poly, shift): C = poly(q) * q^shift  (a Laurent polynomial)."""
    D = _qpoch(2 * n + 1) * _qpoch(n)
    terms = []
    for k in range(n + 1):
        e = n + p * k * (k + 1) + k * (k - 1) // 2
        num = -((-1) ** k) * (_q ** (2 * k + 1) - 1) * _qpoch(n)
        den = _qpoch(n + k + 1) * _qpoch(n - k)
        f, r = divmod(num * D, den)
        assert r == 0
        terms.append((e, f))
    emin = min(e for e, _ in terms)
    tot = fmpz_poly(0)
    for e, f in terms:
        tot += f * _q ** (e - emin)
    f, r = divmod(tot, D)
    if r != 0:
        raise ArithmeticError(f"C^{p}_{n} is not a Laurent polynomial")
    # normalise shift (remove factors of q)
    cs = f.coeffs()
    lo = next(i for i, c in enumerate(cs) if c != 0)
    return fmpz_poly(cs[lo:]), emin + lo


def laurent_dict(poly: fmpz_poly, shift: int) -> Dict[int, int]:
    return {i + shift: int(c) for i, c in enumerate(poly.coeffs()) if c != 0}


def h_expand(d: Dict[int, int], L: int) -> List[Fraction]:
    """Coefficients h^0..h^L of sum_j c_j e^{j h}  (notebook Dq)."""
    out = []
    for l in range(L + 1):
        s = sum(c * j ** l for j, c in d.items())
        out.append(Fraction(s, factorial(l)))
    return out


def colored_jones(p: int, N: int) -> Dict[int, int]:
    """J_{K_p,N}(q) as a Laurent polynomial {exponent: coeff} (normalised J_unknot = 1)."""
    tot: Dict[int, int] = {}
    for n in range(N):
        c, s = masbaum_C(p, n)
        f = c
        # (q^{1+N};q)_n (q^{1-N};q)_n
        for j in range(1, n + 1):
            f = f * (1 - _q ** (N + j))
        g = fmpz_poly([1])
        sh = 0
        for j in range(1, n + 1):
            # 1 - q^{j-N} = q^{j-N}(q^{N-j} - 1)
            g = g * (_q ** (N - j) - 1)
            sh += j - N
        for e, v in laurent_dict(f * g, s + sh).items():
            tot[e] = tot.get(e, 0) + v
    return {e: v for e, v in tot.items() if v}


def phi0_x1(p: int, L: int) -> List[Fraction]:
    """phi_0(1; h) = sum_n C_n (q;q)_n^2 at q = e^h, coefficients h^0..h^L (exact)."""
    out = [Fraction(0)] * (L + 1)
    for n in range(L // 2 + 1):
        c, s = masbaum_C(p, n)
        d = laurent_dict(c * _qpoch(n) ** 2, s)
        for l, v in enumerate(h_expand(d, L)):
            out[l] += v
    return out


# ----------------------------------------------------------------------------------------------
# route (i), eps-form
# ----------------------------------------------------------------------------------------------
def mmr_eps(p: int, M: int, extra: int = 6) -> List[fmpq_poly]:
    """P_0..P_M in Q[eps] with phi_0 = sum_m P_m(eps) h^m / (1 - p eps)^{2m+1}.

    Exact.  The eps-expansion is carried to order 2M + extra; the coefficients of P_m Delta^{-2m-1}
    beyond degree 2m must vanish after multiplying by Delta^{2m+1} (asserted).
    """
    A = 2 * M + extra                     # eps-degree kept
    # term n contributes to eps^a h^m only if n <= a + m/2 (since each (eps + s_j) with s_j = O(h^2))
    nmax = A + M // 2 + 1
    # bivariate series: list over h-degree of fmpq_poly in eps (truncated deg <= A)
    total = [fmpq_poly(0) for _ in range(M + 1)]

    def trunc(f):
        cs = f.coeffs()[: A + 1]
        return fmpq_poly(cs)

    # running product prod_{j<=n} (eps + s_j), as list over h-degree
    prod = [fmpq_poly([1])] + [fmpq_poly(0) for _ in range(M)]
    eps = fmpq_poly([0, 1])
    for n in range(0, nmax + 1):
        if n >= 1:
            # s_n = sum_{k>=1} 2 (n h)^{2k}/(2k)!
            s = [Fraction(0)] * (M + 1)
            for k in range(1, M // 2 + 1):
                s[2 * k] = Fraction(2 * n ** (2 * k), factorial(2 * k))
            new = [trunc(eps * prod[m]) for m in range(M + 1)]
            for m in range(M + 1):
                for k2 in range(2, m + 1, 2):
                    if s[k2]:
                        new[m] += prod[m - k2] * fmpq(s[k2].numerator, s[k2].denominator)
            prod = [trunc(f) for f in new]
        c, sh = masbaum_C(p, n)
        H = laurent_dict(c, sh + n * (n + 1) // 2)       # H_n = C_n q^{n(n+1)/2}
        hc = h_expand(H, M)
        for m in range(M + 1):
            acc = fmpq_poly(0)
            for k in range(m + 1):
                if hc[k]:
                    acc += prod[m - k] * fmpq(hc[k].numerator, hc[k].denominator)
            total[m] += acc
    out = []
    Delta = fmpq_poly([1, -p])
    for m in range(M + 1):
        f = trunc(total[m] * Delta ** (2 * m + 1))
        cs = f.coeffs()
        tail = cs[2 * m + 1:]
        if any(c != 0 for c in tail):
            raise AssertionError(f"MMR form violated at h^{m}: P_m has degree > 2m")
        out.append(fmpq_poly(cs[: 2 * m + 1]))
    return out


def minimal_denominator(p: int, Pm: fmpq_poly, m: int) -> Tuple[fmpq_poly, int]:
    """Cancel common factors (1 - p eps) between P_m and Delta^{2m+1}: returns (P_m', e) with R_m = P_m'/Delta^e."""
    e = 2 * m + 1
    if p == 0:
        return Pm, 0
    D = fmpq_poly([1, -p])
    while e > 0 and Pm != 0:
        q_, r = divmod(Pm, D)
        if r != 0:
            break
        Pm, e = q_, e - 1
    return Pm, e


def eps_to_x_laurent(P: fmpq_poly) -> Dict[int, Fraction]:
    """P(eps) with eps = 2 - x - 1/x, as a Laurent polynomial in x."""
    out: Dict[int, Fraction] = {}
    base = {0: Fraction(2), 1: Fraction(-1), -1: Fraction(-1)}
    pw = {0: Fraction(1)}
    for c in P.coeffs():
        c = Fraction(int(c.p), int(c.q))
        for e, v in pw.items():
            out[e] = out.get(e, 0) + c * v
        nw: Dict[int, Fraction] = {}
        for e1, v1 in pw.items():
            for e2, v2 in base.items():
                nw[e1 + e2] = nw.get(e1 + e2, 0) + v1 * v2
        pw = nw
    return {e: v for e, v in out.items() if v != 0}


# ----------------------------------------------------------------------------------------------
# route (i), notebook form (Dq + Lagrange in n + resummation)
# ----------------------------------------------------------------------------------------------
def hpoly(p: int, L: int) -> List[List[Fraction]]:
    """H_l(n): coefficient of h^l of J_{K_p,n}(e^h), as polynomials in n (coefficient lists), l = 0..L.

    Computed as in the notebook: values at n = 1..L+2 (deg_n H_l <= l) and Lagrange interpolation,
    with one extra value as a sanity check.
    """
    npts = L + 2
    vals = {}
    for n in range(1, npts + 2):
        vals[n] = h_expand(colored_jones(p, n), L)
    out = []
    for l in range(L + 1):
        pts = [(n, vals[n][l]) for n in range(1, npts + 1)]
        poly = _lagrange(pts)
        chk = sum(c * (npts + 1) ** i for i, c in enumerate(poly))
        assert chk == vals[npts + 1][l], f"Lagrange check failed at l={l}"
        while poly and poly[-1] == 0:
            poly.pop()
        out.append(poly)
    return out


def _lagrange(pts):
    n = len(pts)
    res = [Fraction(0)] * n
    for j, (xj, yj) in enumerate(pts):
        num = [Fraction(1)]
        den = Fraction(1)
        for m, (xm, _) in enumerate(pts):
            if m == j:
                continue
            num = [Fraction(0)] + num
            for i in range(len(num) - 1):
                num[i] -= xm * num[i + 1]
            den *= (xj - xm)
        for i, c in enumerate(num):
            res[i] += yj * c / den
    return res


def mmr_notebook(p: int, dmax: int, L: int | None = None) -> List[List[Fraction]]:
    """Resummation of hpoly (notebook step 3) -> phi_0 coefficients h^0..h^dmax as power series in t = log x
    (coefficient lists, truncated at t^{L - m}); compare with mmr_eps after t-expansion."""
    L = L if L is not None else 4 * dmax + 2
    H = hpoly(p, L)
    out = []
    for m in range(dmax + 1):
        # coefficient of h^m in sum_l H_l(t/h) h^l = sum_l sum_j c_{l,j} t^j h^{l-j}: need l - j = m
        ser = [Fraction(0)] * (L - m + 1)
        for l in range(m, L + 1):
            j = l - m
            if j < len(H[l]):
                ser[j] += H[l][j]
        out.append(ser)
    return out


def R_to_t_series(Pm: fmpq_poly, p: int, m: int, T: int) -> List[Fraction]:
    """P_m(eps)/Delta^{2m+1} expanded in t = log x (eps = 2 - 2cosh t) to t^T."""
    # eps(t) = -(t^2 + t^4/12 + ...) = -sum_{k>=1} 2 t^{2k}/(2k)!
    eps = [Fraction(0)] * (T + 1)
    for k in range(1, T // 2 + 1):
        eps[2 * k] = Fraction(-2, factorial(2 * k))

    def mul(a, b):
        r = [Fraction(0)] * (T + 1)
        for i, u in enumerate(a):
            if u:
                for j, v in enumerate(b[: T + 1 - i]):
                    if v:
                        r[i + j] += u * v
        return r

    def compose(poly_coeffs):
        r = [Fraction(0)] * (T + 1)
        pw = [Fraction(1)] + [Fraction(0)] * T
        for c in poly_coeffs:
            for i in range(T + 1):
                r[i] += c * pw[i]
            pw = mul(pw, eps)
        return r

    num = compose([Fraction(int(c.p), int(c.q)) for c in Pm.coeffs()])
    D = compose([Fraction(1), Fraction(-p)])
    # inverse of D^{2m+1}
    Dk = [Fraction(1)] + [Fraction(0)] * T
    for _ in range(2 * m + 1):
        Dk = mul(Dk, D)
    inv = [Fraction(0)] * (T + 1)
    inv[0] = 1 / Dk[0]
    for k in range(1, T + 1):
        inv[k] = -sum(Dk[j] * inv[k - j] for j in range(1, k + 1)) / Dk[0]
    return mul(num, inv)


# ----------------------------------------------------------------------------------------------
# route (ii): residues (4_1)
# ----------------------------------------------------------------------------------------------
def _bernoulli(n: int) -> Fraction:
    b = fmpq.bernoulli(n)
    return Fraction(int(b.p), int(b.q))


def bernoulli_poly(n: int) -> List[Fraction]:
    """Coefficients of B_n(y) in powers of y."""
    return [comb(n, k) * _bernoulli(n - k) for k in range(n + 1)]


def _polyval(coeffs, y):
    r = 0 * y
    for c in reversed(coeffs):
        r = r * y + c
    return r


def marino_x1(M: int) -> List[complex]:
    """Marino (3.103): coefficients of hbar^{2m}, m = 0..M, of
         sum_k (k!)^2 hbar^{2k} (1 + sum_n C_n(-i(k+1/2)) hbar^{2n}),
       e^{2i sum_n hbar^{2n} c_{2n}(y)} = 1 + sum_n C_n(y) hbar^{2n},
       c_{2k}(y) = -i (-1)^k B_{2k} B_{2k+1}(1/2 + i y) / (2k (2k+1)!)   (3.92).
    Exact: returned as Fractions (the i's cancel since 1/2 + i y = k + 1 at the poles)."""
    out = [Fraction(0)] * (M + 1)
    for k in range(M + 1):
        arg = Fraction(k + 1)                 # 1/2 + i y at y = -i (k + 1/2)
        # exponent 2 i sum_n c_{2n} hbar^{2n} = sum_n e_n hbar^{2n},  e_n = 2i * (-i)(-1)^n B_2n B_{2n+1}(arg)/(2n(2n+1)!)
        e = [Fraction(0)] * (M + 1)
        for n in range(1, M + 1):
            e[n] = 2 * Fraction((-1) ** n) * _bernoulli(2 * n) * _polyval(bernoulli_poly(2 * n + 1), arg) / (2 * n * factorial(2 * n + 1))
        ex = _exp_series(e, M)
        for m in range(k, M + 1):
            out[m] += factorial(k) ** 2 * ex[m - k]
    return out


def _exp_series(e: Sequence, M: int):
    """exp(sum_{n>=1} e_n t^n) coefficients t^0..t^M (e_0 ignored)."""
    out = [Fraction(1)] + [Fraction(0)] * M
    for k in range(1, M + 1):
        out[k] = sum(j * e[j] * out[k - j] for j in range(1, k + 1)) / k
    return out


def phi0x_residue(dmax: int):
    """Port of Phi0x[dmax] (asymp-x.nb, 'Stand-alone program'); 4_1 only.

    Returns reslist[1..dmax] as sympy rational functions of x.  Conversion to phi_0 coefficients is
    fixed by comparison with route (i) (see recipes/perturbative_series/algorithm.md).
    """
    import sympy as sp
    x, k = sp.symbols("x k")
    ln = 3 * dmax - 2

    def coeflist(a):
        # Taylor coefficients of -a/(e^t - a) up to t^ln
        t = sp.symbols("t")
        ser = sp.series(-a / (sp.exp(t) - a), t, 0, ln + 1).removeO()
        return [sp.together(ser.coeff(t, j)) for j in range(ln + 1)]

    cx = coeflist(x)
    nb = ln // 2
    bn = [2 * cx[2 * n - 1] for n in range(1, nb + 1)]
    an = [(-1) ** (n + 1) * sp.bernoulli(2 * n + 1, -k) / (2 * n * (2 * n + 1)) * bn[n - 1] for n in range(1, nb + 1)]
    # exponential series of sum_n an[n] h^n up to h^dmax
    h = sp.symbols("h")
    f = sum(an[n - 1] * h ** n for n in range(1, min(nb, dmax) + 1))
    E = sp.series(sp.exp(f), h, 0, dmax).removeO()
    Q = [sp.expand(E.coeff(h, n)) for n in range(dmax)]
    w = x + 1 / x - 2
    cw = coeflist(w)
    res = []
    for n in range(dmax):
        poly = sp.Poly(Q[n], k)
        tot = 0
        for (j,), c in poly.terms():
            tot += c * cw[j] * sp.factorial(j)
        res.append(sp.together(tot / w))
    return res


# ----------------------------------------------------------------------------------------------
# route (iii): formal solution of an inhomogeneous q-difference equation
# ----------------------------------------------------------------------------------------------
def qdiff_formal(coeffs, order: int, y0=1):
    """Solve sum_j a_j(y, q) f(q^j y) = 1 formally, f = sum_k f_k(y) h^k with q = e^h, and return f_k(y0).

    ``coeffs``: list of sympy expressions a_j(y, q, x) (symbols y, q, x).  Uses
    f(q^j y) = sum_r (j h)^r/r! (y d/dy)^r f(y).  Returns sympy rational functions of x.
    """
    import sympy as sp
    y, q, x, h = sp.symbols("y q x h")
    J = len(coeffs)
    # expand a_j(y, e^h) in h
    A = []
    for a in coeffs:
        ser = sp.series(a.subs(q, sp.exp(h)), h, 0, order + 1).removeO()
        A.append([sp.together(ser.coeff(h, i)) for i in range(order + 1)])
    f = []

    def theta(expr, r):
        for _ in range(r):
            expr = sp.together(y * sp.diff(expr, y))
        return expr

    lead = sp.together(sum(A[j][0] for j in range(J)))
    for kk in range(order + 1):
        # coefficient of h^kk: sum_j sum_{i + r + s = kk} A_j[i] (j^r/r!) theta^r f_s = delta_{kk,0}
        rhs = 1 if kk == 0 else 0
        acc = 0
        for j in range(J):
            for i in range(kk + 1):
                for r in range(kk - i + 1):
                    s = kk - i - r
                    if s == kk:
                        continue  # unknown f_kk appears only with i = r = 0
                    if A[j][i] == 0:
                        continue
                    acc += A[j][i] * sp.Rational(j ** r, factorial(r)) * theta(f[s], r)
        f.append(sp.together(sp.cancel((rhs - acc) / lead)))
    return [sp.factor(sp.cancel(fk.subs(y, y0))) for fk in f]
