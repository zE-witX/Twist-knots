"""Exact upper-sweep Stokes matrix S^+ of 7_2 from its two-dimensional holomorphic blocks.

Reconstruction of the procedure of the 6_1/7_2 notes (labels: (72factor), (72NI), Thm 72Splus):
    Z_n = sum_{j in {2,1,0}, eps in {0,1}} C_j(u) F^eps_j(x;q) xt^{-j n_1} G^eps_j(xt;qt;n)        (m = 0)
    G^eps_j = sum_{l1,l2>=0} (-1)^{l1+(1+eps) l2} qt^{l1^2/2 + l1 l2 + l2^2/4 + (l1+l2)/2 - n.l} xt^{(j-1)(l1+l2)}
              / ((qt;qt)_{l1} (qt;qt)_{l2} prod_{j' != j} (qt xt^{j-j'};qt)_{l1})
    conjugation tau -> -conj(tau):  conj C_j = r_j C_j,  r_2 = -xt th1 th2, r_1 = -xt^2, r_0 = -xt^3/(th1 th2),
                                    th_k = (qt xt^{-k};qt)_inf/(qt xt^k;qt)_inf
    T = Ytilde_B(1/xt) R E (Ytilde_B E)^{-1}    on the basis B = {(0,0),(0,1),(1,0),(1,1),(2,1)}
    S^+ = diag(-1,-1,-1,-1,-xt^2) Pi N_I(1/xt, qt) T(xt, qt) N_I(xt, qt)^{-1},   Pi = (s1 s2)(s3 s4).
Series are in s = qt^{1/4}; the result must contain integer powers of qt only (asserted by the caller).
Rows/columns: sigma_1 .. sigma_5 (geometric, conjugate, complex pair, real).
"""
from __future__ import annotations

from typing import List, Tuple

from ..algebra import RatFun, TSeries, inv_generic, mat_mul
from ..blocks import Ring

BASIS = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]
COLUMNS = [(j, e) for j in (2, 1, 0) for e in (0, 1)]


def _S(R, cs, N, val=0):
    return TSeries(cs, N, val, R.zero)


def _poch_s(R, a, k0, n, N):
    """prod_{i=0}^{n-1} (1 - a qt^{k0+i}) in s = qt^{1/4}."""
    r = _S(R, [R.one], N)
    for i in range(n):
        e = 4 * (k0 + i)
        if e > N:
            break
        f = [R.one] + [R.zero] * N
        f[e] = f[e] - a
        r = r * _S(R, f, N)
    return r


def _xp(R, x, e):
    return x ** e if e >= 0 else (R.one / x) ** (-e)


def G_block(R: Ring, x, j: int, eps: int, n: Tuple[int, int], N: int) -> TSeries:
    """G^eps_j(xt; qt; n) as a series in s = qt^{1/4}, truncated at s^N (negative powers allowed)."""
    n1, n2 = n
    terms = {}
    vmin = 0
    L = 0
    while 2 * L * L - 4 * (n1 + n2) * L - 8 * L <= N + 8:
        L += 1
    L += 2
    for l1 in range(L + 1):
        for l2 in range(2 * L + 2):
            e4 = 2 * l1 * l1 + 4 * l1 * l2 + l2 * l2 + 2 * l1 + 2 * l2 - 4 * (n1 * l1 + n2 * l2)
            if e4 > N:
                continue
            sign = (-1) ** (l1 + (1 + eps) * l2)
            coef = R.one * sign * _xp(R, x, (j - 1) * (l1 + l2))
            den = _poch_s(R, R.one, 1, l1, N) * _poch_s(R, R.one, 1, l2, N)
            for jp in (2, 1, 0):
                if jp != j:
                    den = den * _poch_s(R, _xp(R, x, j - jp), 1, l1, N)
            term = den.inverse() * coef
            # shift by s^{e4}
            terms[(l1, l2)] = (e4, term)
            vmin = min(vmin, e4)
    out = _S(R, [], N, vmin)
    for e4, term in terms.values():
        shifted = TSeries([R.zero] * (e4 - vmin) + term.c, N, vmin, R.zero) if True else None
        out = out + shifted
    return out


def theta_k(R, x, k: int, N: int) -> TSeries:
    num = _poch_s(R, _xp(R, x, -k), 1, N // 4 + 2, N)
    den = _poch_s(R, _xp(R, x, k), 1, N // 4 + 2, N)
    return num / den


def N_I(R: Ring, x, N: int):
    """(72NI): rows sigma_1..sigma_5, columns Z_(0,0), Z_(0,1), Z_(1,0), Z_(1,1), Z_(2,1)."""
    o, z = R.one, R.zero

    def c(v):
        return TSeries.const(v, N, R.zero)

    qinv = TSeries.monomial(R.one, -4, N, R.zero)  # qt^{-1} = s^{-4}
    M = [[c(-x ** 2), c(z), c(-x ** 3), c(z), c(z)],
         [c(-o), c(z), c(z), c(z), c(z)],
         [c(o + x + x ** 2), c(z), c(-x - x ** 2 - x ** 3) + qinv * (x ** 2), c(z), c(x ** 3)],
         [c(z), c(z), c(-x ** 2), c(x ** 2), c(z)],
         [c(-x ** 2), c(x ** 2), c(z), c(z), c(z)]]
    return M


def Ytilde(R: Ring, x, N: int):
    """5 x 6 matrix xt^{-j n_1} G^eps_j(n), rows n in BASIS, columns (j, eps) in COLUMNS."""
    return [[G_block(R, x, j, e, n, N) * _xp(R, x, -j * n[0]) for (j, e) in COLUMNS] for n in BASIS]


def splus(order_qt: int, xt=None, columns=(0, 1, 2, 3, 4), guard: int = 32):
    """S^+ of 7_2 as 5x5 matrix of series in s = qt^{1/4}, through qt^{order_qt}."""
    R = Ring.symbolic() if xt is None else Ring.at(xt)
    x = R.x
    N = 4 * order_qt + guard
    Y = Ytilde(R, x, N)
    Yi = Ytilde(R, R.one / x, N)
    th1, th2 = theta_k(R, x, 1, N), theta_k(R, x, 2, N)
    r = {2: -(th1 * th2) * x, 1: TSeries.const(-x ** 2, N, R.zero), 0: (th1 * th2).inverse() * (-x ** 3)}
    YiR = [[Yi[a][b] * r[COLUMNS[b][0]] for b in range(6)] for a in range(5)]
    E = list(columns)
    YE = [[Y[a][b] for b in E] for a in range(5)]
    YiRE = [[YiR[a][b] for b in E] for a in range(5)]
    T = mat_mul(YiRE, inv_generic(YE))
    NIx = N_I(R, x, N)
    NIi = N_I(R, R.one / x, N)
    Pi = [1, 0, 3, 2, 4]
    D = [-R.one, -R.one, -R.one, -R.one, -(x ** 2)]
    M = mat_mul(mat_mul(NIi, T), inv_generic(NIx))
    S = [[M[Pi[a]][b] * D[a] for b in range(5)] for a in range(5)]
    return S, R


def qt_coefficients(S_entry: TSeries, order_qt: int):
    """Return [c_0, ..., c_K] of qt^k, asserting that only powers s^{4k} occur."""
    out = []
    if not isinstance(S_entry, TSeries):
        return [S_entry] + [0 * S_entry] * order_qt
    if S_entry.N < 4 * order_qt:
        raise ValueError(f"precision lost: entry known to s^{S_entry.N} < qt^{order_qt}; raise guard")
    for k4 in range(min(S_entry.val, 0), 4 * order_qt + 1):
        c = S_entry[k4]
        zero = c.num == 0 if isinstance(c, RatFun) else c == 0
        if k4 % 4 != 0 or k4 < 0:
            if not zero:
                raise AssertionError(f"non-integral qt power s^{k4} with coefficient {c}")
            continue
        out.append(c)
    return out
