"""6_1: q~-side holomorphic blocks of the 2d NZ integral and the exact upper-sweep Stokes matrix S^+ (ours).

Derived here from the general residue recipe of the 6_1/7_2 notes ((rank2general), steps 1-4) applied to (int61):
  chirals X_j(s) = M_j.s + c_j(u):  M = ((1,-1),(1,-1),(1,0),(0,1)),  c = (-2u, -u, 0, 0),
  Gaussian  e^{-pi i s^T Q s - 2 pi i l(u).s},  Q = ((0,-1),(-1,2)),  l = u (-1, 3),  descendants e^{-2 pi (n/b).s}.
For a Jeffrey-Kirwan pair P = {i, j} (det A_P = 1), with pole indices l_i, l_j >= 0 on the q~ side, lambda = A_P^{-1} (l_i, l_j),
L_r = M_r.lambda (r not in P), s_0 = u g_P and gamma_r = X_r(s_0)/u:
  G_P(xt; qt; n) = sum (-1)^{l_i + l_j} qt^{l_i(l_i+1)/2 + l_j(l_j+1)/2 - lambda^T Q lambda/2 - n.lambda}
                       xt^{-g^T Q lambda - lhat.lambda} / ((qt)_{l_i} (qt)_{l_j} prod_{r not in P} (qt^{-1} xt^{-gamma_r}; qt^{-1})_{L_r}),
the column factor of descendant n is xt^{-n.g_P}.  Conjugation factors r_P = -prod_{r not in P} theta_{gamma_r},
theta_g = (xt^{-g}; qt)_inf / (xt^{g}; qt)_inf (note, (61T)).  On the basis B_4 = {(0,0),(1,0),(2,0),(-1,1)}:
  T_44 = Ytilde(1/xt) diag(r_P) Ytilde(xt)^{-1},   S^+ = -Pi N_I(1/xt) T_44 N_I(xt)^{-1},  Pi = (s1 s2)(s3 s4).
The boundary (abelian) family of the note is assumed to decouple from B_4 (as stated there, T_{B_4,(1,-1)} = 0);
the result is checked against the S^+ printed in the note (golden tests).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Tuple

from ..algebra import RatFun, TSeries, inv_generic, mat_mul
from ..blocks import Ring

M = [(1, -1), (1, -1), (1, 0), (0, 1)]
Q = ((0, -1), (-1, 2))
LHAT = (-1, 3)
PAIRS = {  # pair: (indices (0-based), g_P = s_0/u, gamma_r for r not in P)
    "13": ((0, 2), (0, -2), {1: 1, 3: -2}),
    "23": ((1, 2), (0, -1), {0: -1, 3: -1}),
    "14": ((0, 3), (2, 0), {1: 1, 2: 2}),
    "24": ((1, 3), (1, 0), {0: -1, 2: 1}),
}
BASIS4 = [(0, 0), (1, 0), (2, 0), (-1, 1)]
COLS = ["13", "23", "14", "24"]


def _xp(R, x, e):
    return x ** e if e >= 0 else (R.one / x) ** (-e)


def _poch(R, a, s, n, N):
    """prod_{j=0}^{n-1} (1 - a qt^{s+j}) as TSeries (s may be 0)."""
    r = TSeries([R.one], N, 0, R.zero)
    for j in range(n):
        e = s + j
        if e > N:
            break
        f = [R.one] + [R.zero] * N
        f[e] = f[e] - a
        r = r * TSeries(f, N, 0, R.zero)
    return r


def _inv2(A):
    (a, b), (c, d) = A
    det = a * d - b * c
    assert det in (1, -1)
    return ((d * det, -b * det), (-c * det, a * det))


def _remaining_factor(R, x, gamma: int, L: int, N: int) -> Tuple[TSeries, int]:
    """1/(qt^{-1} xt^{-gamma}; qt^{-1})_L as (series, extra qt power), expanded in qt."""
    a = _xp(R, x, -gamma)
    if L >= 0:
        # prod_{t=1}^{L} 1/(1 - a qt^{-t}) = prod_t (-a^{-1} qt^t) / (1 - a^{-1} qt^t)
        ai = R.one / a
        ser = _poch(R, ai, 1, L, N).inverse() * ((-ai) ** L if L else R.one)
        return ser, L * (L + 1) // 2
    Nn = -L
    # (b; qt^{-1})_{-N} = 1/(b qt^{N}; qt^{-1})_N with b = qt^{-1} a  ->  1/(...)^{-1} = prod_{s=0}^{N-1} (1 - a qt^s)
    return _poch(R, a, 0, Nn, N), 0


def G_block(R: Ring, x, pair: str, n: Tuple[int, int], N: int) -> TSeries:
    (i, j), g, gammas = PAIRS[pair]
    Ainv = _inv2((M[i], M[j]))
    out = TSeries([], N, 0, R.zero)
    vmin = 0
    terms = []
    lim = 4 * N + 12
    for li in range(lim):
        for lj in range(lim):
            lam = (Ainv[0][0] * li + Ainv[0][1] * lj, Ainv[1][0] * li + Ainv[1][1] * lj)
            quad = lam[0] * (Q[0][0] * lam[0] + Q[0][1] * lam[1]) + lam[1] * (Q[1][0] * lam[0] + Q[1][1] * lam[1])
            e = Fraction(li * (li + 1) + lj * (lj + 1), 2) - Fraction(quad, 2) - (n[0] * lam[0] + n[1] * lam[1])
            Ls = {r: M[r][0] * lam[0] + M[r][1] * lam[1] for r in gammas}
            extra = sum(L * (L + 1) // 2 for L in Ls.values() if L > 0)
            if e.denominator != 1:
                raise ValueError("non-integral qt power")
            e = int(e) + extra
            if e > N:
                continue
            gq = g[0] * (Q[0][0] * lam[0] + Q[0][1] * lam[1]) + g[1] * (Q[1][0] * lam[0] + Q[1][1] * lam[1])
            xe = -gq - (LHAT[0] * lam[0] + LHAT[1] * lam[1])
            coef = R.one * ((-1) ** (li + lj)) * _xp(R, x, xe)
            ser = (_poch(R, R.one, 1, li, N) * _poch(R, R.one, 1, lj, N)).inverse() * coef
            for r, L in Ls.items():
                f, _ = _remaining_factor(R, x, gammas[r], L, N)
                ser = ser * f
            terms.append((e, ser))
            vmin = min(vmin, e)
    out = TSeries([], N, vmin, R.zero)
    for e, ser in terms:
        out = out + TSeries(ser.c, N, e, R.zero)
    return out


def theta(R, x, gam: int, N: int) -> TSeries:
    num = _poch(R, _xp(R, x, -gam), 0, N + 1, N)
    den = _poch(R, _xp(R, x, gam), 0, N + 1, N)
    return num / den


def N_I(R: Ring, x, N: int):
    o, z = R.one, R.zero
    c = lambda v: TSeries.const(v, N, R.zero)
    return [[c(-2 * o - o / x), c(z), c(z), c(o / (x * x))],
            [c(-o), c(z), c(z), c(z)],
            [c(-x), c(z), c(x ** 3), c(z)],
            [c(z), c(o), c(z), c(z)]]


def Ytilde(R: Ring, x, N: int):
    return [[G_block(R, x, P, n, N) * _xp(R, x, -(n[0] * PAIRS[P][1][0] + n[1] * PAIRS[P][1][1])) for P in COLS]
            for n in BASIS4]


def splus(order_qt: int, xt=None, guard: int = 12):
    R = Ring.symbolic() if xt is None else Ring.at(xt)
    x = R.x
    N = order_qt + guard
    Y = Ytilde(R, x, N)
    Yi = Ytilde(R, R.one / x, N)
    rP = {}
    for P in COLS:
        prod = TSeries.const(-R.one, N, R.zero)
        for gam in PAIRS[P][2].values():
            prod = prod * theta(R, x, gam, N)
        rP[P] = prod
    YiR = [[Yi[a][b] * rP[COLS[b]] for b in range(4)] for a in range(4)]
    T = mat_mul(YiR, inv_generic(Y))
    M1 = mat_mul(mat_mul(N_I(R, R.one / x, N), T), inv_generic(N_I(R, x, N)))
    Pi = [1, 0, 3, 2]
    S = [[-M1[Pi[a]][b] for b in range(4)] for a in range(4)]
    return S, R
