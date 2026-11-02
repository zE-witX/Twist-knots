"""The 3D index of Dimofte-Gaiotto-Gukov from Neumann-Zagier data of an ideal triangulation.

Tetrahedron index (exponents of q may be half-integral; we work in units of q^{1/2}):
    I_Delta(m, e)(q) = sum_{n >= max(0,-e)} (-1)^n q^{n(n+1)/2 - (n + e/2) m} / ((q;q)_n (q;q)_{n+e}).
Gluing data: SnapPy's gluing equations (rows: edges, meridian, longitude; columns (z, z', z'') per
tetrahedron).  Eliminating z' (z z' z'' = -1): A = a - b, B = c - b, nu_i = 2 - sum_j b_ij (edges), 0 (cusp).
Index (N tetrahedra, sum over N-1 edge charges k, one edge equation is redundant):
    I(m, e)(q) = sum_k (-q^{1/2})^{nu.k + e nu_mu + m nu_lam/2} prod_j I_Delta(m_j, e_j),
    m_j = -sum_i k_i B_ij - e B_mu,j - m B_lam,j / 2,      e_j = sum_i k_i A_ij + e A_mu,j + m A_lam,j / 2,
(the electric charge couples to the meridian, the magnetic flux to lambda/2).  Conventions are fixed by:
  * Ind^rot(0, x; q) = S^+_{gg}(x; q) exactly (x-graded): 4_1, 5_2, 7_2 (and 6_1 at x = 1 against the notes);
  * flux: for 5_2, sum_alpha B^alpha(q^{m/2} z; q) B^alpha(q^{m/2}/z; 1/q) = z^{-3m} Ind^rot(m, z; q) (BDP / GGM (19),
    with the monomial of GGM (314)), checked numerically for m = 0, +-2.
Rotated index: Ind^rot(m, zeta; q) = sum_e I(m, e)(q) zeta^e.
"""
from __future__ import annotations

import functools
import itertools
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple


class Index:
    """3D index calculator for fixed NZ data and truncation order (q^order, q^{1/2} units internally)."""

    def __init__(self, gluing: Sequence[Sequence[int]], order: int):
        self.G = [list(r) for r in gluing]
        self.N = len(self.G[0]) // 3
        self.H = 2 * order
        N = self.N
        self.A = [[r[3 * j] - r[3 * j + 1] for j in range(N)] for r in self.G]
        self.B = [[r[3 * j + 2] - r[3 * j + 1] for j in range(N)] for r in self.G]
        self.nu = [(2 if i < N else 0) - sum(r[3 * j + 1] for j in range(N)) for i, r in enumerate(self.G)]
        self.cusp_sign = 1
        self._tc = {}
        self._ipc = {}

    # ---- tetrahedron index -------------------------------------------------------------------
    def _inv_poch(self, n: int, L: int):
        """1/(q;q)_n through q^{L/2} (q^{1/2} units), cached at the largest length requested."""
        have = self._ipc.get(n)
        if have is not None and len(have) > L:
            return have
        L2 = max(L, 2 * (len(have) if have else 0), 16)
        r = [0] * (L2 + 1)
        r[0] = 1
        for j in range(1, n + 1):          # multiply by 1/(1 - q^j) = sum_k q^{jk}
            s = 2 * j
            if s > L2:
                break
            for i in range(s, L2 + 1):
                r[i] += r[i - s]
        r = tuple(r)
        self._ipc[n] = r
        return r

    def _tet_raw(self, m: int, e: int, H: int) -> Dict[int, int]:
        out: Dict[int, int] = {}
        n = max(0, -e)
        while True:
            ex2 = n * (n + 1) - (2 * n + e) * m
            if ex2 > H and n > abs(m) + abs(e) + 2:
                break
            if ex2 <= H:
                L = H - ex2
                a = self._inv_poch(n, L)
                b = self._inv_poch(n + e, L)
                sg = -1 if n % 2 else 1
                for i in range(L + 1):
                    x = a[i]
                    if x:
                        for j in range(L + 1 - i):
                            y = b[j]
                            if y:
                                out[ex2 + i + j] = out.get(ex2 + i + j, 0) + sg * x * y
            n += 1
        return {k: v for k, v in out.items() if v}

    def tet(self, m: int, e: int, H: int = None) -> Dict[int, int]:
        """Tetrahedron index known through q^{H/2} (default: the global order); cached at the largest H."""
        H = self.H if H is None else H
        got = self._tc.get((m, e))
        if got is None or got[0] < H:
            Hc = max(H, got[0] * 2 if got else 0, self.H)
            got = (Hc, self._tet_raw(m, e, Hc))
            self._tc[(m, e)] = got
        if got[0] == H:
            return got[1]
        return {k: v for k, v in got[1].items() if k <= H}

    def mindeg(self, m: int, e: int) -> int:
        """Lowest exponent (q^{1/2} units) of I_Delta(m, e)."""
        key = ("deg", m, e)
        if key in self._tc:
            return self._tc[key]
        lo = min(n * (n + 1) - (2 * n + e) * m for n in range(max(0, -e), abs(m) + abs(e) + 3))
        H = max(lo + 8, self.H)
        while True:
            d = self.tet(m, e, H)
            if d:
                val = min(d)
                break
            if H > 6 * self.H + 4 * abs(lo) + 40:
                val = H + 1          # starts beyond anything that can matter
                break
            H = 2 * H + 8
        self._tc[key] = val
        return val

    # ---- state sum ------------------------------------------------------------------------------
    def _mul(self, a, b):
        H = self.H
        out = {}
        for x, u in a.items():
            for y, v in b.items():
                if x + y <= H:
                    out[x + y] = out.get(x + y, 0) + u * v
        return out

    def _term(self, kk, m, e):
        N = self.N
        mu, lam = N, N + 1
        args = []
        for j in range(N):
            mj = Fraction(-sum(kk[i] * self.B[i][j] for i in range(N)) - e * self.B[mu][j]) - Fraction(m * self.B[lam][j], 2)
            ej = Fraction(sum(kk[i] * self.A[i][j] for i in range(N)) + e * self.A[mu][j]) + Fraction(m * self.A[lam][j], 2)
            if mj.denominator != 1 or ej.denominator != 1:
                raise ValueError("half-integral tetrahedron charges: use even flux m")
            args.append((int(mj), int(ej)))
        pre = Fraction(sum(self.nu[i] * kk[i] for i in range(N))) + self.cusp_sign * e * self.nu[mu] \
            + self.cusp_sign * Fraction(m * self.nu[lam], 2)
        if pre.denominator != 1:
            raise ValueError("half-integral cusp phase: use even flux m")
        pre = int(pre)
        degs = [self.mindeg(*a) for a in args]
        total = pre + sum(degs)
        if total > self.H:
            return None
        prod = {pre: -1 if pre % 2 else 1}
        for a, d in zip(args, degs):
            # this factor is needed up to H - (lowest degree of everything else)
            prod = self._mul(prod, self.tet(*a, H=self.H - (total - d)))
            if not prod:
                break
        return prod

    def _degree(self, kk, m, e):
        """Lowest q^{1/2}-exponent of the summand at edge charges kk (None if charges are half-integral)."""
        N = self.N
        mu, lam = N, N + 1
        tot = Fraction(sum(self.nu[i] * kk[i] for i in range(N))) + self.cusp_sign * e * self.nu[mu] \
            + self.cusp_sign * Fraction(m * self.nu[lam], 2)
        for j in range(N):
            mj = Fraction(-sum(kk[i] * self.B[i][j] for i in range(N)) - e * self.B[mu][j]) - Fraction(m * self.B[lam][j], 2)
            ej = Fraction(sum(kk[i] * self.A[i][j] for i in range(N)) + e * self.A[mu][j]) + Fraction(m * self.A[lam][j], 2)
            if mj.denominator != 1 or ej.denominator != 1:
                raise ValueError("half-integral tetrahedron charges: use even flux m")
            tot += self.mindeg(int(mj), int(ej))
        return tot

    def I(self, m: int, e: int, margin: int = 2) -> Dict[int, int]:
        """I(m, e)(q) as {exponent in q^{1/2} units: coefficient}, truncated at q^{order}.

        The summation region {k in Z^{N-1}: degree(k) <= H} is found by descending from k = 0 to a minimum of the
        degree function and flood-filling (with a safety margin of `margin` layers beyond the region)."""
        N = self.N
        d = N - 1
        if d == 0:
            t = self._term([0], m, e)
            return {k: v for k, v in (t or {}).items() if v}
        deg = {}

        def D(k):
            if k not in deg:
                deg[k] = self._degree(list(k) + [0], m, e)
            return deg[k]

        nbrs = [tuple((1 if i == j else 0) * s for i in range(d)) for j in range(d) for s in (1, -1)]
        k = tuple([0] * d)
        while True:  # steepest descent
            best = min(((D(tuple(a + b for a, b in zip(k, v))), tuple(a + b for a, b in zip(k, v))) for v in nbrs))
            if best[0] < D(k):
                k = best[1]
            else:
                break
        if D(k) > self.H:
            return {}
        # flood fill: points with degree <= H, plus `margin` layers of points above H (to cross small barriers)
        seen = {k: 0}
        frontier = [k]
        while frontier:
            nxt = []
            for p in frontier:
                lay = seen[p]
                for v in nbrs:
                    r = tuple(a + b for a, b in zip(p, v))
                    if r in seen:
                        continue
                    l2 = 0 if D(r) <= self.H else lay + 1
                    if l2 > margin:
                        continue
                    seen[r] = l2
                    nxt.append(r)
            frontier = nxt
        tot: Dict[int, int] = {}
        for p, lay in seen.items():
            if lay:
                continue
            t = self._term(list(p) + [0], m, e)
            if t:
                for x, v in t.items():
                    tot[x] = tot.get(x, 0) + v
        return {k: v for k, v in tot.items() if v}

    def rotated_x(self, m: int) -> List[Dict[int, int]]:
        """(computed internally two orders deeper than requested, then truncated)"""
        return self._rotated_x(m)

    def _rotated_x(self, m: int) -> List[Dict[int, int]]:
        """Ind^rot(m, x; q) as a list over q^0..q^order of Laurent polynomials {e: coeff} (integral powers only)."""
        order = self.H // 2
        out = [dict() for _ in range(order + 1)]
        e = 0
        empty_run = 0
        for e_abs in range(0, 4 * order + 8):
            any_term = False
            for e in ({0} if e_abs == 0 else {e_abs, -e_abs}):
                d = self.I(m, e)
                for k, v in d.items():
                    if k % 2:
                        raise ValueError("half-integral power of q in Ind^rot")
                    out[k // 2][e] = out[k // 2].get(e, 0) + v
                    any_term = True
            empty_run = 0 if any_term else empty_run + 1
            if empty_run >= 3:
                break
        return [{e: v for e, v in d.items() if v} for d in out]

    def rotated(self, m: int, emax: int) -> Dict[Tuple[int, int], int]:
        """Ind^rot(m, zeta; q) = sum_e I(m,e) zeta^e as {(e, half-exponent): coefficient}, |e| <= emax."""
        out = {}
        for e in range(-emax, emax + 1):
            for k, v in self.I(m, e).items():
                out[(e, k)] = v
        return out


def as_series(d: Dict[int, int], order: int) -> List:
    """q^{1/2}-unit dict -> list of coefficients of q^0..q^order (asserts integral powers)."""
    out = [0] * (order + 1)
    for k, v in d.items():
        if k % 2:
            raise ValueError("half-integral power of q in the index")
        if 0 <= k // 2 <= order:
            out[k // 2] += v
        elif k < 0:
            raise ValueError("negative power of q in the index")
    return out


def gluing_from_snappy(name: str):
    import snappy
    M = snappy.Manifold(name)
    return [list(map(int, r)) for r in M.gluing_equations()], str(M.identify())
