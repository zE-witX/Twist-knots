"""Exact algebra used throughout kdb.

* ``RatFun``  : rational functions of one variable over Q (Laurent monomials allowed), backed by
                python-flint ``fmpq_poly`` with gcd normalisation.
* ``TSeries`` : truncated power series ``sum_{k=val}^{N} c_k t^k`` with coefficients in any field-like
                type (Fraction, RatFun, mpmath numbers, flint acb ...).
* small dense-matrix helpers over these types.

Everything is exact unless the coefficient type itself is numeric.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, List, Sequence

from flint import fmpq, fmpq_poly


# ----------------------------------------------------------------------------------------------
# Rational functions in one variable
# ----------------------------------------------------------------------------------------------
def _to_fmpq(c) -> fmpq:
    if isinstance(c, fmpq):
        return c
    if isinstance(c, Fraction):
        return fmpq(c.numerator, c.denominator)
    return fmpq(c)


class RatFun:
    """num/den with fmpq_poly num, den; den monic; gcd(num, den) = 1."""

    __slots__ = ("num", "den")
    var = "xt"

    def __init__(self, num, den=None, _normal=False):
        if not isinstance(num, fmpq_poly):
            num = fmpq_poly([_to_fmpq(num)])
        if den is None:
            den = fmpq_poly([1])
        elif not isinstance(den, fmpq_poly):
            den = fmpq_poly([_to_fmpq(den)])
        if not _normal:
            if den == 0:
                raise ZeroDivisionError("RatFun with zero denominator")
            if num == 0:
                den = fmpq_poly([1])
            else:
                g = num.gcd(den)
                if g.degree() > 0:
                    num = num // g
                    den = den // g
                lc = den.coeffs()[-1]
                if lc != 1:
                    num = num / lc
                    den = den / lc
        self.num = num
        self.den = den

    # constructors --------------------------------------------------------------------------
    @classmethod
    def x(cls) -> "RatFun":
        return cls(fmpq_poly([0, 1]), _normal=True)

    @classmethod
    def monomial(cls, c, e: int) -> "RatFun":
        c = _to_fmpq(c)
        if e >= 0:
            return cls(fmpq_poly([0] * e + [c]), _normal=c != 0)
        return cls(fmpq_poly([c]), fmpq_poly([0] * (-e) + [1]), _normal=c != 0)

    @classmethod
    def from_laurent(cls, d: Dict[int, object]) -> "RatFun":
        r = cls(0)
        for e, c in d.items():
            r = r + cls.monomial(c, e)
        return r

    @staticmethod
    def coerce(o) -> "RatFun":
        return o if isinstance(o, RatFun) else RatFun(o)

    # arithmetic ------------------------------------------------------------------------------
    def __add__(self, o):
        o = RatFun.coerce(o)
        if self.den == o.den:
            return RatFun(self.num + o.num, self.den)
        return RatFun(self.num * o.den + o.num * self.den, self.den * o.den)

    __radd__ = __add__

    def __neg__(self):
        return RatFun(-self.num, self.den, _normal=True)

    def __sub__(self, o):
        return self + (-RatFun.coerce(o))

    def __rsub__(self, o):
        return RatFun.coerce(o) - self

    def __mul__(self, o):
        o = RatFun.coerce(o)
        if o.den == 1 and o.num.degree() == 0:
            c = o.num.coeffs()[0] if o.num != 0 else fmpq(0)
            if c == 0:
                return RatFun(0)
            return RatFun(self.num * c, self.den, _normal=True)
        return RatFun(self.num * o.num, self.den * o.den)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = RatFun.coerce(o)
        if o.num == 0:
            raise ZeroDivisionError
        return RatFun(self.num * o.den, self.den * o.num)

    def __rtruediv__(self, o):
        return RatFun.coerce(o) / self

    def __pow__(self, n: int):
        if n < 0:
            return RatFun(1) / (self ** (-n))
        return RatFun(self.num ** n, self.den ** n, _normal=True)

    def __eq__(self, o):
        try:
            o = RatFun.coerce(o)
        except Exception:
            return NotImplemented
        return self.num == o.num and self.den == o.den

    def __hash__(self):
        return hash((str(self.num), str(self.den)))

    def is_zero(self) -> bool:
        return self.num == 0

    # evaluation / inspection -----------------------------------------------------------------
    def __call__(self, v):
        """Evaluate at v (Fraction / int exact, or any numeric type)."""
        if isinstance(v, (int, Fraction)):
            v = Fraction(v)
            n = sum((Fraction(int(c.p), int(c.q)) * v ** i for i, c in enumerate(self.num.coeffs())), Fraction(0))
            d = sum((Fraction(int(c.p), int(c.q)) * v ** i for i, c in enumerate(self.den.coeffs())), Fraction(0))
            return n / d
        n = sum((_num(c) * v ** i for i, c in enumerate(self.num.coeffs())), 0 * v)
        d = sum((_num(c) * v ** i for i, c in enumerate(self.den.coeffs())), 0 * v)
        return n / d

    def laurent(self) -> Dict[int, Fraction] | None:
        """If self is a Laurent polynomial c_k x^k, return {k: c_k}, else None."""
        dc = self.den.coeffs()
        # den monic; Laurent iff den = x^m
        if any(c != 0 for c in dc[:-1]):
            return None
        m = len(dc) - 1
        return {i - m: Fraction(int(c.p), int(c.q)) for i, c in enumerate(self.num.coeffs()) if c != 0}

    def is_integral_laurent(self) -> bool:
        d = self.laurent()
        return d is not None and all(c.denominator == 1 for c in d.values())

    def __repr__(self):
        return self.to_str()

    def to_str(self, var: str | None = None) -> str:
        var = var or RatFun.var
        lp = self.laurent()
        if lp is not None:
            return laurent_to_str(lp, var)
        return "(" + _poly_str(self.num, var) + ")/(" + _poly_str(self.den, var) + ")"


def _num(c):
    return Fraction(int(c.p), int(c.q))


def _poly_str(p: fmpq_poly, var: str) -> str:
    return laurent_to_str({i: _num(c) for i, c in enumerate(p.coeffs()) if c != 0}, var)


def laurent_to_str(d: Dict[int, Fraction], var: str) -> str:
    """Text parseable by sympy and Mathematica, e.g. ``2*xt^-1 + 3 - 1/2*xt^2``."""
    if not d:
        return "0"
    parts = []
    for e in sorted(d):
        c = Fraction(d[e])
        if c == 0:
            continue
        if e == 0:
            mon = ""
        elif e == 1:
            mon = var
        else:
            mon = f"{var}^({e})" if e < 0 else f"{var}^{e}"
        if mon == "":
            s = str(c)
        elif c == 1:
            s = mon
        elif c == -1:
            s = "-" + mon
        else:
            s = f"{c}*{mon}"
        parts.append(s)
    if not parts:
        return "0"
    out = parts[0]
    for p in parts[1:]:
        out += (" - " + p[1:]) if p.startswith("-") else (" + " + p)
    return out


# ----------------------------------------------------------------------------------------------
# Truncated power series
# ----------------------------------------------------------------------------------------------
class TSeries:
    """sum_{k=val}^{N} c[k-val] t^k, i.e. known modulo O(t^{N+1})."""

    __slots__ = ("c", "val", "N", "zero")

    def __init__(self, coeffs: Sequence, N: int, val: int = 0, zero=None):
        self.zero = zero if zero is not None else (coeffs[0] * 0 if coeffs else 0)
        self.val = val
        self.N = N
        c = list(coeffs)[: max(0, N - val + 1)]
        c += [self.zero] * (N - val + 1 - len(c))
        self.c = c

    @classmethod
    def const(cls, a, N, zero=None):
        return cls([a], N, 0, zero if zero is not None else a * 0)

    @classmethod
    def monomial(cls, a, k, N, zero=None):
        z = zero if zero is not None else a * 0
        if k > N:
            return cls([], N, 0, z)
        if k < 0:
            return cls([a], N, k, z)
        return cls([z] * k + [a], N, 0, z)

    def __getitem__(self, k):
        i = k - self.val
        if 0 <= i < len(self.c):
            return self.c[i]
        if k > self.N:
            raise IndexError(f"coefficient t^{k} beyond truncation order {self.N}")
        return self.zero

    def coeffs(self, start=None):
        start = self.val if start is None else start
        return [self[k] for k in range(start, self.N + 1)]

    def _aligned(self, o):
        o = self._coerce(o)
        N = min(self.N, o.N)
        v = min(self.val, o.val)
        return N, v, o

    def _coerce(self, o):
        if isinstance(o, TSeries):
            return o
        return TSeries([o], self.N, 0, self.zero)

    def __add__(self, o):
        N, v, o = self._aligned(o)
        return TSeries([self._get0(k) + o._get0(k) for k in range(v, N + 1)], N, v, self.zero)

    __radd__ = __add__

    def _get0(self, k):
        i = k - self.val
        return self.c[i] if 0 <= i < len(self.c) else self.zero

    def __neg__(self):
        return TSeries([-a for a in self.c], self.N, self.val, self.zero)

    def __sub__(self, o):
        return self + (-self._coerce(o))

    def __rsub__(self, o):
        return self._coerce(o) - self

    def __mul__(self, o):
        if not isinstance(o, TSeries):
            return TSeries([a * o for a in self.c], self.N, self.val, self.zero)
        # leading zero coefficients would otherwise cost precision in the bookkeeping below
        if self.c and _iszero(self.c[0]):
            self = self.normalize()
        if o.c and _iszero(o.c[0]):
            o = o.normalize()
        if not self.c or not o.c:
            N = min(self.N + (o.val if o.c else o.N + 1), o.N + (self.val if self.c else self.N + 1))
            return TSeries([], N, 0, self.zero)
        v = self.val + o.val
        # precision: (a + O(t^{Na+1}))(b + O(t^{Nb+1})) known to min(Na + vb, Nb + va)
        N = min(self.N + o.val, o.N + self.val)
        out = [self.zero] * (N - v + 1)
        for i, a in enumerate(self.c):
            if _iszero(a):
                continue
            for j, b in enumerate(o.c):
                k = i + j
                if k > N - v:
                    break
                if _iszero(b):
                    continue
                out[k] = out[k] + a * b
        return TSeries(out, N, v, self.zero)

    __rmul__ = __mul__

    def normalize(self):
        """Drop leading zeros (raise valuation)."""
        i = 0
        while i < len(self.c) and _iszero(self.c[i]):
            i += 1
        return TSeries(self.c[i:], self.N, self.val + i, self.zero)

    def inverse(self):
        s = self.normalize()
        if not s.c:
            raise ZeroDivisionError("inverse of O(t^N) series")
        a0 = s.c[0]
        n = s.N - s.val
        inv0 = 1 / a0
        r = [inv0]
        for k in range(1, n + 1):
            acc = s.zero
            for j in range(1, k + 1):
                if j < len(s.c) and not _iszero(s.c[j]):
                    acc = acc + s.c[j] * r[k - j]
            r.append(-acc * inv0)
        # (t^v u)^{-1} = t^{-v} u^{-1}, u^{-1} known to order n
        return TSeries(r, n - s.val, -s.val, self.zero)

    def __truediv__(self, o):
        if isinstance(o, TSeries):
            return self * o.inverse()
        return TSeries([a / o for a in self.c], self.N, self.val, self.zero)

    def __rtruediv__(self, o):
        return self._coerce(o) * self.inverse()

    def __pow__(self, n: int):
        if n < 0:
            return (self ** (-n)).inverse()
        r = TSeries([self.zero + 1], self.N, 0, self.zero)
        b = self
        while n:
            if n & 1:
                r = r * b
            b = b * b
            n >>= 1
        return r

    def map(self, f: Callable):
        return TSeries([f(a) for a in self.c], self.N, self.val, f(self.zero))

    def truncate(self, N):
        return TSeries(self.c, min(N, self.N), self.val, self.zero)

    def subs_power(self, m: int):
        """t -> t^m (m >= 1)."""
        out = {}
        for i, a in enumerate(self.c):
            out[(self.val + i) * m] = a
        v = self.val * m
        N = self.N * m + (m - 1)
        cs = [out.get(k, self.zero) for k in range(v, N + 1)]
        return TSeries(cs, N, v, self.zero)

    def __repr__(self):
        terms = [f"({a})*t^{self.val + i}" for i, a in enumerate(self.c) if not _iszero(a)]
        return " + ".join(terms) + f" + O(t^{self.N + 1})"


def _iszero(a) -> bool:
    if isinstance(a, RatFun):
        return a.num == 0
    try:
        return a == 0
    except Exception:
        return False


# ----------------------------------------------------------------------------------------------
# Matrices (lists of lists) over any ring supporting + - *
# ----------------------------------------------------------------------------------------------
def mat_mul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    out = []
    for i in range(n):
        row = []
        for j in range(p):
            acc = None
            for k in range(m):
                a, b = A[i][k], B[k][j]
                if _isint0(a) or _isint0(b):
                    continue
                t = a * b
                acc = t if acc is None else acc + t
            row.append(acc if acc is not None else 0)
        out.append(row)
    return out


def _isint0(a):
    return isinstance(a, (int, Fraction)) and a == 0


def det3(A):
    return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
            - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
            + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))


def inv_generic(A):
    """Inverse by Gauss-Jordan over a field-like type (works for TSeries with invertible pivots)."""
    n = len(A)
    M = [list(row) + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if not _is_zero_any(M[r][col]):
                piv = r
                break
        if piv is None:
            raise ZeroDivisionError("singular matrix")
        M[col], M[piv] = M[piv], M[col]
        pinv = 1 / M[col][col] if not isinstance(M[col][col], TSeries) else M[col][col].inverse()
        M[col] = [_mul(a, pinv) for a in M[col]]
        for r in range(n):
            if r != col and not _is_zero_any(M[r][col]):
                f = M[r][col]
                M[r] = [_sub(a, _mul(f, b)) for a, b in zip(M[r], M[col])]
    return [row[n:] for row in M]


def _is_zero_any(a):
    if isinstance(a, TSeries):
        return all(_iszero(c) for c in a.c)
    return _iszero(a)


def _mul(a, b):
    if _isint0(a) or _isint0(b):
        return 0
    if isinstance(b, TSeries) and not isinstance(a, TSeries):
        return b * a
    return a * b


def _sub(a, b):
    if _isint0(b):
        return a
    if _isint0(a):
        return -b
    if isinstance(b, TSeries) and not isinstance(a, TSeries):
        return -(b - a)
    return a - b


def const_mat(M, embed: Callable):
    return [[embed(a) for a in row] for row in M]


def int_inverse(M: List[List]) -> List[List[Fraction]]:
    """Exact inverse of a small matrix with Fraction/int (or RatFun) entries."""
    return inv_generic([[a for a in row] for row in M])
