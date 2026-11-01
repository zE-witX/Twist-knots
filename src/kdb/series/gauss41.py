"""phi_g, phi_c of 4_1: formal Gaussian expansion of the new state integral.

Port of the step list of the porting guide ``phi_gc_gaussian`` (notebook functions in brackets):
  1. exponent coefficients p_1..p_{2n}            [exponentTerms, classicalD, quantumD, liMinus]
  2. exponentiate e_j = (1/j) sum_k k p_k e_{j-k}  [expCoefficients]
  3. Gaussian average a_n = <e_{2n}>               [gaussianAverage, gaussianMoment]
  4. generic x: exact over Q(x)[y]/(x y^2 + (x^2-x-1) y + 1); scaled (y - 1/(xy))^{3n} a_n is y-free
  5. fixed x: numerics at working precision (default 240 digits), both roots y_pm.
Stored coefficients are h-coefficients (h = 2 pi i tau); they include the regular factor e^{-h/24}.
Full sector (notebook normalisation, = GGMW (85)-(88)):  e^{A/h} i x^{-1/2} F / sqrt(i delta).
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import List, Sequence

from flint import acb, acb_poly, ctx, fmpq

from ..algebra import RatFun


# ----------------------------------------------------------------------------------------------
# combinatorial helpers
# ----------------------------------------------------------------------------------------------
def stirling2(n: int, k: int) -> int:
    S = [[0] * (k + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            S[i][j] = j * S[i - 1][j] + S[i - 1][j - 1]
    return S[n][k]


_STIR = {}


def stirling_row(m: int) -> List[int]:
    """[k! S(m+1, k+1) for k = 0..m]  (coefficients of Li_{-m}(t) in powers of t/(1-t))."""
    if m not in _STIR:
        _STIR[m] = [factorial(k) * stirling2(m + 1, k + 1) for k in range(m + 1)]
    return _STIR[m]


def bernoulli_half(n: int) -> Fraction:
    """B_n(1/2) = (2^{1-n} - 1) B_n."""
    b = fmpq.bernoulli(n)
    return (Fraction(2) ** (1 - n) - 1) * Fraction(int(b.p), int(b.q))


def li_minus(m: int, t):
    """Li_{-m}(t) = sum_k k! S(m+1,k+1) (t/(1-t))^{k+1}  (liMinus); t in any field."""
    w = t / (1 - t)
    acc = 0 * t
    pw = w
    for c in stirling_row(m):
        acc = acc + pw * c
        pw = pw * w
    return acc


def double_factorial(n: int) -> int:
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


# ----------------------------------------------------------------------------------------------
# the quadratic field K = F[y]/(x y^2 + (x^2 - x - 1) y + 1), F = Q(x) or Q (x rational)
# ----------------------------------------------------------------------------------------------
class QF:
    """a + b y with y a root of x y^2 + (x^2-x-1) y + 1 = 0 (the saddle curve)."""

    __slots__ = ("a", "b", "F")

    def __init__(self, a, b, F):
        self.a, self.b, self.F = a, b, F

    def _c(self, o):
        return o if isinstance(o, QF) else QF(self.F.one * o, self.F.zero, self.F)

    def __add__(self, o):
        o = self._c(o)
        return QF(self.a + o.a, self.b + o.b, self.F)

    __radd__ = __add__

    def __neg__(self):
        return QF(-self.a, -self.b, self.F)

    def __sub__(self, o):
        return self + (-self._c(o))

    def __rsub__(self, o):
        return self._c(o) - self

    def __mul__(self, o):
        if not isinstance(o, QF):
            if isinstance(o, (int, Fraction)) or hasattr(o, "num"):
                return QF(self.a * o, self.b * o, self.F)
            o = self._c(o)
        F = self.F
        # y^2 = -(s y + 1)/x with s = x^2 - x - 1
        bb = self.b * o.b
        a = self.a * o.a - bb * F.invx
        b = self.a * o.b + self.b * o.a - bb * F.s * F.invx
        return QF(a, b, F)

    __rmul__ = __mul__

    def conj(self):
        """Galois conjugate y -> y' = -s/x - y."""
        F = self.F
        return QF(self.a - self.b * F.s * F.invx, -self.b, F)

    def norm(self):
        F = self.F
        return self.a * self.a - self.a * self.b * F.s * F.invx + self.b * self.b * F.invx

    def inverse(self):
        n = self.norm()
        c = self.conj()
        return QF(c.a / n, c.b / n, self.F)

    def __truediv__(self, o):
        if not isinstance(o, QF):
            return QF(self.a / o, self.b / o, self.F)
        return self * o.inverse()

    def __rtruediv__(self, o):
        return self._c(o) * self.inverse()

    def __pow__(self, n):
        if n < 0:
            return (self ** (-n)).inverse()
        r = QF(self.F.one, self.F.zero, self.F)
        for _ in range(n):
            r = r * self
        return r

    def is_rational(self) -> bool:
        return _is0(self.b)

    def __eq__(self, o):
        o = self._c(o)
        return self.a == o.a and self.b == o.b

    def __repr__(self):
        return f"({self.a}) + ({self.b})*y"


def _is0(v):
    return v.num == 0 if isinstance(v, RatFun) else v == 0


class _Field:
    def __init__(self, x):
        self.x = x
        self.one = x * 0 + 1
        self.zero = x * 0
        self.invx = self.one / x
        self.s = x * x - x - self.one


def field(x=None) -> _Field:
    """Base field: symbolic Q(x) (x=None) or Q with x a rational number."""
    return _Field(RatFun.x() if x is None else Fraction(x))


def gen_y(F: _Field) -> QF:
    return QF(F.zero, F.one, F)


# ----------------------------------------------------------------------------------------------
# the algorithm, generic over the coefficient field (QF exact, or acb numeric)
# ----------------------------------------------------------------------------------------------
def classical_D(x, y, r):
    """classicalD: delta_{r,2} + Li_{2-r}(y) + Li_{2-r}(x y)."""
    v = li_minus(r - 2, y) + li_minus(r - 2, x * y)
    return v + 1 if r == 2 else v


def quantum_D(x, y, r, k):
    """quantumD: B_{2r}(1/2) (Li_{-(2r+k-2)}(y) + Li_{-(2r+k-2)}(xy)) / (2r)!."""
    m = 2 * r + k - 2
    bh = bernoulli_half(2 * r)
    return (li_minus(m, y) + li_minus(m, x * y)) * bh.numerator / (bh.denominator * factorial(2 * r))


def exponent_terms(x, y, n, zero):
    """exponentTerms: p_j (j = 1..2n) as coefficient lists in z; p_j multiplies h^{j/2}."""
    ps = []
    for j in range(1, 2 * n + 1):
        deg = j + 2
        coeffs = [zero] * (deg + 1)
        coeffs[deg] = classical_D(x, y, j + 2) / factorial(j + 2)
        for r in range(1, (j + 2) // 4 + 1):
            kk = j - 4 * r + 2
            coeffs[kk] = coeffs[kk] + quantum_D(x, y, r, kk) / factorial(kk)
        if j == 2:
            coeffs[0] = coeffs[0] - (zero + 1) / 24
        ps.append(coeffs)
    return ps


def _pmul(a, b, zero):
    out = [zero] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        if _isz(u):
            continue
        for j, v in enumerate(b):
            if _isz(v):
                continue
            out[i + j] = out[i + j] + u * v
    return out


def _isz(u):
    if isinstance(u, QF):
        return _is0(u.a) and _is0(u.b)
    try:
        return u == 0
    except Exception:
        return False


def exp_coefficients(ps, zero, one):
    """expCoefficients: e_0 = 1, e_j = (1/j) sum_{k=1}^j k p_k e_{j-k}."""
    es = [[one]]
    for j in range(1, len(ps) + 1):
        acc = [zero]
        for k in range(1, j + 1):
            prod = _pmul(ps[k - 1], es[j - k], zero)
            if len(prod) > len(acc):
                acc = acc + [zero] * (len(prod) - len(acc))
            for i, v in enumerate(prod):
                acc[i] = acc[i] + v * k
        es.append([v / j for v in acc])
    return es


def gaussian_average(coeffs, delta, zero):
    """gaussianAverage with moments <z^{2k}> = (2k-1)!! (-1/delta)^k."""
    m = -1 / delta
    acc = zero
    pw = zero + 1
    for k in range(0, len(coeffs), 2):
        if k > 0:
            pw = pw * m
        c = coeffs[k]
        if not _isz(c):
            acc = acc + c * pw * double_factorial(k - 1)
    return acc


def hessian(x, y):
    return 1 + y / (1 - y) + x * y / (1 - x * y)


# ----------------------------------------------------------------------------------------------
# exact (generic x or rational x)
# ----------------------------------------------------------------------------------------------
def exact_coefficients(n: int, x=None) -> List[QF]:
    """a_0..a_n as elements of K = F[y]/(curve); F = Q(x) (x=None) or Q (x rational)."""
    F = field(x)
    y = gen_y(F)
    xv = QF(F.x, F.zero, F)
    zero = QF(F.zero, F.zero, F)
    one = QF(F.one, F.zero, F)
    ps = exponent_terms(xv, y, n, zero)
    es = exp_coefficients(ps, zero, one)
    d = hessian(xv, y)
    return [gaussian_average(es[2 * k], d, zero) for k in range(n + 1)]


def scaled_generic(n: int):
    """scaledGeneric: (y - 1/(x y))^{3k} a_k, which must be free of y (returned as RatFun in x)."""
    F = field(None)
    y = gen_y(F)
    xv = QF(F.x, F.zero, F)
    dd = y - 1 / (xv * y)
    out = []
    for k, a in enumerate(exact_coefficients(n, None)):
        s = a * dd ** (3 * k)
        if not s.is_rational():
            raise AssertionError(f"scaled coefficient {k} is not y-free: {s}")
        out.append(s.a)
    return out


def one_loop(x):
    """oneLoop numerator/(24 (y-1/(xy))^3) with sign: returns the y-free numerator part."""
    return -(x ** -3 - x ** -2 - 2 * x ** -1 + 15 - 2 * x - x ** 2 + x ** 3) / 24


def two_loop(x):
    return (x ** -6 - 2 * x ** -5 - 3 * x ** -4 + 610 * x ** -3 - 606 * x ** -2 - 1210 * x ** -1 + 3117
            - 1210 * x - 606 * x ** 2 + 610 * x ** 3 - 3 * x ** 4 - 2 * x ** 5 + x ** 6) / 1152


# ----------------------------------------------------------------------------------------------
# numeric at fixed x (flint acb)
# ----------------------------------------------------------------------------------------------
def saddle_root(x: acb, sign: int) -> acb:
    """saddleRoot: (1 + x - x^2 + sign sqrt(disc))/(2x), disc = (x^2-3x+1)(x^2+x+1), principal sqrt."""
    disc = (x * x - 3 * x + 1) * (x * x + x + 1)
    return (1 + x - x * x + sign * disc.sqrt()) / (2 * x)


def _to_acb(v) -> acb:
    if isinstance(v, acb):
        return v
    v = Fraction(v)
    return acb(v.numerator) / v.denominator


def numeric_coefficients(x, sign: int, n: int, digits: int = 240) -> List[acb]:
    """saddleCoefficients[x, sign, n, wp] with flint balls (prec ~ 3.33*digits + guard)."""
    old = ctx.prec
    ctx.prec = int(digits * 3.33) + 64 + 8 * n
    try:
        xx = _to_acb(x)
        yy = saddle_root(xx, sign)
        ps = exponent_terms(xx, yy, n, acb(0))
        ps = [acb_poly([c.mid() for c in p]) for p in ps]
        es = [acb_poly([1])]
        for j in range(1, 2 * n + 1):
            acc = acb_poly([0])
            for k in range(1, j + 1):
                acc += ps[k - 1] * es[j - k] * k
            # drop ball radii (working-precision semantics; validated by a precision repeat)
            es.append(acb_poly([c.mid() for c in (acc * (acb(1) / j)).coeffs()]))
        d = hessian(xx, yy)
        out = []
        for k in range(n + 1):
            out.append(gaussian_average(es[2 * k].coeffs(), d, acb(0)))
        return out
    finally:
        ctx.prec = old


def delta_ggmw(x):
    """GGMW (87): delta(x) = sqrt(-x^-2 + 2x^-1 + 1 + 2x - x^2) (principal)."""
    x = _to_acb(x)
    return (-1 / (x * x) + 2 / x + 1 + 2 * x - x * x).sqrt()


def sector_prefactor(x, sign: int, digits: int = 50) -> acb:
    """c_{0,s} = i x^{-1/2} / sqrt(i delta_s), with sqrt(i delta_g) = i|delta_g|^{1/2}, sqrt(i delta_c) = |delta_c|^{1/2}
    on the middle interval (Pengxu's branch convention; delta_s = S_vv)."""
    old = ctx.prec
    ctx.prec = int(digits * 3.33) + 32
    try:
        xx = _to_acb(x)
        yy = saddle_root(xx, sign)
        d = hessian(xx, yy)
        ad = abs(d).sqrt()
        root = acb(0, 1) * ad if sign > 0 else acb(ad)
        return acb(0, 1) / xx.sqrt() / root
    finally:
        ctx.prec = old
