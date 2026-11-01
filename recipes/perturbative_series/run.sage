# kdb recipe perturbative_series -- SageMath engine (independent implementation of algorithm.md).
# Usage:  sage run.sage '{"knot": "4_1", "sector": "0", "order": 4, "x": "symbol"}'
# Prints one JSON object {"coefficients": [...]} (strings parseable by sympy / Mathematica).
# Implemented: trivial sector phi_0 of any twist knot K_p (Habiro/MMR, eps-route; x = 1 exact),
#              sectors g, c of 4_1 (formal Gaussian expansion; exact generic x through h^6, numeric fixed x).
# Not ported: the 2d I_{K_p} saddle engine for non-trivial sectors of K_p, p != -1 (python only).
import json, sys

try:
    PARAMS
except NameError:
    PARAMS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

TWIST = {"4_1": -1, "5_2": 2, "6_1": -2, "7_2": 3, "8_1": -3, "9_2": 4, "10_1": -4, "3_1": 1}

R.<q> = LaurentPolynomialRing(QQ)

def qpoch(k):
    r = R(1)
    for j in range(1, k + 1):
        r *= (1 - q^j)
    return r

def masbaum_C(p, n):
    """C^p_n(q) (Marino 5.4 / CLAUDE.md 3.0) as a Laurent polynomial."""
    s = 0
    for k in range(n + 1):
        s += (-1)^k * q^(p*k*(k+1) + k*(k-1)//2) * (q^(2*k+1) - 1) * qpoch(n) / (qpoch(n+k+1) * qpoch(n-k))
    s = -q^n * s
    num, den = s.numerator(), s.denominator()
    return R(num) / R(den)

def h_expand(f, L):
    """Coefficients h^0..h^L of f(e^h) for a Laurent polynomial f (notebook Dq)."""
    f = R(f)
    out = []
    for l in range(L + 1):
        out.append(sum(c * e^l for e, c in zip(f.exponents(), f.coefficients())) / factorial(l))
    return out

def phi0_x1(p, L):
    out = [QQ(0)] * (L + 1)
    for n in range(L // 2 + 1):
        g = masbaum_C(p, n) * qpoch(n)^2
        for l, v in enumerate(h_expand(g, L)):
            out[l] += v
    return out

def mmr_eps(p, M, extra=6):
    """P_m(eps), phi_0 = sum_m P_m(eps) h^m / (1 - p eps)^(2m+1)  (eps = 2 - x - 1/x)."""
    A = 2*M + extra
    Pe.<eps> = QQ[]
    nmax = A + M//2 + 1
    def trunc(f):
        return Pe(f.list()[:A+1])
    total = [Pe(0)] * (M + 1)
    prod = [Pe(1)] + [Pe(0)] * M
    for n in range(nmax + 1):
        if n >= 1:
            s = [QQ(0)] * (M + 1)
            for k in range(1, M//2 + 1):
                s[2*k] = 2 * n^(2*k) / factorial(2*k)
            new = [trunc(eps * prod[m]) for m in range(M + 1)]
            for m in range(M + 1):
                for k2 in range(2, m + 1, 2):
                    new[m] += prod[m - k2] * s[k2]
            prod = [trunc(f) for f in new]
        H = masbaum_C(p, n) * q^(n*(n+1)//2)
        hc = h_expand(H, M)
        for m in range(M + 1):
            total[m] += sum(prod[m - k] * hc[k] for k in range(m + 1))
    out = []
    for m in range(M + 1):
        f = trunc(total[m] * (1 - p*eps)^(2*m + 1))
        assert all(c == 0 for c in f.list()[2*m+1:]), "MMR form violated"
        out.append(Pe(f.list()[:2*m+1]))
    return out

def phi0_symbolic(p, M):
    X.<x> = FractionField(QQ['x'])
    out = []
    for m, P in enumerate(mmr_eps(p, M)):
        e = 2 - x - 1/x
        r = P(e) / (1 - p*e)^(2*m + 1)
        out.append(str(r.factor()) if r != 0 else "0")
    return out

# ---------------- 4_1, sectors g and c ----------------
def stirling_coeffs(m):
    return [factorial(k) * stirling_number2(m + 1, k + 1) for k in range(m + 1)]

def li_minus(m, t):
    w = t / (1 - t)
    return sum(c * w^(k + 1) for k, c in enumerate(stirling_coeffs(m)))

def bern_half(n):
    return (2^(1 - n) - 1) * bernoulli(n)

def gauss41(F, xx, yy, n):
    """a_0..a_n (notebook steps 1-3) with coefficients in the field F containing x, y."""
    Pz.<z> = F[]
    ps = []
    for j in range(1, 2*n + 1):
        p = (li_minus(j, yy) + li_minus(j, xx*yy)) * z^(j + 2) / factorial(j + 2)
        for r in range(1, (j + 2)//4 + 1):
            k = j - 4*r + 2
            m = 2*r + k - 2
            p += bern_half(2*r) * (li_minus(m, yy) + li_minus(m, xx*yy)) / factorial(2*r) * z^k / factorial(k)
        if j == 2:
            p -= F(1)/24
        ps.append(p)
    es = [Pz(1)]
    for j in range(1, 2*n + 1):
        es.append(sum(k * ps[k-1] * es[j-k] for k in range(1, j + 1)) / j)
    d = 1 + yy/(1 - yy) + xx*yy/(1 - xx*yy)
    out = []
    for k in range(n + 1):
        c = es[2*k].list()
        out.append(sum(c[i] * (-1/d)^(i//2) * prod(range(i - 1, 0, -2)) for i in range(0, len(c), 2)))
    return out

def gc_symbolic(n, sign):
    K0.<x> = FractionField(QQ['x'])
    Ky.<Y> = K0[]
    K.<y> = K0.extension(x*Y^2 + (x^2 - x - 1)*Y + 1)
    a = gauss41(K, K(x), y, n)
    dd = y - 1/(x*y)
    out = []
    for k, ak in enumerate(a):
        s = ak * dd^(3*k)
        v = s.list() if hasattr(s, "list") else [s]
        assert all(c == 0 for c in v[1:]), "scaled coefficient not y-free"
        out.append(str(v[0]) if k == 0 else f"({v[0]}) / ({'+' if sign > 0 else '-'}(y_g - y_c))^{3*k}")
    return out

def gc_numeric(xv, n, sign, digits):
    C = ComplexField(int(digits * 3.33) + 64 + 8*n)
    xx = C(QQ(xv))
    disc = (xx^2 - 3*xx + 1) * (xx^2 + xx + 1)
    yy = (1 + xx - xx^2 + sign * disc.sqrt()) / (2*xx)
    a = gauss41(C, xx, yy, n)
    def fmt(v):
        re, im = v.real(), v.imag()
        s = lambda r: r.str(digits=digits).replace("e", "*10^")
        return f"{s(re)} + ({s(im)})*I"
    return [fmt(v) for v in a]

P = PARAMS
knot = P.get("knot")
p = int(P["twist"]) if "twist" in P else TWIST[knot]
sector = str(P.get("sector", "0"))
order = int(P.get("order", 4))
x = str(P.get("x", "symbol"))
res = {"engine": "sage", "knot": knot, "twist_p": int(p), "sector": sector}
if sector == "0":
    if x == "1":
        res["coefficients"] = [str(c) for c in phi0_x1(p, order)]
    elif x == "symbol":
        res["coefficients"] = phi0_symbolic(p, order)
    else:
        X.<xs> = QQ[]
        xv = QQ(x)
        res["coefficients"] = [str(QQ(SR(c).subs(x=xv))) for c in phi0_symbolic(p, order)]
elif p == -1 and sector in ("g", "c"):
    sign = 1 if sector == "g" else -1
    if x == "symbol":
        res["coefficients"] = gc_symbolic(order, sign)
    else:
        res["coefficients"] = gc_numeric(x, order, sign, int(P.get("digits", 40)))
else:
    res["error"] = "not implemented in the Sage engine (python only)"
print(json.dumps(res))
