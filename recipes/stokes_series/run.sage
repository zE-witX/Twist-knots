# kdb recipe stokes_series -- SageMath engine (independent implementation of algorithm.md).
# Usage:  sage run.sage '{"knot": "4_1", "from_cone": "I", "to_cone": "II", "order": 4, "x": "symbol"}'
# Prints {"entries": {"gg": [...], "gc": [...], "0g": [...], ...}} with q~-coefficients as strings in xt.
# Implemented: 4_1, cones I -> II (|q~| < 1), decorated blocks (GGMW (107)-(109) with squared theta),
#              S = M_II^{-1} M_I and its standard (LDU) form; P-frame on request.
import json, sys

try:
    PARAMS
except NameError:
    PARAMS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

P = PARAMS
N = int(P.get("order", 4))
xs = str(P.get("x", "symbol"))
if xs == "symbol":
    F = FractionField(QQ['xt'])
    X = F.gen()
else:
    F = QQ
    X = QQ(xs)
S.<qt> = PowerSeriesRing(F, default_prec=N + 1)

def O():
    return S(0).add_bigoh(N + 1)

def poch(a, s, n):
    """prod_{j=0}^{n-1} (1 - a qt^(s+j))"""
    r = S(1).add_bigoh(N + 1)
    for j in range(n):
        if s + j > N:
            break
        r *= (1 - a * qt^(s + j))
    return r

def poch_inf(a, s):
    return poch(a, s, max(0, N + 1 - s))

def krange(m):
    k = 0
    while True:
        e = k*(k+1)//2 + k*m
        if e > N and k > max(0, -m):
            return
        if e <= N:
            yield k, e
        k += 1

def Cb(m):
    return sum((-1)^k * qt^e / (poch(1/X, 0, k+1) * poch(X, 0, k+1)) for k, e in krange(m)) + O()

def Ab(m, x):
    return sum((-1)^k * qt^e * x^(k+m) / (poch(F(1), 1, k) * poch(x^2, 1, k)) for k, e in krange(m)) + O()

thp = poch_inf(X, 1) * poch_inf(1/X, 0)
thm = poch_inf(1/X, 1) * poch_inf(X, 0)
alpha = poch_inf(X^2, 1) / thp^2
beta = X * poch_inf(X^-2, 1) / thm^2

def calJ(m):
    return matrix(S, [[1, Cb(m), Cb(m+1)],
                      [0, alpha*Ab(m, X), alpha*Ab(m+1, X)],
                      [0, beta*Ab(m, 1/X), beta*Ab(m+1, 1/X)]])

J = calJ(-1)
PI = matrix(S, [[1,0,0],[0,0,-1],[0,1,-1]])
PII = matrix(S, [[1,0,0],[0,1,0],[0,1,-1]])
LII = matrix(S, [[1,0,0],[0,0,1],[0,1,0]])
MI = J * PI
MII = LII * J * PII
Smat = MII.inverse() * MI

S11 = Smat[1,1]
ent = {"gg": S11, "gc": Smat[1,2]/S11, "0g": Smat[0,1]/S11, "cg": Smat[2,1]/S11}
ent["0c"] = Smat[0,2] - ent["0g"]*Smat[1,2]
ent["cc"] = Smat[2,2] - ent["cg"]*Smat[1,2]
ent["00"] = Smat[0,0]
if P.get("frame") == "P":
    fac = {"0g": X - 1, "0c": 1 - 1/X, "gc": 1/X, "cg": X, "gg": 1, "cc": 1, "00": 1}
    ent = {k: v * fac[k] for k, v in ent.items()}

def cstr(c):
    return str(c).replace("**", "^")

out = {k: [cstr(v[i]) for i in range(N + 1)] for k, v in ent.items()}
print(json.dumps({"engine": "sage", "entries": out}))
