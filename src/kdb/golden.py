"""Golden-test evaluation (python reference engine) and comparison helpers shared with the Sage/Wolfram runners."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Dict, List

import yaml

from .data import repo_root


def load_golden(recipe: str) -> List[Dict]:
    f = repo_root() / "recipes" / recipe / "tests" / "golden.yaml"
    return yaml.safe_load(f.read_text())["cases"]


# ----------------------------------------------------------------------------------------------
# sympy helpers
# ----------------------------------------------------------------------------------------------
def sym(expr: str, names=("x", "xt", "eps", "z", "w", "n")):
    import sympy as sp
    loc = {n: sp.symbols(n) for n in names}
    loc["s"] = 1 + loc["xt"] + 1 / loc["xt"]      # s(xt) = xt + 1 + xt^-1 (GGMW (225))
    loc["I"] = sp.I
    loc["sqrt"] = sp.sqrt
    e = str(expr).replace("^", "**")
    return sp.sympify(e, locals=loc)


def same_rational(a: str, b: str) -> bool:
    import sympy as sp
    return sp.simplify(sym(a) - sym(b)) == 0


def num(expr: str, digits: int = 60):
    import mpmath
    import sympy as sp
    mpmath.mp.dps = digits + 10
    v = sym(expr)
    return mpmath.mpmathify(sp.N(v, digits + 10))


# ----------------------------------------------------------------------------------------------
# python engine
# ----------------------------------------------------------------------------------------------
def run_python(recipe: str, case: Dict) -> Dict:
    from .recipes import run_perturbative_series, run_stokes_series
    p = dict(case["params"])
    if recipe == "perturbative_series":
        return run_perturbative_series(p)
    p.pop("at_xt", None)
    return run_stokes_series(p)


def check_case(recipe: str, case: Dict, result: Dict | None = None) -> List[str]:
    """Return a list of failure messages (empty = pass) for the python engine."""
    chk = case["check"]
    fn = CHECKS.get(chk)
    if fn is None:
        return [f"unknown check {chk}"]
    return fn(case, result)


def _coeffs_ps(case, result):
    result = result or run_python("perturbative_series", case)
    sec = next(iter(result["sectors"].values()))
    return sec


def c_rational_functions(case, result=None):
    sec = _coeffs_ps(case, result)
    errs = []
    for n, (got, exp) in enumerate(zip(sec["coefficients"], case["expect"])):
        if not same_rational(got, exp):
            errs.append(f"h^{n}: got {got} expected {exp}")
    return errs


def c_rationals(case, result=None):
    sec = _coeffs_ps(case, result)
    return [f"h^{n}: {g} != {e}" for n, (g, e) in enumerate(zip(sec["coefficients"], case["expect"]))
            if Fraction(g) != Fraction(e)]


def c_P_eps(case, result=None):
    sec = _coeffs_ps(case, result)
    errs = []
    for m, exp in enumerate(case["expect"]):
        got = sec["P_m(eps)"][m]
        # got = (P)/(1 - p eps)^e with possibly reduced e; compare P_m Delta^{-2m-1}
        import sympy as sp
        eps = sp.symbols("eps")
        p = sec.get("_p")
        g = sym(got.replace("eps", "eps"))
        p_ = case.get("p")
        e = sym(exp)
        # recover p from the denominator string
        import re
        mm = re.search(r"\(1 - \((-?\d+)\)\*eps\)\^(\d+)", got)
        pp, ee = int(mm.group(1)), int(mm.group(2))
        if sp.simplify(g - e / (1 - pp * eps) ** (2 * m + 1)) != 0:
            errs.append(f"P_{m}: got {got}, expected ({exp})/Delta^{2 * m + 1}")
    return errs


def c_scaled_numerators(case, result=None):
    sec = _coeffs_ps(case, result)
    errs = []
    for n, (got, exp) in enumerate(zip(sec["coefficients"], case["expect"])):
        num_ = got.split(" / ")[0] if " / " in got else got
        if not same_rational(num_, exp):
            errs.append(f"h^{n}: numerator {num_} != {exp}")
    return errs


def c_numeric(case, result=None):
    sec = _coeffs_ps(case, result)
    tol = float(case.get("tol", 1e-30))
    errs = []
    for n, (got, exp) in enumerate(zip(sec["coefficients"], case["expect"])):
        d = abs(num(got) - num(exp))
        if d > tol:
            errs.append(f"h^{n}: |{got} - {exp}| = {d}")
    return errs


def c_private_phi_gc_series(case, result=None):
    from flint import acb, ctx
    from .private_inputs import load_phi_gc_series
    from .series.gauss41 import numeric_coefficients
    data = load_phi_gc_series()
    if data is None:
        return ["SKIP: private inputs not available"]
    n = int(case["params"]["order"])
    digits = int(case["params"]["digits"])
    tol = float(case.get("tol", 1e-200))
    errs = []
    keys = case["params"].get("points", list(data.keys()))
    anomalies = case.get("known_anomalies", {})
    for key in keys:
        x = Fraction(key)
        for root, sign in (("RootPlus", 1), ("RootMinus", -1)):
            if anomalies.get(key) == root:
                src = data[case["anomaly_copy_of"]][root]
                pre = lambda L: [(a[:40], b[:40]) for a, b in L]
                if pre(data[key][root]) != pre(src):
                    errs.append(f"x={key} {root}: documented anomaly changed (file corrected?) - update golden.yaml")
                continue
            ours = numeric_coefficients(x, sign, n, digits)
            ctx.prec = int(digits * 3.4) + 64
            worst = 0.0
            for a, (re_, im_) in zip(ours, data[key][root][: n + 1]):
                ref = acb(re_, im_)
                worst = max(worst, _relerr(a, ref))
            mp = data[key].get("MinimumStoredPrecision")
            tol_k = max(tol, 10.0 ** (-(mp - 3))) if mp else tol
            if worst > tol_k:
                errs.append(f"x={key} {root}: max rel. error {worst:.3e} (stored precision {mp})")
    return errs


def c_private_phi1x1d280(case, result=None):
    from flint import acb, ctx
    from .private_inputs import load_phi1x1d280
    from .series.gauss41 import numeric_coefficients
    ref = load_phi1x1d280()
    if ref is None:
        return ["SKIP: private inputs not available"]
    n = int(case["params"]["order"])
    digits = int(case["params"]["digits"])
    ours_g = numeric_coefficients(1, 1, n, digits)
    ours_c = numeric_coefficients(1, -1, n, digits)
    ctx.prec = int(digits * 3.4) + 64
    s3 = acb(3).sqrt()
    errs = []
    worst = 0.0
    for k in range(1, n + 1):
        r, has = ref[k - 1]
        v = acb(r.numerator) / r.denominator
        if has:
            v = v / s3
        g = v * (acb(0, 1) / 2) ** k        # referenceG = res (-pi)^n/(2 pi i)^n
        c = v * (acb(0, -1) / 2) ** k       # referenceC = res pi^n/(2 pi i)^n
        for a, b in ((ours_g[k], g), (ours_c[k], c)):
            worst = max(worst, _relerr(a, b))
    if worst > float(case.get("tol", 1e-200)):
        errs.append(f"max rel. error {worst:.3e}")
    return errs


def c_private_PDzlist(case, result=None):
    import sympy as sp
    from .private_inputs import load_PDzlist
    from .series.phi0 import mmr_eps
    order = int(case["params"]["order"])
    nterms = order // 2 + 1
    ref = load_PDzlist(nterms)
    if ref is None:
        return ["SKIP: private inputs not available"]
    P = mmr_eps(-1, order)
    z, eps = sp.symbols("z eps")
    errs = []
    for d in range(nterms):
        m = 2 * d
        Pm = sum(sp.Rational(int(c.p), int(c.q)) * eps ** i for i, c in enumerate(P[m].coeffs()))
        ours = (Pm / (1 + eps) ** (2 * m + 1)).subs(eps, -z ** 2)
        if sp.cancel(ours - ref[d]) != 0:
            errs.append(f"PDzlist[{d}] (h^{m}) differs")
    return errs


def c_private_hpoly2(case, result=None):
    import sympy as sp
    from .private_inputs import load_hpoly2
    from .series.phi0 import hpoly
    lmax = int(case["params"]["lmax"])
    ref = load_hpoly2(lmax)
    if ref is None:
        return ["SKIP: private inputs not available"]
    H = hpoly(-1, 2 * lmax)
    n = sp.symbols("n")
    errs = []
    for l in range(lmax + 1):
        ours = sum(sp.Rational(c.numerator, c.denominator) * n ** i for i, c in enumerate(H[2 * l]))
        if sp.expand(ours - ref[l]) != 0:
            errs.append(f"hpoly2[{l}] differs")
    return errs


def _relerr(a, b) -> float:
    from flint import arb
    d = abs(a - b).mid()
    m = abs(b).mid()
    den = m if m > 1 else arb(1)
    return float((d / den).mid())


def _refined_roots(coeffs, digits):
    from flint import acb, acb_poly
    f = acb_poly([acb(c) for c in coeffs])
    df = f.derivative()
    out = []
    for r in f.roots():
        r = r.mid()
        for _ in range(4 * digits):
            st = (f(r) / df(r)).mid()
            r = (r - st).mid()
        out.append(r)
    return out


def _eval_sym(expr, w):
    """Evaluate a sympy expression in w at an acb value, exactly in ball arithmetic."""
    import sympy as sp
    from flint import acb

    def ev(e):
        if e.is_Symbol:
            return w
        if e.is_Integer:
            return acb(int(e))
        if e.is_Rational:
            return acb(int(e.p)) / int(e.q)
        if e.is_Add:
            r = acb(0)
            for a in e.args:
                r += ev(a)
            return r
        if e.is_Mul:
            r = acb(1)
            for a in e.args:
                r *= ev(a)
            return r
        if e.is_Pow:
            return ev(e.args[0]) ** int(e.args[1])
        raise ValueError(e)

    return ev(sp.sympify(expr))


def _logs(a):
    from .series.saddle import log_series
    return log_series(a)


def c_twist_vs_41(case, result=None):
    from flint import acb, ctx
    from .series.gauss41 import numeric_coefficients
    from .series.twist import series
    P = case["params"]
    x = Fraction(P["x"])
    n, digits = int(P["order"]), int(P["digits"])
    res = series(int(P["twist"]), x, n, digits)
    refs = [numeric_coefficients(x, s, n, digits) for s in (1, -1)]
    ctx.prec = int(digits * 3.4)
    off = Fraction(case["offset"])
    errs = []
    for s, r in res:
        L = _logs(r.a)
        best = None
        for ref in refs:
            Lr = _logs(ref)
            dev = abs(L[1] - Lr[1] - acb(off.numerator) / off.denominator)
            for k in range(2, n + 1):
                dev = max(dev, abs(L[k] - Lr[k]))
            best = dev if best is None else min(best, dev)
        if float(best.mid()) > 1e-30:
            errs.append(f"sector A={s.A.str(5)}: deviation {float(best.mid()):.2e}")
    return errs


def c_twist_vs_52(case, result=None):
    from flint import acb, acb_poly, ctx
    from .series.twist import series
    P = case["params"]
    n, digits = int(P["order"]), int(P["digits"])
    res = series(int(P["twist"]), Fraction(P["x"]), n, digits)
    ctx.prec = int(digits * 3.4)
    xis = _refined_roots([1, -3, 2, -1], digits)

    def ggm(xi):
        return [acb(1),
                (-242 * xi ** 2 + 209 * xi - 454) / (4 * 23 ** 2),
                (12643 * xi ** 2 - 22668 * xi + 25400) / (32 * 23 ** 3),
                (-35443870 * xi ** 2 + 85642761 * xi - 164659509) / (2 ** 7 * 3 * 5 * 23 ** 5),
                (-1861268771 * xi ** 2 + 5671212343 * xi - 4655525678) / (2 ** 11 * 3 * 5 * 23 ** 6)]

    off = Fraction(case["offset"])
    errs = []
    for s, r in res:
        L = _logs(r.a)
        best = None
        for xi in xis:
            Lr = _logs(ggm(xi)[: n + 1])
            dev = abs(L[1] - Lr[1] - acb(off.numerator) / off.denominator)
            for k in range(2, n + 1):
                dev = max(dev, abs(L[k] - Lr[k]))
            best = dev if best is None else min(best, dev)
        if float(best.mid()) > 1e-30:
            errs.append(f"sector A={s.A.str(5)}: deviation {float(best.mid()):.2e}")
    return errs


def c_twist_a1_numberfield(case, result=None):
    from flint import acb, acb_poly, ctx
    from .series.twist import series
    P = case["params"]
    digits = int(P["digits"])
    res = series(int(P["twist"]), Fraction(P["x"]), 1, digits)
    ctx.prec = int(digits * 3.4)
    roots = _refined_roots(case["field"], digits)
    theirs = [_eval_sym(sym(case["a1"]), r) for r in roots]
    off = Fraction(case["offset"])
    offa = acb(off.numerator) / off.denominator
    errs = []
    used = set()
    for s, r in res:
        devs = [(float(abs(r.a[1] - t - offa).mid()), j) for j, t in enumerate(theirs) if j not in used]
        d, j = min(devs)
        used.add(j)
        if d > 1e-30:
            errs.append(f"sector A={s.A.str(5)}: a_1 deviation {d:.2e}")
    return errs


def c_twist_volumes(case, result=None):
    from flint import acb, ctx
    from .series.twist import saddles
    errs = []
    ctx.prec = 200

    def D(z):
        return z.polylog(2).imag + (1 - z).arg() * abs(z).log()

    for p_s, vols in case["expect"].items():
        p = int(p_s)
        got = set()
        for s in saddles(p, 1, 40):
            v = -D(s.A) + D(s.A * s.B) + D(s.A / s.B) - D(s.A) - D(s.A)
            got.add(round(abs(float(v.mid())), 8))
        for v in vols:
            if not any(abs(g - float(v)) < 1e-8 for g in got):
                errs.append(f"K_{p}: volume {v} not found in {sorted(got)}")
    return errs


# ---------------- stokes checks ----------------
def _st(case, result):
    from .recipes import run_stokes_series
    p = dict(case["params"])
    p.pop("at_xt", None)
    return result or run_stokes_series(p)


def _laurent_eq(got: str, exp: str, at1: bool = False) -> bool:
    import sympy as sp
    g, e = sym(got), sym(exp)
    if at1:
        g = g.subs(sp.symbols("xt"), 1)
    return sp.simplify(g - e) == 0


def c_laurent_series(case, result=None, at1=False):
    res = _st(case, result)
    errs = []
    for ent, exp in case["expect"].items():
        got = res["entries"][str(ent)]["coefficients"] if str(ent) in res["entries"] else None
        if got is None:
            errs.append(f"entry {ent} missing")
            continue
        for k, (g, e) in enumerate(zip(got, exp)):
            if not _laurent_eq(g, e, at1):
                errs.append(f"S_{ent} qt^{k}: got {g} expected {e}")
    if res.get("integral_laurent") is False:
        errs.append("non-integral Laurent coefficients")
    return errs


def c_laurent_series_at1(case, result=None):
    return c_laurent_series(case, result, at1=True)


def c_ggm24(case, result=None):
    import sympy as sp
    from .blocks import Blocks, Ring
    from .stokes.cones import stokes_matrix
    from .algebra import mat_mul
    N = int(case["params"]["order"])
    B = Blocks(Ring.symbolic(), N)
    S1 = stokes_matrix(B, "I", "II")
    S0 = stokes_matrix(B, "IV", "I")
    Sp = mat_mul(S1, S0)

    def at1(ser):
        from .algebra import TSeries
        return TSeries([Fraction(c(1)) for c in ser.coeffs(0)], ser.N, 0, Fraction(0))

    g = {(i, j): at1(Sp[i][j]) for i in (1, 2) for j in (1, 2)}
    s11 = g[(1, 1)]
    out = {"11": s11 - 1, "12": g[(1, 2)] / s11 - 3, "21": g[(2, 1)] / s11,
           "22": g[(2, 2)] - 1 - g[(1, 2)] * g[(2, 1)] / s11}
    errs = []
    for key, exp in case["expect"].items():
        got = [out[key][k] for k in range(1, N + 1)]
        if [str(c) for c in got] != [str(Fraction(e)) for e in exp]:
            errs.append(f"S+_{key}: got {[str(c) for c in got]} expected {exp}")
    return errs


def c_constant_matrix(case, result=None):
    res = _st(case, result)
    errs = []
    for i in range(3):
        for j in range(3):
            g = res["matrix"][i][j][0]
            if not _laurent_eq(g, case["expect"][i][j]):
                errs.append(f"[{i}][{j}]: {g} != {case['expect'][i][j]}")
    return errs


def c_pframe_entries(case, result=None):
    res = _st(case, result)
    return [f"S^P_{e}: {res['entries'][e]['coefficients'][0]} != {v}" for e, v in case["expect"].items()
            if not _laurent_eq(res["entries"][e]["coefficients"][0], v)]


def c_euler(case, result=None):
    from .blocks import Blocks, Ring
    from .stokes.cones import stokes_matrix, standard_form, euler_exponents_laurent, characters
    N = int(case["params"]["order"])
    B = Blocks(Ring.symbolic(), N)
    Sgg = standard_form(stokes_matrix(B, "I", "II"))["gg"]
    coeffs = [c.laurent() for c in Sgg.coeffs(0)]
    Om = euler_exponents_laurent(coeffs, N)
    errs = []
    at1 = [int(sum(d.values())) for d in Om]
    if at1 != case["expect"]["omega_at_1"][:N]:
        errs.append(f"Omega_m(1) = {at1}")
    for m, ch in case["expect"]["characters"].items():
        got = characters(Om[int(m) - 1])
        if got != {int(k): int(v) for k, v in ch.items()}:
            errs.append(f"characters of Omega_{m}: {got}")
    return errs


def c_splus_at1(case, result=None):
    res = _st(case, result)
    errs = []
    for k, M in enumerate(case["expect"]):
        for a in range(5):
            for b in range(5):
                g = res["matrix"][a][b][k]
                v = sym(g).subs(__import__("sympy").symbols("xt"), 1)
                if v != M[a][b]:
                    errs.append(f"qt^{k} [{a + 1}][{b + 1}]: {v} != {M[a][b]}")
    if res.get("integral_laurent") is False:
        errs.append("non-integral Laurent coefficients")
    return errs


def c_splus_generic(case, result=None):
    res = _st(case, result)
    errs = []
    for k, M in enumerate(case["expect"]):
        for a in range(5):
            for b in range(5):
                if not _laurent_eq(res["matrix"][a][b][k], M[a][b]):
                    errs.append(f"qt^{k} [{a + 1}][{b + 1}]: {res['matrix'][a][b][k]} != {M[a][b]}")
    return errs


def c_nz_a1_numberfield(case, result=None):
    from flint import acb, ctx
    from .series.integrands import nz_series
    P = case["params"]
    digits = int(P["digits"])
    res = nz_series(P["knot"], Fraction(P["x"]), 1, digits)
    ctx.prec = int(digits * 3.4)
    roots = _refined_roots(case["field"], digits)
    theirs = [_eval_sym(sym(case["a1"]), r) for r in roots]
    off = Fraction(case["offset"])
    offa = acb(off.numerator) / off.denominator
    errs, used = [], set()
    if len(res) != len(theirs):
        errs.append(f"{len(res)} critical points, expected {len(theirs)}")
    for Z, vol, r in res:
        devs = [(float(abs(r.a[1] - t - offa).mid()), j) for j, t in enumerate(theirs) if j not in used]
        d, j = min(devs)
        used.add(j)
        if d > 1e-30:
            errs.append(f"sector vol={vol.str(6)}: a_1 deviation {d:.2e}")
    return errs


def c_inverted_habiro(case, result=None):
    from .series.phi0 import inverted_habiro_coeff
    errs = []
    for item in case["expect"]:
        got = inverted_habiro_coeff(item["p"], item["n"], item["Q"], inverse_nome=item.get("inverse", False))
        exp = {int(k): v for k, v in item["series"].items()}
        if got != exp:
            errs.append(f"p={item['p']} n={item['n']}: {sorted(got.items())[:8]} != {sorted(exp.items())[:8]}")
    return errs


def c_finv(case, result=None):
    import sympy as sp
    from .series.phi0 import inverted_habiro_coeff
    x, q = sp.symbols("x q")
    errs = []
    for p_s, exp in case["expect"].items():
        p = int(p_s)
        Q = len(exp) - 2
        tot = 0
        for n in range(Q + 3):
            a = inverted_habiro_coeff(p, n, Q + 3 * n + 6, inverse_nome=p < 0)
            A = sum(c * q ** e for e, c in a.items())
            den = 1
            for j in range(1, n + 1):
                den *= (x + 1 / x - q ** j - q ** (-j))
            tot += -A / den
        ser = sp.series(tot, q, 0, Q + 1).removeO()
        for k, e in zip(range(-1, Q + 1), exp):
            if sp.simplify(ser.coeff(q, k) - sym(e)) != 0:
                errs.append(f"K_{p}: q^{k} coefficient {sp.factor(ser.coeff(q, k))} != {e}")
    return errs


def c_index_x1(case, result=None):
    from .recipes import run_index3d
    errs = []
    for k, exp in case["expect"].items():
        got = [int(v) for v in run_index3d({"knot": k, "order": int(case["params"]["order"]), "flux": [0]})["fluxes"]["0"]["at_x_1"]]
        if got[: len(exp)] != exp:
            errs.append(f"{k}: {got} != {exp}")
    return errs


def c_index_x_graded(case, result=None):
    import sympy as sp
    from .recipes import run_index3d
    P = case["params"]
    got = run_index3d({"knot": P["knot"], "order": int(P["order"]), "flux": [0]})["fluxes"]["0"]["Ind_rot"]
    return [f"q^{n}: {g} != {e}" for n, (g, e) in enumerate(zip(got, case["expect"])) if sp.simplify(sym(g) - sym(e)) != 0]


CHECKS = {
    "index_x1": c_index_x1,
    "index_x_graded": c_index_x_graded,
    "nz_a1_numberfield": c_nz_a1_numberfield,
    "inverted_habiro": c_inverted_habiro,
    "finv": c_finv,
    "splus_at1": c_splus_at1,
    "splus_generic": c_splus_generic,
    "rational_functions": c_rational_functions,
    "rationals": c_rationals,
    "P_eps": c_P_eps,
    "scaled_numerators": c_scaled_numerators,
    "numeric": c_numeric,
    "private_phi_gc_series": c_private_phi_gc_series,
    "private_phi1x1d280": c_private_phi1x1d280,
    "private_PDzlist": c_private_PDzlist,
    "private_hpoly2": c_private_hpoly2,
    "twist_vs_41": c_twist_vs_41,
    "twist_vs_52": c_twist_vs_52,
    "twist_a1_numberfield": c_twist_a1_numberfield,
    "twist_volumes": c_twist_volumes,
    "laurent_series": c_laurent_series,
    "laurent_series_at1": c_laurent_series_at1,
    "ggm24": c_ggm24,
    "constant_matrix": c_constant_matrix,
    "pframe_entries": c_pframe_entries,
    "euler": c_euler,
}
