"""The two first-release recipes: ``perturbative_series`` and ``stokes_series``.

Each ``run_*`` takes a plain parameter dict and returns a result dict whose numbers are strings (exact
objects in sympy/Mathematica-parseable text, numerics with their digits).  ``kdb.bundle`` writes it out.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List

from flint import acb, arb, ctx

from . import __version__
from .algebra import RatFun, laurent_to_str
from .data import load_knot

# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------


def parse_x(x):
    """'symbol' -> None; '6/5' -> Fraction."""
    if x is None or str(x).lower() in ("symbol", "symbolic", "x"):
        return None
    return Fraction(str(x))


def acb_str(v: acb, digits: int) -> str:
    """Mathematica/sympy-parseable complex number with `digits` significant digits (radius dropped)."""
    v = v.mid()
    re, im = v.real, v.imag
    # components below the printed precision are numerical zeros
    scale = max(abs(v).mid(), arb(1)) if abs(v).mid() > 0 else arb(1)
    cut = scale * arb(10) ** (-(digits + 5))
    if abs(re) < cut:
        re = arb(0)
    if abs(im) < cut:
        im = arb(0)

    def s(a: arb):
        if a == 0:
            return "0"
        t = a.str(digits, radius=False)
        return t.replace("e", "*10^")

    if im == 0:
        return s(re)
    if re == 0:
        return f"({s(im)})*I"
    return f"{s(re)} + ({s(im)})*I"


def frac_str(c: Fraction) -> str:
    return str(Fraction(c))


# ----------------------------------------------------------------------------------------------
# classical data of 4_1
# ----------------------------------------------------------------------------------------------
def classical_41(x, digits=50) -> Dict[str, str]:
    """V^g via GGMW (89) and via S(u, v_g) + pi^2/6 + pi i u, delta_g = S_vv, a_g = xi, gamma(x)."""
    from .series.gauss41 import delta_ggmw, hessian, saddle_root, _to_acb
    old = ctx.prec
    ctx.prec = int(digits * 3.4) + 32
    try:
        xx = _to_acb(x)
        u = xx.log()
        pi = acb.pi()
        I = acb(0, 1)
        d = delta_ggmw(xx)
        t = (-1 - 1 / xx + xx - I * d) / 2
        lt = t.log()
        A89 = lt * lt / 2 + 2 * lt * u + u * u + (-t * xx).polylog(2) + (-t).polylog(2) + pi * pi / 6 + pi * I * u
        yg = saddle_root(xx, 1)
        v = (-yg).log()
        S = u * u + (-v.exp()).polylog(2) + (-(u + v).exp()).polylog(2) + v * v / 2 + 2 * u * v
        Vg = S + pi * pi / 6 + pi * I * u
        xi = (S + pi * pi / 6) / (2 * pi * I)
        gamma = 1 / (xx.sqrt() - 1 / xx.sqrt())
        dg = hessian(xx, yg)
        dc = hessian(xx, saddle_root(xx, -1))
        out = {
            "V_g_from_89": acb_str(A89, digits),
            "V_g_from_S": acb_str(Vg, digits),
            "V_c": acb_str(-Vg, digits),
            "agreement_89_vs_S": acb_str(A89 - Vg, 5),
            "delta_g": acb_str(dg, digits),
            "delta_c": acb_str(dc, digits),
            "delta_g_plus_delta_c": acb_str(dg + dc, 5),
            "a_g (xi)": acb_str(xi, digits),
            "a_c": acb_str(-xi, digits),
        }
        if x != 1:
            out["gamma(x)"] = acb_str(gamma, digits)
        return out
    finally:
        ctx.prec = old


# ----------------------------------------------------------------------------------------------
# perturbative_series
# ----------------------------------------------------------------------------------------------
SECTOR_ALIASES_41 = {"g": 1, "sigma1": 1, "s1": 1, "c": -1, "sigma2": -1, "s2": -1}


def run_perturbative_series(params: Dict) -> Dict:
    knot = load_knot(params.get("knot"), params.get("twist"))
    p = knot.p
    sector = str(params.get("sector", "0"))
    order = int(params.get("order", 8))
    x = parse_x(params.get("x", "symbol"))
    frame = params.get("frame", "standard")
    digits = int(params.get("digits", 60))
    method = params.get("method", "auto")
    res: Dict = {
        "recipe": "perturbative_series",
        "knot": knot.id,
        "twist_p": p,
        "group": {"family": "SL", "N": 2},
        "side": "cs",
        "params": {"sector": sector, "order": order, "x": "symbol" if x is None else str(x),
                   "frame": frame, "digits": digits, "method": method},
        "conventions": {
            "variable": "h = 2*pi*I*tau; coefficients are h-coefficients (tau-coefficients = a_n (2 pi i)^n)",
            "q": "q = exp(h), x = exp(u)",
        },
        "version": __version__,
        "sectors": {},
    }
    if sector in ("0", "trivial", "sigma0"):
        res["sectors"]["0"] = _phi0(p, order, x, method, frame)
    elif p == -1 and sector in SECTOR_ALIASES_41 and method in ("auto", "gauss41"):
        res["sectors"][sector] = _phi_gc_41(SECTOR_ALIASES_41[sector], order, x, digits, frame)
    elif sector in ("all", "nonabelian") or sector.startswith("s") or sector in SECTOR_ALIASES_41:
        if x is None:
            raise ValueError("non-trivial sectors of K_p are computed at a rational x (use --x 1, --x 6/5, ...)")
        res["sectors"].update(_phi_twist(knot, p, order, x, digits, sector))
    else:
        raise ValueError(f"unknown sector {sector!r}")
    return res


def _phi0(p: int, order: int, x, method: str, frame: str) -> Dict:
    from .series.phi0 import mmr_eps, minimal_denominator, phi0_x1, eps_to_x_laurent
    import sympy as sp
    out = {"name": "trivial", "exponential": {"V": "0", "a": "0"},
           "method": "colored Jones / Habiro, MMR resummation (eps-route)", "status": "known (exact)"}
    if frame == "P":
        out["frame"] = "P: P_0 = phi_0"
    if x == 1:
        cs = phi0_x1(p, order)
        out["coefficients"] = [frac_str(c) for c in cs]
        out["series_text"] = " + ".join(f"({c})*h^{i}" for i, c in enumerate(cs) if c)
        return out
    P = mmr_eps(p, order)
    eps, z, xs = sp.symbols("eps z x")
    coeffs, zforms, epsforms, dens = [], [], [], []
    for m, Pm in enumerate(P):
        Pm2, e = minimal_denominator(p, Pm, m)
        pe = sp.Integer(0) + sum(sp.Rational(int(c.p), int(c.q)) * eps ** i for i, c in enumerate(Pm2.coeffs()))
        Rm_eps = pe / (1 - p * eps) ** e
        epsforms.append(f"({sp.sstr(sp.expand(pe))})/(1 - ({p})*eps)^{e}".replace("**", "^"))
        zforms.append((sp.sstr(sp.expand(pe.subs(eps, -z ** 2))) + f" / (1 + ({p})*z^2)^{e}").replace("**", "^"))
        dens.append(e)
        if x is None:
            Rx = sp.factor(sp.together(Rm_eps.subs(eps, 2 - xs - 1 / xs)))
            coeffs.append(sp.sstr(Rx).replace("**", "^"))
        else:
            val = Rm_eps.subs(eps, 2 - sp.Rational(x.numerator, x.denominator) - sp.Rational(x.denominator, x.numerator))
            coeffs.append(sp.sstr(sp.nsimplify(val)))
    out["coefficients"] = coeffs
    out["P_m(eps)"] = epsforms
    out["R_m(z)"] = zforms
    out["denominator_powers"] = dens
    out["notes"] = "eps = 2 - x - 1/x = -z^2, z = x^(1/2) - x^(-1/2); Delta_{K_p} = 1 - p*eps = 1 + p*z^2"
    return out


def _phi_gc_41(sign: int, order: int, x, digits: int, frame: str) -> Dict:
    from .series import gauss41 as G
    name = "g" if sign > 0 else "c"
    out = {"name": "geometric" if sign > 0 else "conjugate",
           "method": "formal Gaussian expansion of the new state integral (porting guide phi_gc_gaussian)",
           "normalisation": "Phi_s = exp(V_s/h) * i x^(-1/2)/sqrt(i delta_s) * F_s(h); F_s = sum a_n h^n incl. exp(-h/24)",
           "status": "known"}
    if x is None:
        if order > 6:
            raise ValueError("symbolic-x closed forms are produced through h^6 (as in the notebook); use a numeric x")
        import sympy as sp
        xs = sp.symbols("x")
        scaled = G.scaled_generic(order)
        sgn = 1 if sign > 0 else -1
        forms = []
        for n, s in enumerate(scaled):
            num = s.to_str("x")
            forms.append(f"({num}) / ({'+' if sgn > 0 else '-'}(y_g - y_c))^{3 * n}" if n else "1")
        out["coefficients"] = forms
        out["notes"] = ("y_g, y_c = (1 + x - x^2 +- sqrt((x^2-3x+1)(x^2+x+1)))/(2x) (middle interval: g = +); "
                        "y_g - y_c = I*delta(x) with delta(x) of GGMW (87); scaled numerators (y - 1/(xy))^(3n) a_n")
        out["prefactor"] = "c_0 = I*x^(-1/2)/sqrt(I*delta_s), delta_s = S_vv = I*delta(x)/x (g), -I*delta(x)/x (c)"
        return out
    cs = G.numeric_coefficients(x, sign, order, digits)
    out["coefficients"] = [acb_str(c, digits) for c in cs]
    out["digits"] = digits
    pref = G.sector_prefactor(x, sign, digits)
    out["prefactor_c0"] = acb_str(pref, digits)
    cl = classical_41(x, digits)
    out["exponential"] = {"V": cl["V_g_from_S"] if sign > 0 else cl["V_c"],
                          "a": cl["a_g (xi)"] if sign > 0 else cl["a_c"]}
    out["classical"] = cl
    if frame == "P":
        out["frame"] = ("P: P_s = tau^(-1/2) exp(a_s/tau) gamma(x) phi_s, gamma = 1/(x^(1/2) - x^(-1/2)), "
                        "phi_s = c_0 * F_s;  a_g = xi = (S(u,v_g) + pi^2/6)/(2 pi i), a_c = -xi")
    return out


def _phi_twist(knot, p: int, order: int, x, digits: int, which: str) -> Dict:
    from .series.twist import series, cyclotomic_action
    from .series.saddle import log_series
    old = ctx.prec
    out = {}
    res = series(p, x, order, digits)
    ctx.prec = int(digits * 3.4) + 64

    def D(z):
        return z.polylog(2).imag + (1 - z).arg() * abs(z).log()

    rows = []
    for s, r in res:
        xa = acb(x.numerator) / x.denominator
        vol = -D(s.A) + D(s.A * s.B) + D(s.A / s.B) - D(xa * s.A) - D(s.A / xa)
        rows.append((s, r, vol))
    # label: geometric = largest volume, then conjugate, then by decreasing |vol|, real (vol = 0) last
    def key(t):
        v = t[2].mid()
        return (abs(v) < 1e-20, -abs(v), -v)
    rows.sort(key=key)
    for i, (s, r, vol) in enumerate(rows):
        sid = f"s{i + 1}"
        if which not in ("all", "nonabelian") and which != sid:
            continue
        W = cyclotomic_action(p, x, s.A, s.B)
        out[sid] = {
            "name": "nonabelian",
            "method": "formal 2d Gaussian expansion of Marino's I_{K_p}(z,w) (Marino 5.4) at the critical point",
            "status": "ours (normalised a_0 = 1)",
            "critical_point": {"A": acb_str(s.A, 30), "B": acb_str(s.B, 30)},
            "volume_bloch_wigner": acb_str(acb(vol), 20),
            "W_cyclotomic_principal_branch": acb_str(W, 30),
            "det_hessian": acb_str(r.detH, 30),
            "coefficients": [acb_str(c, digits) for c in r.a],
            "log_coefficients": [acb_str(c, digits) for c in log_series(r.a)],
            "normalisation": ("fluctuation series normalised to a_0 = 1; agrees with published/notebook series up to "
                              "a sector-independent factor exp(c h) recorded in the knot file (twist_engine_offset)"),
        }
    ctx.prec = old
    return out


# ----------------------------------------------------------------------------------------------
# stokes_series
# ----------------------------------------------------------------------------------------------
ENTRIES = ("0g", "0c", "gc", "cg", "gg", "cc", "00")


def run_stokes_series(params: Dict) -> Dict:
    from .blocks import Blocks, Ring
    from .stokes.cones import stokes_matrix, standard_form, p_frame_factor, default_order
    knot = load_knot(params.get("knot"), params.get("twist"))
    if knot.p != -1:
        raise NotImplementedError(f"stokes_series is implemented for 4_1 only (requested {knot.id}); see REPORT")
    frm, to = params.get("from_cone", "I"), params.get("to_cone", "II")
    entry = params.get("entry", "all")
    order = int(params.get("order", 6))
    x = parse_x(params.get("x", "symbol"))
    frame = params.get("frame", "standard")
    variant = params.get("variant", "corrected")
    R = Ring.symbolic() if x is None else Ring.at(x)
    B = Blocks(R, order)
    S = stokes_matrix(B, frm, to, variant)
    var = "qt" if (frm, to) in (("I", "II"),) else ("1/qt" if (frm, to) == ("III", "IV") else "constant")
    gauss = params.get("gauss", default_order(frm, to))
    sf = standard_form(S, gauss)
    entries = ENTRIES if entry == "all" else (entry,)
    out_entries = {}
    integral = True
    for e in entries:
        ser = sf[e]
        if frame == "P":
            ser = ser * p_frame_factor(R, e)
        cs = ser.coeffs(0)
        strs = []
        for c in cs:
            if isinstance(c, RatFun):
                strs.append(c.to_str("xt"))
                integral &= c.is_integral_laurent()
            else:
                strs.append(str(c))
                integral &= (Fraction(c).denominator == 1) if x == 1 else True
        out_entries[e] = {"coefficients": strs, "variable": var, "order": ser.N}
    matrix = [[[ (c.to_str("xt") if isinstance(c, RatFun) else str(c)) for c in S[i][j].coeffs(0)] for j in range(3)]
              for i in range(3)]
    return {
        "recipe": "stokes_series",
        "knot": knot.id,
        "group": {"family": "SL", "N": 2},
        "side": "cs",
        "params": {"from_cone": frm, "to_cone": to, "entry": entry, "order": order,
                   "x": "symbol" if x is None else str(x), "frame": frame, "variant": variant},
        "conventions": {
            "definition": "S_{R->R'} = M_{R'}^{-1} M_R, M_R = L_R calJ_{-1}(xt,qt) P_R (decorated blocks, squared theta)",
            "standard_form": ("LDU: S = [[1,S0g,0],[0,1,0],[0,Scg,1]] diag(1,Sgg,Scc) [[1,0,S0c],[0,1,Sgc],[0,0,1]]" if gauss == "LDU"
                              else "UDL: S = [[1,0,S0c],[0,1,Sgc],[0,0,1]] diag(1,Sgg,Scc) [[1,S0g,0],[0,1,0],[0,Scg,1]]"),
            "frame": "Delta-normalised (GGMW (120))" if frame != "P" else "P-frame: S^P = E S E^-1, E = D Delta^-1",
            "P_III_P_IV": "entry 1 (corrected)" if variant == "corrected" else "entry xt^-1 - 1 (GGMW (118c,d) as printed)",
            "variables": "qt = exp(-2 pi i/tau), xt = exp(u/tau)",
        },
        "entries": out_entries,
        "matrix": matrix,
        "integral_laurent": bool(integral) if x is None else None,
        "version": __version__,
    }
