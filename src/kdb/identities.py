"""Identities and cross-checks of CLAUDE.md section 3 (run by ``kdb check`` and by pytest).

Each function returns a list of failure messages (empty = pass).  ``QUICK`` lists the cheap ones.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, List


def id_V_from_89_vs_S() -> List[str]:
    """3.1: V^g from GGMW (89) equals S(u, v_g) + pi^2/6 + pi i u; at x = 1 it is i Vol(4_1)."""
    from flint import acb, ctx
    from .recipes import classical_41
    errs = []
    for x in (Fraction(1), Fraction(6, 5), Fraction(9, 10), Fraction(3, 2)):
        cl = classical_41(x, 40)
        d = cl["agreement_89_vs_S"]
        from .golden import num
        if abs(num(d)) > 1e-30:
            errs.append(f"x={x}: (89) - S = {d}")
    cl = classical_41(Fraction(1), 30)
    from .golden import num
    if abs(num(cl["V_g_from_S"]) - num("2.029883212819307250042405108549040571883378*I")) > 1e-25:
        errs.append(f"V^g(1) = {cl['V_g_from_S']} != i Vol(4_1)")
    return errs


def id_delta_g_minus_c() -> List[str]:
    """3.1: delta_g = -delta_c (S_vv at the two saddles), and delta_g = i delta(x)/x with GGMW (87)."""
    from .recipes import classical_41
    from .golden import num
    from .series.gauss41 import delta_ggmw
    from flint import acb, ctx
    errs = []
    for x in (Fraction(1), Fraction(6, 5), Fraction(1, 2)):
        cl = classical_41(x, 40)
        if abs(num(cl["delta_g_plus_delta_c"])) > 1e-30:
            errs.append(f"x={x}: delta_g + delta_c = {cl['delta_g_plus_delta_c']}")
    return errs


def id_conjugate_sector() -> List[str]:
    """phi_gc guide, checks: a_n^(c) = (-1)^n a_n^(g) (exact, generic x, through h^4)."""
    from .series.gauss41 import exact_coefficients
    errs = []
    for n, a in enumerate(exact_coefficients(4, None)):
        if not (a.conj() == a * ((-1) ** n)):
            errs.append(f"h^{n}: conjugate symmetry fails")
    return errs


def id_scaled_y_free() -> List[str]:
    """phi_gc guide step 4: (y - 1/(xy))^{3n} a_n is free of y (curveResidual = 0), n <= 4."""
    from .series.gauss41 import scaled_generic
    try:
        scaled_generic(4)
    except AssertionError as e:
        return [str(e)]
    return []


def id_S_identity_a_b() -> List[str]:
    """3.5: S_{I->II} = M_II^{-1} M_I (a) equals the middle-interval form (b), and the closed forms of S_gg, S_gc, S_0g."""
    from .blocks import Blocks, Ring
    from .stokes.cones import stokes_matrix, stokes_middle_form_b, standard_form, middle_formulas
    errs = []
    for R in (Ring.symbolic(), Ring.at(2), Ring.at(Fraction(3, 7))):
        B = Blocks(R, 5)
        Sa = stokes_matrix(B, "I", "II")
        Sb = stokes_middle_form_b(B)
        for i in range(3):
            for j in range(3):
                d = Sa[i][j] - Sb[i][j]
                if not all(c == 0 for c in d.coeffs(0)):
                    errs.append(f"{R.name}: (a) != (b) at [{i}][{j}]")
        sf = standard_form(Sa)
        mf = middle_formulas(B)
        for k in ("gg", "gc", "0g"):
            d = sf[k] - mf[k]
            if not all(c == 0 for c in d.coeffs(0)):
                errs.append(f"{R.name}: matrix route != closed form for S_{k}")
        if not all(c == 0 for c in (sf["0g"] - sf["0c"]).coeffs(0)):
            errs.append("S_0g != S_0c")
        if not all(c == 0 for c in (sf["cg"] + sf["gc"]).coeffs(0)):
            errs.append("S_cg != -S_gc")
        if not all(c == 0 for c in (sf["cc"] * sf["gg"] - 1).coeffs(0)):
            errs.append("S_cc != 1/S_gg")
    return errs


def id_integrality() -> List[str]:
    """3.5(e): all entries of S_{I->II}, S_{III->IV} (Delta- and P-frame) lie in Z[xt^{+-1}] (to qt^6); S -> 1 at qt -> 0."""
    from .blocks import Blocks, Ring
    from .stokes.cones import stokes_matrix, standard_form, p_frame_factor, default_order
    errs = []
    R = Ring.symbolic()
    B = Blocks(R, 6)
    for frm, to in (("I", "II"), ("III", "IV")):
        S = stokes_matrix(B, frm, to)
        for i in range(3):
            for j in range(3):
                c0 = S[i][j][0]
                if c0 != (1 if i == j else 0):
                    errs.append(f"{frm}->{to}: S[{i}][{j}] at qt^0 = {c0}")
                for c in S[i][j].coeffs(0):
                    if not c.is_integral_laurent():
                        errs.append(f"{frm}->{to}: S[{i}][{j}] has non-integral coefficient {c}")
        sf = standard_form(S, default_order(frm, to))
        for e, ser in sf.items():
            for frame in ("standard", "P"):
                s2 = ser * p_frame_factor(R, e) if frame == "P" else ser
                for c in s2.coeffs(0):
                    if not c.is_integral_laurent():
                        errs.append(f"{frm}->{to} {frame}: S_{e} coefficient {c} not integral")
    return errs


def id_lower_sweep_ggmw130() -> List[str]:
    """GGMW (130) and (66): with the UDL form, S^-_{0c}(qt^-1) = S^+_{0g}(qt) (as series), S^-_{0g} = -S^-_{0c},
    S^-_{gg} = S^+_{cc}, S^-_{gc} = -S^+_{cg}."""
    from .blocks import Blocks, Ring
    from .stokes.cones import stokes_matrix, standard_form
    B = Blocks(Ring.symbolic(), 5)
    up = standard_form(stokes_matrix(B, "I", "II"), "LDU")
    lo = standard_form(stokes_matrix(B, "III", "IV"), "UDL")
    errs = []
    pairs = (("0c", up["0g"], 1), ("0g", up["0g"], -1), ("gg", up["cc"], 1), ("gc", up["cg"], -1))
    for e, ref, sgn in pairs:
        if not all(c == 0 for c in (lo[e] - ref * sgn).coeffs(0)):
            errs.append(f"S^-_{e} != {sgn:+d} * upper")
    return errs


def id_J_matrix() -> List[str]:
    """GGMW Thm 11: det J_m = x^-1 - x and J_{m+1} = J_m A(x, q^m, q) (plain blocks, m = -1, 0)."""
    from .blocks import Blocks, Ring
    from .algebra import TSeries, mat_mul, det3
    errs = []
    R = Ring.symbolic()
    N = 5
    B = Blocks(R, N)
    x = R.x
    for m in (-1, 0):
        J = B.J(m, decorated=False)
        d = det3(J) - (1 / x - x)
        if not all(c == 0 for c in d.coeffs(0)):
            errs.append(f"det J_{m} != x^-1 - x")
        J1 = B.J(m + 1, decorated=False)
        qm = TSeries.monomial(R.one, m + 1, N, R.zero)
        Amat = [[TSeries.const(R.one, N, R.zero), TSeries.const(R.zero, N, R.zero), TSeries.const(R.one, N, R.zero)],
                [TSeries.const(R.zero, N, R.zero), TSeries.const(R.zero, N, R.zero), TSeries.const(-R.one, N, R.zero)],
                [TSeries.const(R.zero, N, R.zero), TSeries.const(R.one, N, R.zero), TSeries.const(x + 1 / x, N, R.zero) - qm]]
        P = mat_mul(J, Amat)
        for i in range(3):
            for j in range(3):
                dd = P[i][j] - J1[i][j]
                if not all(c == 0 for c in dd.coeffs(0)):
                    errs.append(f"J_{m + 1} != J_{m} A at [{i}][{j}]")
    return errs


def id_phi0_routes() -> List[str]:
    """3.1: phi_0 of 4_1 by (i) MMR eps-route, (i') notebook Dq+Lagrange route, (ii) residue routes
    (Marino (3.103) at x = 1 with hbar^2 = -h^2; Phi0x with phi_0 = -sum res_{n+1} (-h^2)^n) agree."""
    import sympy as sp
    from .series.phi0 import mmr_eps, mmr_notebook, R_to_t_series, marino_x1, phi0_x1, phi0x_residue
    errs = []
    M = 6
    P = mmr_eps(-1, M)
    nb = mmr_notebook(-1, 3, 12)
    for m in range(4):
        t = R_to_t_series(P[m], -1, m, 6)
        if nb[m][:7] != t:
            errs.append(f"notebook route != eps route at h^{m}")
    P5 = mmr_eps(2, 3)
    nb5 = mmr_notebook(2, 3, 12)
    for m in range(4):
        if nb5[m][:7] != R_to_t_series(P5[m], 2, m, 6):
            errs.append(f"5_2: notebook route != eps route at h^{m}")
    mar = marino_x1(5)
    x1 = phi0_x1(-1, 10)
    for m in range(6):
        if mar[m] != (-1) ** m * x1[2 * m]:
            errs.append(f"Marino (3.103) hbar^{2 * m} != (-1)^m phi_0[h^{2 * m}]")
    x = sp.symbols("x")
    res = phi0x_residue(3)
    eps = sp.symbols("eps")
    for n in range(3):
        Pm = P[2 * n]
        R = sum(sp.Rational(int(c.p), int(c.q)) * eps ** i for i, c in enumerate(Pm.coeffs())) / (1 + eps) ** (4 * n + 1)
        R = R.subs(eps, 2 - x - 1 / x)
        if sp.cancel(R + (-1) ** n * res[n]) != 0:
            errs.append(f"Phi0x res_{n + 1} != -(-1)^n R_{2 * n}")
    return errs


def id_phi0_qdiff() -> List[str]:
    """phi0 guide 3: the formal solution of C(q^2 y) + C(y) - (x + 1/x - q y) C(q y) = 1 at y = 1 equals phi_0 (to h^2)."""
    import sympy as sp
    from .series.phi0 import qdiff_formal, mmr_eps
    x, y, q = sp.symbols("x y q")
    f = qdiff_formal([sp.Integer(1), -(x + 1 / x - q * y), sp.Integer(1)], 2)
    P = mmr_eps(-1, 2)
    eps = sp.symbols("eps")
    errs = []
    for m in range(3):
        R = sum(sp.Rational(int(c.p), int(c.q)) * eps ** i for i, c in enumerate(P[m].coeffs())) / (1 + eps) ** (2 * m + 1)
        if sp.cancel(R.subs(eps, 2 - x - 1 / x) - f[m]) != 0:
            errs.append(f"q-difference route != MMR at h^{m}")
    return errs


def id_leading_pole_law() -> List[str]:
    """6_1/7_2 note Prop. leadingpole: P_m(1/p) = (2m)!/m! c_p^m, c_p = (p+1)(4p-1)/(12p), p = -5..5, m <= 4."""
    from math import factorial
    from .series.phi0 import mmr_eps
    from flint import fmpq
    errs = []
    for p in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
        P = mmr_eps(p, 4)
        cp = Fraction((p + 1) * (4 * p - 1), 12 * p)
        for m, Pm in enumerate(P):
            v = Pm(fmpq(1, p)) if p != 0 else None
            exp = Fraction(factorial(2 * m), factorial(m)) * cp ** m
            if Fraction(int(v.p), int(v.q)) != exp:
                errs.append(f"p={p} m={m}: P_m(1/p) = {v} != {exp}")
    return errs


def id_jones_N2() -> List[str]:
    """Note C Example: N = 2 colored Jones of 5_2 and 6_1, N = 3 of 4_1 from the Masbaum sum."""
    from .series.phi0 import colored_jones
    errs = []
    exp = {
        (2, 2): {1: 1, 2: -1, 3: 2, 4: -1, 5: 1, 6: -1},
        (-2, 2): {2: 1, 1: -1, 0: 2, -1: -2, -2: 1, -3: -1, -4: 1},
        (-1, 3): {6: 1, 5: -1, 4: -1, 3: 2, 2: -1, 1: -1, 0: 3, -1: -1, -2: -1, -3: 2, -4: -1, -5: -1, -6: 1},
    }
    for (p, N), e in exp.items():
        if colored_jones(p, N) != e:
            errs.append(f"J_{N}(K_{p}) = {colored_jones(p, N)}")
    return errs


def id_riley() -> List[str]:
    """Note A table: Riley polynomials of 4_1 ... 8_1 from the Chebyshev formula; sector counts of the I_{K_p} engine."""
    import sympy as sp
    from .data import riley_polynomial
    from .series.twist import saddles, riley_degree
    u = sp.symbols("u")
    tab = {-1: "u^2+u+1", 2: "u^3-u^2+2*u-1", -2: "u^4+u^3+3*u^2+2*u+1",
           3: "u^5-u^4+4*u^3-3*u^2+3*u-1", -3: "u^6+u^5+5*u^4+4*u^3+6*u^2+3*u+1"}
    errs = []
    for p, s in tab.items():
        if sp.expand(riley_polynomial(p) - sp.sympify(s.replace("^", "**"), locals={"u": u})) != 0:
            errs.append(f"Riley K_{p}")
    for p in (-4, -3, -2, -1, 2, 3, 4):
        if len(saddles(p, Fraction(6, 5), 30)) != riley_degree(p):
            errs.append(f"K_{p}: wrong number of sectors")
    return errs


IDENTITIES: Dict[str, Callable[[], List[str]]] = {
    "V_g (89) = S(u,v_g)+pi^2/6+pi i u": id_V_from_89_vs_S,
    "delta_g = -delta_c": id_delta_g_minus_c,
    "a_n^c = (-1)^n a_n^g (generic x)": id_conjugate_sector,
    "scaled coefficients y-free": id_scaled_y_free,
    "S_{I->II}: (a) = (b), closed forms": id_S_identity_a_b,
    "Stokes integrality (Z[xt^+-1])": id_integrality,
    "lower sweep: GGMW (130), (66)": id_lower_sweep_ggmw130,
    "J_m: det and q-difference (GGMW Thm 11)": id_J_matrix,
    "phi_0 routes agree (MMR, notebook, Marino 3.4, Phi0x)": id_phi0_routes,
    "phi_0 q-difference route": id_phi0_qdiff,
    "leading-pole law P_m(1/p)": id_leading_pole_law,
    "colored Jones N=2,3 (Note C)": id_jones_N2,
    "Riley polynomials, sector counts": id_riley,
}
SLOW = {"phi_0 q-difference route"}
