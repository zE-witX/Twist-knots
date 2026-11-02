"""Cone matrices M_R and Stokes automorphisms of 4_1 (GGMW Conj. 14-15, (116)-(135); CLAUDE.md 3.4-3.5).

Conventions (rows/columns ordered 0, g, c):
    M_R = L_R  calJ_{-1}(xt, qt)  P_R          (calJ decorated, see kdb.blocks)
    S_{R->R'} = M_{R'}^{-1} M_R                (R' reached by increasing arg tau)
with the branch-cut rule GGMW (134) for II -> III.

P_III, P_IV: with the corrections to GGMW of the Phi0 note (item 2: the entry printed xt^-1 - 1 in
(118c,d) is 1).  Check: GGMW (131), (135) and the x -> 1 limit (41c,d), (54) are reproduced.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List

from ..algebra import TSeries, inv_generic, mat_mul, RatFun
from ..blocks import Blocks, Ring

CONES = ("I", "II", "III", "IV")
REGION = {"I": "inside", "II": "inside", "III": "outside", "IV": "outside"}
SECTORS = ("0", "g", "c")


def L_matrix(R: Ring, cone: str):
    o, z = R.one, R.zero
    return {
        "I": [[o, z, z], [z, o, z], [z, z, o]],
        "II": [[o, z, z], [z, z, o], [z, o, z]],
        "III": [[o, z, z], [z, z, -o], [z, -o, z]],
        "IV": [[o, z, z], [z, o, z], [z, z, o]],
    }[cone]


def P_matrix(R: Ring, cone: str, variant: str = "corrected"):
    o, z, x = R.one, R.zero, R.x
    xi = o / x
    t = x + xi
    if variant != "corrected":
        raise ValueError("only the corrected P_III, P_IV (Phi0 note, corrections to GGMW) are supported")
    e = o
    return {
        "I": [[o, z, z], [z, z, -o], [z, o, -o]],
        "II": [[o, z, z], [z, o, z], [z, o, -o]],
        "III": [[o, e, z], [z, -o, z], [z, t, o]],
        "IV": [[o, z, e], [z, z, -o], [z, o, t]],
    }[cone]


def _embed(R, M, N):
    return [[TSeries.const(a, N, R.zero) for a in row] for row in M]


def cone_matrix(B: Blocks, cone: str, variant: str = "corrected"):
    """M_R(xt, qt) as 3x3 matrix of TSeries (in qt for I, II; in 1/qt for III, IV)."""
    R, N = B.R, B.N
    J = B.J(-1, decorated=True, region=REGION[cone])
    return mat_mul(mat_mul(_embed(R, L_matrix(R, cone), N), J), _embed(R, P_matrix(R, cone, variant), N))


def stokes_matrix(B: Blocks, frm: str, to: str, variant: str = "corrected"):
    """Global Stokes matrix S_{frm -> to} for adjacent cones (counter-clockwise), CLAUDE.md 3.5(a).

    Returns a 3x3 matrix whose entries are TSeries (variable qt for I->II, 1/qt for III->IV) or
    constants (IV->I, II->III, where both cone matrices share calJ_{-1} and the product is formal).
    """
    R, N = B.R, B.N
    pair = (frm, to)
    if pair == ("I", "II") or pair == ("III", "IV"):
        Mf = cone_matrix(B, frm, variant)
        Mt = cone_matrix(B, to, variant)
        return mat_mul(inv_generic(Mt), Mf)
    if pair == ("IV", "I"):
        # M_I^{-1} M_IV = P_I^{-1} calJ^{-1} L_I^{-1} L_IV calJ P_IV = P_I^{-1} P_IV   (L_I = L_IV = 1)
        S = mat_mul(inv_generic(P_matrix(R, "I", variant)), P_matrix(R, "IV", variant))
        return _embed(R, S, N)
    if pair == ("II", "III"):
        # GGMW (134): S = diag(-1,1,1) M_III^{-1} diag(-1,1,1) M_II ; the calJ factors cancel because
        # L_III^{-1} diag(-1,1,1) L_II = -1.
        d = [[-R.one, R.zero, R.zero], [R.zero, R.one, R.zero], [R.zero, R.zero, R.one]]
        Lmid = mat_mul(mat_mul(inv_generic(L_matrix(R, "III")), d), L_matrix(R, "II"))
        for i in range(3):
            for j in range(3):
                assert Lmid[i][j] == (-R.one if i == j else R.zero), "branch-cut identity failed"
        S = mat_mul(d, mat_mul(inv_generic(P_matrix(R, "III", variant)), P_matrix(R, "II", variant)))
        S = [[-a for a in row] for row in S]
        return _embed(R, S, N)
    raise ValueError(f"adjacent cone pair {frm}->{to} not implemented (use I->II, II->III, III->IV, IV->I)")


def stokes_middle_form_b(B: Blocks):
    """CLAUDE.md 3.5(b) / Pengxu (3.4): S_{I->II} written with calJ_{-1}^{-1} and calJ_{-1} explicitly."""
    R, N = B.R, B.N
    o, z = R.one, R.zero
    J = B.J(-1, decorated=True, region="inside")
    left = _embed(R, [[o, z, z], [z, o, z], [z, o, -o]], N)
    mid = _embed(R, [[o, z, z], [z, z, o], [z, o, z]], N)
    right = _embed(R, [[o, z, z], [z, z, -o], [z, o, -o]], N)
    return mat_mul(mat_mul(mat_mul(mat_mul(left, inv_generic(J)), mid), J), right)


def standard_form(S, order: str = "LDU") -> Dict[str, object]:
    """Gauss factorisation of a global Stokes matrix into its elementary pieces.

    order 'LDU' (upper sweep, I -> II; Pengxu (3.5)):
        S = [[1,S0g,0],[0,1,0],[0,Scg,1]] diag(1,Sgg,Scc) [[1,0,S0c],[0,1,Sgc],[0,0,1]]
    order 'UDL' (lower sweep, III -> IV; rays ordered the other way round):
        S = [[1,0,S0c],[0,1,Sgc],[0,0,1]] diag(1,Sgg,Scc) [[1,S0g,0],[0,1,0],[0,Scg,1]]
    With 'UDL', S_{III->IV} reproduces GGMW (130).  Returns the six entries plus S00.
    """

    def inv(a):
        return a.inverse() if isinstance(a, TSeries) else 1 / a

    if order == "LDU":
        S11 = S[1][1]
        i11 = inv(S11)
        Sgc = S[1][2] * i11
        S0g = S[0][1] * i11
        Scg = S[2][1] * i11
        return {"00": S[0][0], "gg": S11, "gc": Sgc, "0g": S0g, "cg": Scg,
                "0c": S[0][2] - S0g * S[1][2], "cc": S[2][2] - Scg * S[1][2]}
    if order == "UDL":
        S22 = S[2][2]
        i22 = inv(S22)
        S0c = S[0][2] * i22
        Sgc = S[1][2] * i22
        Scg = S[2][1] * i22
        return {"00": S[0][0], "cc": S22, "0c": S0c, "gc": Sgc, "cg": Scg,
                "0g": S[0][1] - S0c * S[2][1], "gg": S[1][1] - Sgc * S[2][1]}
    raise ValueError(order)


def default_order(frm: str, to: str) -> str:
    return "UDL" if (frm, to) == ("III", "IV") else "LDU"


def middle_formulas(B: Blocks):
    """Closed forms of CLAUDE.md 3.5(b): S_gg, S_gc, S_0g from decorated blocks at (xt, qt)."""
    A0, A1 = B.calA(0), B.calA(-1)
    B0, B1 = B.calB(0), B.calB(-1)
    C0, C1 = B.C(0), B.C(-1)
    num = B0 * B0 - A0 * A0
    gg = num / (A1 * B0 - A0 * B1)
    gc = (A0 * A1 - B0 * B1) / num - 1
    g0 = -C1 + C0 * (A1 + B1) / (A0 + B0)
    return {"gg": gg, "gc": gc, "0g": g0}


def p_frame_factor(R: Ring, entry: str):
    """Multiplier turning a standard-form entry into its P-frame value, CLAUDE.md 3.5(d)."""
    x, o = R.x, R.one
    return {
        "0g": x - o,
        "0c": o - o / x,
        "gc": o / x,
        "cg": x,
        "gg": o,
        "cc": o,
        "00": o,
    }[entry]


def euler_exponents_laurent(series_coeffs: List[Dict[int, Fraction]], M: int):
    """Plethystic (Euler) exponents of a series 1 + sum_{k>=1} c_k(x) q^k with Laurent coefficients.

    Input: list of dicts {l: coeff} for q^0..q^M.  Output: list of dicts Omega_m = {l: Omega(m,l)}.
    """
    # work with bivariate dict {(k,l): c}
    cur = {}
    for k, d in enumerate(series_coeffs[: M + 1]):
        for l, c in d.items():
            if c != 0:
                cur[(k, l)] = Fraction(c)
    out = []
    for m in range(1, M + 1):
        Om = {l: -c for (k, l), c in cur.items() if k == m and c != 0}
        out.append(Om)
        # divide cur by prod_l (1 - q^m x^l)^{Om[l]}  ==  multiply by prod_l (1 - q^m x^l)^{-Om[l]}
        for l, e in Om.items():
            e = int(e) if Fraction(e).denominator == 1 else None
            if e is None:
                raise ValueError("non-integral Euler exponent")
            cur = _mul_binomial_power(cur, m, l, -e, M)
    return out


def _mul_binomial_power(cur, m, l, e, M):
    """cur * (1 - q^m x^l)^e truncated at q^M (e any integer)."""
    # binomial series of (1 - y)^e, y = q^m x^l
    coeffs = [Fraction(1)]
    j = 1
    while j * m <= M:
        # coefficient of y^j in (1-y)^e = (-1)^j binom(e, j)
        prev = coeffs[-1]
        coeffs.append(prev * Fraction(-(e - j + 1), j))
        j += 1
    out = {}
    for (k, ll), c in cur.items():
        for j, b in enumerate(coeffs):
            kk = k + j * m
            if kk > M:
                break
            if b == 0:
                continue
            key = (kk, ll + j * l)
            out[key] = out.get(key, 0) + c * b
    return {k: v for k, v in out.items() if v != 0}


def characters(Omega: Dict[int, Fraction]) -> Dict[int, int]:
    """SU(2) character multiplicities of a palindromic Laurent polynomial (Pengxu (3.9))."""
    if not Omega:
        return {}
    top = max(abs(l) for l in Omega)
    return {j: int(Omega.get(j, 0) - Omega.get(j + 1, 0)) for j in range(0, top + 1)
            if Omega.get(j, 0) - Omega.get(j + 1, 0) != 0}
