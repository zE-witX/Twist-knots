"""5_2: holomorphic blocks, cone matrices and Stokes matrices with x-deformation.

Sources (labels): GGM Peacock 6.2, 6.4 (blocks (241)-(242), Wronskian (244), cones (304a-d), S^pm (305));
GGMW 4.5-4.6 (Campbell's D_m (219), trivial row (234)-(237)); Gu notes (5_2 deformed, f_m = D_m).

Conventions for |q| > 1 (GGM sec. 4): (x; q^-1)_inf = 1/(qx; q)_inf, theta(x; q^-1) = 1/theta(x; q), and
H(x, y, z; q^-1) = H_-(x, y, z; q).  All series below are in a variable q with |q| < 1; the 'minus'
blocks are the values at nome q^-1 written as series in q.
Sectors: 0 (trivial), s1 (geometric), s2 (conjugate), s3 (real), ordered as in GGM/GGMW.
"""
from __future__ import annotations

from typing import Dict, List

from ..algebra import TSeries, inv_generic, mat_mul
from ..blocks import Ring, poch, poch_inf


def _S(R, cs, N, val=0):
    return TSeries(cs, N, val, R.zero)


def _add_shifted(R, acc, term, e, N):
    if e >= 0:
        return acc + TSeries([R.zero] * e + term.c, N, 0, R.zero)
    return acc + TSeries(term.c, N, e, R.zero)


def H_plus(R, x, y, zc, zq, N):
    """H_+(x, y, z; q) of GGM (242a), z = zc q^zq."""
    pre = poch_inf(R, x, 1, N) * poch_inf(R, y, 1, N)
    s = _S(R, [], N)
    n = 0
    while True:
        e = n * (n + 1) + zq * n
        if e > N and n > abs(zq) + 1:
            break
        if e <= N:
            den = poch(R, R.one, 1, n, N) * poch(R, x, 1, n, N) * poch(R, y, 1, n, N)
            s = _add_shifted(R, s, den.inverse() * (zc ** n), e, N)
        n += 1
    return pre * s


def H_minus(R, x, y, zc, zq, N):
    """H_-(x, y, z; q) of GGM (242b), z = zc q^zq."""
    pre = (poch_inf(R, x, 0, N) * poch_inf(R, y, 0, N)).inverse()
    xi, yi = R.one / x, R.one / y
    s = _S(R, [], N)
    n = 0
    while True:
        e = n * (n + 1) // 2 + zq * n
        if e > N and n > 2 * abs(zq) + 2:
            break
        if e <= N:
            den = poch(R, R.one, 1, n, N) * poch(R, xi, 1, n, N) * poch(R, yi, 1, n, N)
            s = _add_shifted(R, s, den.inverse() * ((-1) ** n * (xi * yi * zc) ** n), e, N)
        n += 1
    return pre * s


def _xp(R, x, e):
    return x ** e if e >= 0 else (R.one / x) ** (-e)


def blocks(R: Ring, x, m: int, N: int, nome: str = "+"):
    """(A_m, B_m, C_m)(x; q^{+-1}) of GGM (241); x is a ring element (pass 1/x for the reflected blocks)."""
    xi = R.one / x
    if nome == "+":
        th_p = poch_inf(R, x, 1, N) * poch_inf(R, xi, 0, N)       # theta(-q^{1/2} x; q)
        th_m = poch_inf(R, x, 0, N) * poch_inf(R, xi, 1, N)       # theta(-q^{-1/2} x; q)
        A = H_plus(R, x, xi, R.one, m, N)
        B = (th_p * th_p).inverse() * _xp(R, x, m) * H_plus(R, x, x * x, x * x, m, N)
        C = (th_m * th_m).inverse() * _xp(R, x, -m) * H_plus(R, xi, xi * xi, xi * xi, m, N)
    else:
        # theta(-Q^{1/2} x; Q)^{-2} = theta(-q^{-1/2} x; q)^2, theta(-Q^{-1/2} x; Q)^{-2} = theta(-q^{1/2} x; q)^2
        th_m = poch_inf(R, x, 0, N) * poch_inf(R, xi, 1, N)
        th_p = poch_inf(R, x, 1, N) * poch_inf(R, xi, 0, N)
        A = H_minus(R, x, xi, R.one, -m, N)
        B = (th_m * th_m) * _xp(R, x, m) * H_minus(R, x, x * x, x * x, -m, N)
        C = (th_p * th_p) * _xp(R, x, -m) * H_minus(R, xi, xi * xi, xi * xi, -m, N)
    return A, B, C


def wronskian(R: Ring, x, m: int, N: int, nome: str = "+"):
    """W_m(x; q^{+-1}) of GGM (244): rows m, m+1, m+2; columns A, B, C."""
    return [list(blocks(R, x, m + k, N, nome)) for k in range(3)]


def D_block(R: Ring, m: int, N: int) -> TSeries:
    """Campbell's D_m(x, q) (GGMW (219)) = Gu's f_m; |q| < 1; may start at q^{-(m+1)}."""
    x = R.x
    xi = R.one / x
    M = N + m + 4
    out = _S(R, [], N, -(m + 1))
    for n in range(0, M + 2):
        for k in range(0, 2 * M + 6):
            e = n * (n + 1) + k * (k + 1) // 2 - n * k - (m + 1) * (n + 1)
            if e > N:
                continue
            num = poch(R, R.one, 1, n + k, M)
            den = poch(R, R.one, 1, k, M) * poch(R, R.one, 1, n, M) * poch(R, xi, 0, n + 1, M) * poch(R, x, 0, n + 1, M)
            t = (num / den) * ((-1) ** (k + 1))
            out = out + TSeries(t.c, N, e, R.zero)
    return out


def _c(R, v, N):
    return TSeries.const(R.c(v), N, R.zero)


def cone_R(R: Ring, cone: str):
    o, z = R.one, R.zero
    s = R.one / R.x + o + R.x
    return {
        "I": [[z, z, -o], [o, -s, z], [z, o, z]],
        "II": [[z, z, -o], [-s, o, z], [o, z, z]],
        "III": [[z, z, -o], [o, o, z], [o, z, z]],
        "IV": [[z, z, -o], [o, o, z], [z, o, z]],
    }[cone]


def cone_matrix(R: Ring, cone: str, N: int):
    """M_R(x; q) of GGM (304a-d) (reduced 3x3, nonabelian sectors); I, II at nome q, III, IV at nome q^-1."""
    x = R.x
    nome = "+" if cone in ("I", "II") else "-"
    arg = x if cone in ("I", "IV") else R.one / x
    W = wronskian(R, arg, -1, N, nome)
    WT = [[W[j][i] for j in range(3)] for i in range(3)]
    Rc = [[_c(R, v, N) for v in row] for row in cone_R(R, cone)]
    return mat_mul(WT, Rc)


def stokes_red(R: Ring, frm: str, to: str, N: int):
    """Reduced (nonabelian) S_{R->R'} = M_{R'}^{-1} M_R for I->II (series in qt) and III->IV (series in 1/qt)."""
    return mat_mul(inv_generic(cone_matrix(R, to, N)), cone_matrix(R, frm, N))


def splus_ggm(R: Ring, N: int):
    """GGM (305a): S^+ = L W_{-1}(1/x; q^-1) W_{-1}(x; q)^T R_IV  (includes the real-axis factor S_{IV->I})."""
    x = R.x
    L = [[0, 1, 0], [0, 1, 1], [-1, 0, 0]]
    Wm = wronskian(R, R.one / x, -1, N, "-")
    Wp = wronskian(R, x, -1, N, "+")
    WpT = [[Wp[j][i] for j in range(3)] for i in range(3)]
    Lc = [[_c(R, v, N) for v in row] for row in L]
    Rc = [[_c(R, v, N) for v in row] for row in cone_R(R, "IV")]
    return mat_mul(mat_mul(mat_mul(Lc, Wm), WpT), Rc)


def sminus_ggm(R: Ring, N: int):
    """GGM (305b): S^- = [[0,-s,1],[0,1,0],[-1,0,0]] W_{-1}(x; q) W_{-1}(1/x; q^-1)^T [[0,0,-1],[-s,1,0],[1,0,0]]."""
    x = R.x
    s = R.one / x + R.one + x
    L = [[R.zero, -s, R.one], [R.zero, R.one, R.zero], [-R.one, R.zero, R.zero]]
    Wp = wronskian(R, x, -1, N, "+")
    Wm = wronskian(R, R.one / x, -1, N, "-")
    WmT = [[Wm[j][i] for j in range(3)] for i in range(3)]
    Lc = [[_c(R, v, N) for v in row] for row in L]
    Rc = [[_c(R, v, N) for v in row] for row in cone_R(R, "II")]
    return mat_mul(mat_mul(mat_mul(Lc, Wp), WmT), Rc)


def trivial_row_M(R: Ring, cone: str, N: int):
    """GGMW (235)-(236): M_R = (qt D_0, qt^2 D_1, qt^3 D_2) M^{(s0)}_R  (row vector), R = I, II."""
    o, z = R.one, R.zero
    s = R.one / R.x + o + R.x
    G = N + 4
    q = TSeries.monomial(o, 1, G, z)
    D = [q * D_block(R, 0, G), q * q * D_block(R, 1, G), q * q * q * D_block(R, 2, G)]
    if cone == "I":
        M0 = [[_c(R, 1, G), _c(R, -1, G), q * (-s)], [_c(R, 0, G), _c(R, -1, G), _c(R, -1, G) + q * s],
              [_c(R, 0, G), _c(R, 0, G), -q]]
    else:
        M0 = [[_c(R, -1, G), _c(R, 1, G), q * (-s)], [_c(R, -1, G), _c(R, 0, G), _c(R, -1, G) + q * s],
              [_c(R, 0, G), _c(R, 0, G), -q]]
    return [x.truncate(N) for x in mat_mul([D], M0)[0]]


def stokes_I_II(R: Ring, N: int):
    """Full 4x4 S_{I->II} (rows/cols 0, s1, s2, s3) in the Delta-normalisation of GGMW (229)-(230)."""
    Sred = stokes_red(R, "I", "II", N)
    MI = trivial_row_M(R, "I", N)
    MII = trivial_row_M(R, "II", N)
    row0 = []
    for j in range(3):
        acc = MI[j]
        for k in range(3):
            acc = acc - MII[k] * Sred[k][j]
        row0.append(acc.truncate(N))
    one = _c(R, 1, N)
    zero = _c(R, 0, N)
    S = [[one] + row0]
    for i in range(3):
        S.append([zero] + [Sred[i][j].truncate(N) for j in range(3)])
    return S


def numeric_blocks(xv, qv, nome: str = "+", dps: int = 30):
    """(A_0, B_0, C_0) at numeric x and 0 < |q| < 1 (nome q) or nome 1/q, by direct summation (mpmath)."""
    import mpmath as mp
    mp.mp.dps = dps
    q = mp.mpmathify(qv)
    x = mp.mpmathify(xv)

    def poch(a, n):
        r = mp.mpf(1)
        for j in range(n):
            r *= (1 - a * q ** j)
        return r

    def pinf(a):
        return poch(a, int(4 * dps / max(1e-3, -float(mp.log10(abs(q))))) + 20)

    def Hp(a, b, z):
        s = mp.fsum(q ** (n * (n + 1)) * z ** n / (poch(q, n) * poch(q * a, n) * poch(q * b, n)) for n in range(60))
        return pinf(q * a) * pinf(q * b) * s

    def Hm(a, b, z):
        s = mp.fsum((-1) ** n * q ** (n * (n + 1) / 2) * a ** (-n) * b ** (-n) * z ** n /
                    (poch(q, n) * poch(q / a, n) * poch(q / b, n)) for n in range(80))
        return s / (pinf(a) * pinf(b))

    def th(y):
        return pinf(-mp.sqrt(q) * y) * pinf(-mp.sqrt(q) / y)

    if nome == "+":
        return [Hp(x, 1 / x, 1), th(-mp.sqrt(q) * x) ** -2 * Hp(x, x * x, x * x),
                th(-x / mp.sqrt(q)) ** -2 * Hp(1 / x, 1 / x ** 2, 1 / x ** 2)]
    return [Hm(x, 1 / x, 1), th(-x / mp.sqrt(q)) ** 2 * Hm(x, x * x, x * x),
            th(-mp.sqrt(q) * x) ** 2 * Hm(1 / x, 1 / x ** 2, 1 / x ** 2)]
