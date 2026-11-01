"""``kdb data build``: write the permanent data items data/knots/<knot>/SL2/<side>/<kind>.json (git-ignored)."""
from __future__ import annotations

import json
from fractions import Fraction
from typing import List

from .data import TWIST_NAMES, builtin_knots, load_knot, riley_polynomial, write_item
from .registry import validate

# phi_0-related Stokes data at k = 0 recorded from the notes (Borel analysis of phi_0, upper/lower sweep)
PHI0_K0 = {
    "6_1": {"source": "6_1/7_2 note Prop. 61phi0, phi0x", "status": "known", "notes": "numerical (Borel-Pade / large order)",
            "constants": {"s3": "-(xt^(1/2) - xt^(-1/2))", "s4": "-(xt^(1/2) - xt^(-1/2))",
                          "s1": "+(xt^(1/2) - xt^(-1/2))", "s2": "+(xt^(1/2) - xt^(-1/2))"},
            "sweep": {"s3": "upper", "s4": "upper", "s1": "lower", "s2": "lower"},
            "trivial_row_n": {"s3": 1, "s4": -1, "s1": -1, "s2": 1, "half_plane": "lower"}},
    "7_2": {"source": "6_1/7_2 note Prop. 72phi0, phi0x, trivrow", "status": "partial", "notes": "numerical; the sign for s1, s2 is conjectural",
            "constants": {"s5": "+(xt^(1/2) - xt^(-1/2))", "s3": "-(xt^(1/2) - xt^(-1/2))", "s4": "-(xt^(1/2) - xt^(-1/2))",
                          "s1": "+(xt^(1/2) - xt^(-1/2)) (conjectural sign)", "s2": "+(xt^(1/2) - xt^(-1/2)) (conjectural sign)"},
            "sweep": {"s5": "lower", "s3": "upper", "s4": "upper", "s1": "upper", "s2": "upper"},
            "trivial_row_n": {"s5": -1, "s3": 1, "s4": 1, "s1": -1, "s2": -1, "half_plane": "upper"}},
    "5_2": {"source": "GGMW (240) (generating series of the trivial-row Stokes constants, upper sweep)",
            "status": "known", "notes": "published; not yet re-derived (needs the ordered ray factorisation)",
            "S_plus_0j": {"s1": "-1 + qt + s*qt^2 + (-2 + 3*s^2)*qt^3 + (2 - s - 2*s^2 + 5*s^3 + 2*s^4)*qt^4",
                          "s2": "1 - qt - s*qt^2 - (-2 + 3*s^2)*qt^3 + (2 - s - 2*s^2 + 5*s^3 + 2*s^4)*qt^4",
                          "s3": "0", "s": "xt + 1 + xt^-1"}},
}


def build(knots: List[str] | None = None):
    from .recipes import run_perturbative_series, run_stokes_series, run_index3d
    from .series.phi0 import inverted_habiro_coeff
    knots = knots or sorted(set(builtin_knots()) | {TWIST_NAMES[p] for p in range(-4, 5) if p not in (0, 1)})
    written = []
    for kid in knots:
        k = load_knot(kid)
        cl = {"status": "known", "source": "knot definition file (see per-item sources)",
              "items": k.raw.get("classical", {}), "riley_from_formula": str(riley_polynomial(k.p))}
        written.append(write_item(k, "classical", cl))
        si = {"status": "conjectural", "source": "Marino 5.4 (new integral); GGM (113), (230), 6_1/7_2 note (int61), (int72) (NZ)",
              "integrals": {
                  "marino": "I_{K_p}(z,w): Phi_b^-1(z-i/2b+u) Phi_b^-1(z-i/2b-u) Phi_b^-1(z-i/2b) Phi_b(z-w+ib/2-i/2b) "
                            "Phi_b(z+w+ib/2-i/2b) e^{-2 pi i (p+1/2)(w+i/2b)^2} (e^{2 pi b(z+w)} - e^{2 pi b(z-w)}) "
                            "tanh(pi z/b) tanh(pi w/b)  (kdb.series.twist.integrand)",
                  "nz": ("kdb.series.integrands.nz_integrand('%s')" % k.id) if k.id in ("4_1", "5_2", "6_1", "7_2")
                        else "not recorded (needs the 2d reduction of the NZ integral; NZ data in knots/nz_data.yaml)"},
              "description": k.raw.get("state_integral", {})}
        written.append(write_item(k, "state_integral", si))
        ser = run_perturbative_series({"knot": k.id, "sector": "0", "order": 8, "x": "symbol"})
        written.append(write_item(k, "series", {"status": "known", "source": "Masbaum/Habiro + MMR (recipe perturbative_series)",
                                                "sector": "0", "result": ser["sectors"]["0"]}))
        if k.p != -1:
            nonab = run_perturbative_series({"knot": k.id, "sector": "all", "order": 6, "x": "1", "digits": 40})
        else:
            nonab = run_perturbative_series({"knot": k.id, "sector": "g", "order": 12, "x": "1", "digits": 60})
        written.append(write_item(k, "series_nonabelian_x1", {"status": "ours", "source": "recipe perturbative_series",
                                                              "digits": 40, "result": nonab["sectors"]}))
        # inverted Habiro coefficients a^p_{-n-1}
        inv = k.p < 0
        coeffs = {str(n): {str(e): v for e, v in sorted(inverted_habiro_coeff(k.p, n, 30, inverse_nome=inv).items())}
                  for n in range(4)}
        written.append(write_item(k, "inverted_habiro", {
            "status": "ours", "notes": "formula of Marino 5.4 with the corrected exponent for p > 0", "source": "Marino 5.4 (ap-n), (iap-n)",
            "variable": "q" if not inv else "Q = 1/q", "a_minus_n_minus_1": coeffs}))
        # 3D index (dual side)
        try:
            idx = run_index3d({"knot": k.id, "order": 6, "flux": [0, 2, -2]})
            written.append(write_item(k, "index", {"status": "ours", "notes": "flux 0 checked against S^+_gg",
                                                   "source": "SnapPy NZ data + DGG state sum", "result": idx},
                                      side="dual"))
        except KeyError:
            pass
        if k.id in PHI0_K0:
            written.append(write_item(k, "stokes_phi0", dict(PHI0_K0[k.id])))
        if k.p in (-1, 2, 3):
            st = run_stokes_series({"knot": k.id, "from_cone": "I", "to_cone": "II", "order": 4, "x": "symbol"})
            written.append(write_item(k, "stokes", {"status": "conjectural", "source": st["conventions"]["definition"],
                                                    "result": st}))
    for path in written:
        payload = json.loads(path.read_text())
        kind = path.stem
        errs = validate(kind if kind in ("classical", "state_integral", "series", "stokes") else "series", payload) \
            if kind != "index" else []
        if errs:
            raise ValueError(f"{path}: {errs}")
    return written
