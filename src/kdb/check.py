"""``kdb check``: golden tests of every engine + the identities of CLAUDE.md section 3."""
from __future__ import annotations

import time
import traceback
from fractions import Fraction
from typing import Dict, List

from .golden import CHECKS, check_case, load_golden, same_rational, sym

RECIPES = ("perturbative_series", "stokes_series", "index3d")
# cases skipped in --quick mode (minutes of CPU)
SLOW_CASES = {"private_phi_gc_series", "private_phi1x1d280", "private_PDzlist"}


# ----------------------------------------------------------------------------------------------
# comparison of external-engine output with golden expectations
# ----------------------------------------------------------------------------------------------
def _clean_num(s: str) -> str:
    import re
    s = re.sub(r"`[\d.]*", "", s.replace("``", "`"))
    return s.replace("*^", "e").replace("*10^", "e")


def engine_compare(case: Dict, out: Dict) -> List[str]:
    if "error" in out:
        return [f"engine error: {out['error']}"]
    chk = case["check"]
    errs = []
    if chk in ("rational_functions", "rationals"):
        for n, (g, e) in enumerate(zip(out["coefficients"], case["expect"])):
            if not same_rational(g, e):
                errs.append(f"h^{n}: {g} != {e}")
    elif chk == "scaled_numerators":
        for n, (g, e) in enumerate(zip(out["coefficients"], case["expect"])):
            g = g.split(" / ")[0]
            if not same_rational(g, e):
                errs.append(f"h^{n}: {g} != {e}")
    elif chk == "numeric":
        import mpmath
        from .golden import num
        mpmath.mp.dps = 60
        for n, (g, e) in enumerate(zip(out["coefficients"], case["expect"])):
            gv = mpmath.mpmathify(complex(0)) if False else None
            g2 = _clean_num(g).replace(" ", "")
            import sympy as sp
            gv = sp.N(sym(g2), 50)
            d = abs(complex(sp.N(gv - num(e), 30)))
            if d > float(case.get("tol", 1e-30)) * 1e5:
                errs.append(f"h^{n}: |{g} - {e}| = {d}")
    elif chk in ("laurent_series", "laurent_series_at1"):
        import sympy as sp
        for ent, exp in case["expect"].items():
            got = out["entries"].get(str(ent))
            if got is None:
                errs.append(f"entry {ent} missing")
                continue
            for k, (g, e) in enumerate(zip(got, exp)):
                gg = sym(g)
                if chk == "laurent_series_at1":
                    gg = gg.subs(sp.symbols("xt"), 1)
                if sp.simplify(gg - sym(e)) != 0:
                    errs.append(f"S_{ent} qt^{k}: {g} != {e}")
    else:
        errs.append(f"no engine comparison for check {chk}")
    return errs


# ----------------------------------------------------------------------------------------------
def main(quick: bool = False, engines=None) -> int:
    from .engines.runner import find_engine, run_engine
    from .identities import IDENTITIES, SLOW
    engines = engines or ["python", "sage", "wolfram"]
    avail = {e: (e == "python" or find_engine(e) is not None) for e in engines}
    rows = []
    t0 = time.time()

    def record(group, name, engine, errs, dt):
        status = "PASS" if not errs else ("SKIP" if all(e.startswith("SKIP") for e in errs) else "FAIL")
        rows.append((group, name, engine, status, errs, dt))
        print(f"[{status}] {group:20s} {name:45s} {engine:8s} {dt:6.1f}s" + ("" if status == "PASS" else "  " + "; ".join(errs[:2])))

    for recipe in RECIPES:
        for case in load_golden(recipe):
            for eng in case.get("engines", ["python"]):
                if eng not in engines:
                    continue
                if quick and case["id"] in SLOW_CASES:
                    record(recipe, case["id"], eng, ["SKIP: --quick"], 0.0)
                    continue
                if not avail[eng]:
                    record(recipe, case["id"], eng, [f"SKIP: {eng} not installed (untested)"], 0.0)
                    continue
                t = time.time()
                try:
                    if eng == "python":
                        errs = check_case(recipe, case)
                    else:
                        params = dict(case["params"])
                        params.pop("at_xt", None)
                        errs = engine_compare(case, run_engine(recipe, eng, params))
                except Exception as ex:
                    errs = [f"exception {ex!r}"]
                    traceback.print_exc()
                record(recipe, case["id"], eng, errs, time.time() - t)
    if "python" in engines:
        for name, fn in IDENTITIES.items():
            if quick and name in SLOW:
                record("identities", name, "python", ["SKIP: --quick"], 0.0)
                continue
            t = time.time()
            try:
                errs = fn()
            except Exception as ex:
                errs = [f"exception {ex!r}"]
                traceback.print_exc()
            record("identities", name, "python", errs, time.time() - t)
    n = {s: sum(1 for r in rows if r[3] == s) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\nkdb check: {n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped  ({time.time() - t0:.0f}s)")
    return 1 if n["FAIL"] else 0
