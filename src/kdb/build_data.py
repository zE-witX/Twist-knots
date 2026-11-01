"""``kdb data build``: write the permanent data items data/knots/<knot>/SL2/cs/<kind>.json (git-ignored)."""
from __future__ import annotations

from fractions import Fraction
from typing import List

from .data import builtin_knots, load_knot, riley_polynomial, write_item
from .registry import validate


def build(knots: List[str] | None = None):
    from .recipes import run_perturbative_series, run_stokes_series
    from .data import TWIST_NAMES
    knots = knots or sorted(set(builtin_knots()) | {TWIST_NAMES[p] for p in range(-4, 5) if p not in (0, 1)})
    written = []
    for kid in knots:
        k = load_knot(kid)
        cl = {"status": "known", "source": "knot definition file (see per-item sources)",
              "items": k.raw.get("classical", {}), "riley_from_formula": str(riley_polynomial(k.p))}
        written.append(write_item(k, "classical", cl))
        si = {"status": k.raw.get("state_integral", {}).get("status", "unknown"),
              "source": k.raw.get("state_integral", {}).get("description", ""),
              "integrand": k.raw.get("state_integral", {})}
        written.append(write_item(k, "state_integral", si))
        ser = run_perturbative_series({"knot": k.id, "sector": "0", "order": 8, "x": "symbol"})
        written.append(write_item(k, "series", {"status": "known", "source": "Masbaum/Habiro + MMR (recipe perturbative_series)",
                                                "sector": "0", "result": ser["sectors"]["0"]}))
        if k.p is not None:
            nonab = run_perturbative_series({"knot": k.id, "sector": "all", "order": 6, "x": "1", "digits": 40}) \
                if k.p != -1 else run_perturbative_series({"knot": k.id, "sector": "g", "order": 12, "x": "1", "digits": 60})
            written.append(write_item(k, "series_nonabelian_x1",
                                      {"status": "ours", "source": "recipe perturbative_series", "digits": 40,
                                       "result": nonab["sectors"]}))
        if k.p == -1:
            st = run_stokes_series({"knot": k.id, "from_cone": "I", "to_cone": "II", "order": 6, "x": "symbol"})
            written.append(write_item(k, "stokes", {"status": "conjectural", "source": "GGMW Conj. 14-15 (corrected P_III, P_IV)",
                                                    "result": st}))
    for path in written:
        import json
        payload = json.loads(path.read_text())
        errs = validate(path.stem if path.stem in ("classical", "state_integral", "series", "stokes") else "series", payload)
        if errs:
            raise ValueError(f"{path}: {errs}")
    return written
