"""Sage and Wolfram engines against the same golden cases (skipped when the tool is not installed)."""
import pytest

from kdb.check import engine_compare
from kdb.engines.runner import find_engine, run_engine
from kdb.golden import load_golden

CASES = [(r, c, e) for r in ("perturbative_series", "stokes_series") for c in load_golden(r)
         for e in c.get("engines", []) if e in ("sage", "wolfram")]


@pytest.mark.engine
@pytest.mark.parametrize("recipe,case,engine", CASES, ids=[f"{e}:{r}:{c['id']}" for r, c, e in CASES])
def test_engine(recipe, case, engine):
    if find_engine(engine) is None:
        pytest.skip(f"{engine} not installed")
    params = dict(case["params"])
    params.pop("at_xt", None)
    assert engine_compare(case, run_engine(recipe, engine, params)) == []
