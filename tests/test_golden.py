"""Golden tests of both recipes, python reference engine (see recipes/*/tests/golden.yaml)."""
import pytest

from kdb.check import SLOW_CASES
from kdb.golden import check_case, load_golden

CASES = [(r, c) for r in ("perturbative_series", "stokes_series") for c in load_golden(r) if "python" in c.get("engines", ["python"])]


@pytest.mark.parametrize("recipe,case", CASES, ids=[f"{r}:{c['id']}" for r, c in CASES])
def test_golden(recipe, case):
    if case["id"] in SLOW_CASES and not pytest.slow_enabled:
        pytest.skip("slow case (run with --runslow or kdb check)")
    errs = check_case(recipe, case)
    if errs and all(e.startswith("SKIP") for e in errs):
        pytest.skip(errs[0])
    assert not errs, errs
