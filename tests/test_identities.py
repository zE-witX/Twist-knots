"""Identities and cross-checks of CLAUDE.md section 3."""
import pytest

from kdb.identities import IDENTITIES, SLOW


@pytest.mark.parametrize("name", list(IDENTITIES))
def test_identity(name):
    if name in SLOW and not pytest.slow_enabled:
        pytest.skip("slow")
    assert IDENTITIES[name]() == []
