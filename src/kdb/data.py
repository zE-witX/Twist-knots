"""Knot registry and the permanent data layout.

Axes: (knot, group, side, kind).  Built-in knot definitions live in ``src/kdb/knots/<id>.yaml`` (committed);
generated data go to ``data/knots/<knot>/<group>/<side>/<kind>.json`` (git-ignored by default).
Any twist knot K_p without a YAML file is generated from p (Alexander, Riley polynomial, sector count).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PKG = Path(__file__).resolve().parent
REPO = PKG.parent.parent

# Rolfsen names of the twist knots K_p (Masbaum's convention, K_{-1} = 4_1, K_2 = 5_2, ...).
TWIST_NAMES = {1: "3_1", -1: "4_1", 2: "5_2", -2: "6_1", 3: "7_2", -3: "8_1", 4: "9_2", -4: "10_1"}
NAME_TO_P = {v: k for k, v in TWIST_NAMES.items()}


def repo_root() -> Path:
    return Path(os.environ.get("KDB_ROOT", REPO))


def data_dir() -> Path:
    return repo_root() / "data"


def out_dir() -> Path:
    return repo_root() / "out"


def private_inputs() -> Optional[Path]:
    """Location of the private research inputs (never copied into the repo); None if absent."""
    env = os.environ.get("KDB_PRIVATE_INPUTS")
    cands = [Path(env)] if env else []
    cands.append(repo_root().parent / "inputs")
    for c in cands:
        if c.is_dir():
            return c
    return None


def group_id(group: Dict) -> str:
    return f"{group.get('family', 'SL')}{group.get('N', 2)}"


@dataclass
class Knot:
    id: str
    p: Optional[int]
    raw: Dict = field(default_factory=dict)

    @property
    def aliases(self) -> Dict:
        return self.raw.get("aliases", {})

    @property
    def sectors(self) -> List[Dict]:
        return self.raw.get("sectors", [])

    def path(self, kind: str, group: Dict = None, side: str = "cs", ext: str = "json") -> Path:
        g = group_id(group or {"family": "SL", "N": 2})
        return data_dir() / "knots" / self.id / g / side / f"{kind}.{ext}"


def chebyshev_S(n: int, xs):
    """S_n(x) with S_{-1} = 0, S_0 = 1, S_n = x S_{n-1} - S_{n-2}; sympy expression."""
    import sympy as sp
    if n == -1:
        return sp.Integer(0)
    a, b = sp.Integer(0), sp.Integer(1)
    if n < -1:
        # S_{-k} = -S_{k-2}
        return -chebyshev_S(-n - 2, xs)
    for _ in range(n):
        a, b = b, sp.expand(xs * b - a)
    return b


def riley_polynomial(p: int):
    """Riley polynomial of K_p at the parabolic point, Note A Prop. (twistriley) in Masbaum's index.

    Note A's index is k = -p (its K_1 is 4_1, its K_{-1} the trefoil); returned monic.
    """
    import sympy as sp
    u = sp.symbols("u")
    xs = u ** 2 + 2
    k = -p
    if k > 0:
        f = chebyshev_S(k - 1, xs) * (u ** 2 + u + 1) - chebyshev_S(k - 2, xs)
    else:
        n = -k - 1
        f = chebyshev_S(n - 1, xs) * (-u ** 3 + u ** 2 - 2 * u + 1) - chebyshev_S(n - 2, xs) * (1 - u)
    f = sp.expand(f)
    lc = sp.Poly(f, u).LC()
    return sp.expand(f / lc)


def generic_twist(p: int) -> Knot:
    name = TWIST_NAMES.get(p, f"K_{p}")
    from .series.twist import riley_degree
    raw = {
        "schema": "knot/1",
        "id": name,
        "twist_p": p,
        "aliases": {"masbaum": f"K_{p}", "rolfsen": TWIST_NAMES.get(p)},
        "amphichiral": p == -1,
        "classical": {
            "alexander_z": {"value": f"1 + ({p})*z^2", "source": "CLAUDE.md 3.0", "status": "known"},
            "alexander_t": {"value": f"({p})*t + ({1 - 2 * p}) + ({p})*t^-1",
                            "source": "Note B, Cor. twistcheck with n = -p (symmetric normalisation)",
                            "status": "known"},
        },
        "sectors": [{"id": "0", "name": "trivial", "V": "0"}]
                   + [{"id": f"s{i + 1}", "name": "nonabelian"} for i in range(riley_degree(p))],
        "state_integral": {"kind": "two_dimensional_habiro",
                           "description": "Marino 5.4 I_{K_p}(z,w)", "status": "conjectural"},
        "generated": True,
    }
    return Knot(name, p, raw)


def load_knot(knot: str = None, twist: int = None) -> Knot:
    if twist is not None:
        name = TWIST_NAMES.get(twist)
        if name and (PKG / "knots" / f"{name}.yaml").exists():
            return load_knot(name)
        return generic_twist(twist)
    if knot is None:
        raise ValueError("give --knot or --twist")
    if knot.startswith("K_"):
        return load_knot(twist=int(knot[2:]))
    f = PKG / "knots" / f"{knot}.yaml"
    if f.exists():
        raw = yaml.safe_load(f.read_text())
        return Knot(raw["id"], raw.get("twist_p"), raw)
    if knot in NAME_TO_P:
        return generic_twist(NAME_TO_P[knot])
    raise KeyError(f"unknown knot {knot!r}")


def builtin_knots() -> List[str]:
    return sorted(f.stem for f in (PKG / "knots").glob("*.yaml"))


def write_item(knot: Knot, kind: str, payload: Dict, group=None, side="cs") -> Path:
    """Write a data item (adds the standard envelope fields if missing)."""
    path = knot.path(kind, group, side)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("schema", f"{kind}/1")
    payload.setdefault("knot", knot.id)
    payload.setdefault("group", group or {"family": "SL", "N": 2})
    payload.setdefault("side", side)
    payload.setdefault("refs", [])
    path.write_text(json.dumps(payload, indent=1))
    return path
