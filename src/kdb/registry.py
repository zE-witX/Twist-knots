"""Module registry: one directory per data kind under src/kdb/modules/<kind>/ (module.yaml, schema.json, README.md)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml

MOD_DIR = Path(__file__).resolve().parent / "modules"


def modules() -> List[Dict]:
    out = []
    for d in sorted(MOD_DIR.iterdir()):
        f = d / "module.yaml"
        if f.exists():
            m = yaml.safe_load(f.read_text())
            m["path"] = str(d)
            out.append(m)
    return out


def schema(kind: str) -> Dict:
    return json.loads((MOD_DIR / kind / "schema.json").read_text())


def validate(kind: str, payload: Dict) -> List[str]:
    """Minimal validation (required keys and enums) without external dependencies."""
    s = schema(kind)
    errs = [f"missing key {k!r}" for k in s.get("required", []) if k not in payload]
    for k, spec in s.get("properties", {}).items():
        if k in payload and "enum" in spec and payload[k] not in spec["enum"]:
            errs.append(f"{k}={payload[k]!r} not in {spec['enum']}")
    return errs
