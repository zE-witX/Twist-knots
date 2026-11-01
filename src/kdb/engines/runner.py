"""Run the Sage / Wolfram implementations of a recipe and return their JSON output."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from ..data import repo_root

WOLFRAM_CANDIDATES = ["wolframscript", "/Applications/Wolfram.app/Contents/MacOS/wolframscript",
                      "/Applications/Mathematica.app/Contents/MacOS/wolframscript"]


def find_engine(engine: str) -> Optional[str]:
    if engine == "sage":
        return os.environ.get("KDB_SAGE") or shutil.which("sage")
    if engine == "wolfram":
        env = os.environ.get("KDB_WOLFRAMSCRIPT")
        if env:
            return env
        for c in WOLFRAM_CANDIDATES:
            p = shutil.which(c) or (c if Path(c).exists() else None)
            if p:
                return p
    return None


def script(recipe: str, engine: str) -> Path:
    return repo_root() / "recipes" / recipe / ("run.sage" if engine == "sage" else "run.wl")


def run_engine(recipe: str, engine: str, params: Dict, timeout: int = 900) -> Dict:
    exe = find_engine(engine)
    if exe is None:
        raise FileNotFoundError(f"{engine} not installed")
    arg = json.dumps(params)
    if engine == "sage":
        cmd = [exe, str(script(recipe, engine)), arg]
    else:
        cmd = [exe, "-file", str(script(recipe, engine)), arg]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(script(recipe, engine).parent))
    lines = [l for l in out.stdout.splitlines() if l.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"{engine} produced no JSON:\nstdout: {out.stdout[-2000:]}\nstderr: {out.stderr[-2000:]}")
    return json.loads(lines[-1])
