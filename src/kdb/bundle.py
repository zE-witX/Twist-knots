"""Result bundles: out/<recipe>/<hash>/{result.json, result.csv, result.tex, result.md, manifest.json}."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Dict

from .data import out_dir, repo_root


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def params_hash(recipe: str, knot: str, params: Dict) -> str:
    s = json.dumps({"recipe": recipe, "knot": knot, "params": params}, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _rows(result: Dict):
    if result["recipe"] == "perturbative_series":
        for sid, sec in result["sectors"].items():
            for n, c in enumerate(sec.get("coefficients", [])):
                yield {"sector": sid, "power": f"h^{n}", "coefficient": c}
    else:
        for e, d in result["entries"].items():
            for k, c in enumerate(d["coefficients"]):
                yield {"entry": e, "power": f"qt^{k}" if d["variable"] == "qt" else
                       (f"qt^-{k}" if d["variable"] == "1/qt" else "const"), "coefficient": c}


def to_csv(result: Dict) -> str:
    rows = list(_rows(result))
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue()


def _tex_escape_series(coeffs, var: str, nmax: int = 4) -> str:
    terms = []
    for n, c in enumerate(coeffs[: nmax + 1]):
        if c in ("0", 0):
            continue
        c = str(c).replace("*", " ").replace("I", "i").replace("xt", r"\tilde x").replace("^(", "^{").replace(")", ")")
        if var == "const":
            terms.append(f"{c}")
        else:
            terms.append(f"\\left({c}\\right) {var}^{{{n}}}" if n else f"\\left({c}\\right)")
    return " + ".join(terms) + (f" + O({var}^{{{nmax + 1}}})" if var != "const" else "")


def to_tex(result: Dict) -> str:
    lines = [f"% kdb {result['recipe']} for {result['knot']} -- generated, first terms only",
             f"% params: {json.dumps(result['params'])}"]
    if result["recipe"] == "perturbative_series":
        for sid, sec in result["sectors"].items():
            lines.append(f"% sector {sid}")
            lines.append(r"\varphi^{(" + sid + r")}(h) = " + _tex_escape_series(sec.get("coefficients", []), "h"))
    else:
        for e, d in result["entries"].items():
            var = {"qt": r"\tilde q", "1/qt": r"\tilde q^{-1}", "constant": "const"}[d["variable"]]
            lines.append(r"\mathsf S_{" + e + "} = " + _tex_escape_series(d["coefficients"], var))
    return "\n".join(lines) + "\n"


def to_md(result: Dict) -> str:
    out = [f"# {result['recipe']} - {result['knot']}", "", f"Parameters: `{json.dumps(result['params'])}`", ""]
    for k, v in result.get("conventions", {}).items():
        out.append(f"* **{k}**: {v}")
    out.append("")
    if result["recipe"] == "perturbative_series":
        for sid, sec in result["sectors"].items():
            out.append(f"## sector {sid}")
            for key in ("name", "method", "status", "normalisation"):
                if key in sec:
                    out.append(f"* {key}: {sec[key]}")
            out.append("")
            out.append("| n | coefficient of h^n |")
            out.append("|---|---|")
            for n, c in enumerate(sec.get("coefficients", [])[:12]):
                out.append(f"| {n} | `{c[:120]}` |")
            out.append("")
    else:
        for e, d in result["entries"].items():
            out.append(f"## S_{e} ({d['variable']})")
            out.append("| k | coefficient |")
            out.append("|---|---|")
            for k, c in enumerate(d["coefficients"]):
                out.append(f"| {k} | `{c}` |")
            out.append("")
    return "\n".join(out)


def write_bundle(result: Dict, base: Path | None = None) -> Path:
    base = base or out_dir()
    h = params_hash(result["recipe"], result["knot"], result["params"])
    d = base / result["recipe"] / h
    d.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    result.setdefault("meta", {})
    result["meta"].update({"git_commit": git_commit(), "hash": h})
    files = {
        "result.json": json.dumps(result, indent=1),
        "result.csv": to_csv(result),
        "result.tex": to_tex(result),
        "result.md": to_md(result),
    }
    for name, text in files.items():
        (d / name).write_text(text)
    manifest = {"recipe": result["recipe"], "knot": result["knot"], "params": result["params"],
                "hash": h, "git_commit": result["meta"]["git_commit"], "version": result.get("version"),
                "files": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in files.items()}}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return d
