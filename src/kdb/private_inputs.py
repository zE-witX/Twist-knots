"""Read-only access to Pengxu's private data files for golden tests.

The files stay where they are (``../inputs`` next to the repository, or ``$KDB_PRIVATE_INPUTS``); nothing
is copied into the repository.  Mathematica ``InputForm`` text is parsed, never evaluated.
Tests that need these files are skipped when the directory is absent.
"""
from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .data import private_inputs


def _file(name: str) -> Optional[Path]:
    base = private_inputs()
    if base is None:
        return None
    f = base / "notebooks" / name
    return f if f.exists() else None


def _join(text: str) -> str:
    return text.replace("\\\n", "")


def _top_level_split(s: str) -> List[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [t.strip() for t in out]


_NUM = r"(\d+\.?\d*|\.\d+)(?:`{1,2}[\d.]*)?(?:\*\^([+-]?\d+))?"


def _parse_real(tok: str) -> Tuple[str, int]:
    """Mathematica real token -> (decimal string, exponent)."""
    m = re.fullmatch(_NUM, tok)
    if not m:
        raise ValueError(f"cannot parse number {tok!r}")
    return m.group(1), int(m.group(2) or 0)


def parse_complex(expr: str):
    """'a`p + b`p*I', '0``p - b`p*I', '1' -> (re, im) as decimal strings with exponent applied (mpmath-ready)."""
    e = expr.replace(" ", "").replace("\n", "")
    # split into signed terms
    terms = re.findall(r"[+-]?[^+-]+(?:\*\^[+-]?\d+[^+-]*)?", e)
    re_s, im_s = "0", "0"
    for t in terms:
        sign = -1 if t.startswith("-") else 1
        t = t.lstrip("+-")
        is_im = t.endswith("*I")
        if is_im:
            t = t[:-2]
        if t == "I":
            val = "1"
        else:
            mant, ex = _parse_real(t)
            val = mant + (f"e{ex}" if ex else "")
        val = ("-" if sign < 0 else "") + val
        if is_im:
            im_s = val
        else:
            re_s = val
    return re_s, im_s


def load_phi_gc_series() -> Optional[Dict[str, Dict[str, List[Tuple[str, str]]]]]:
    """Phi_gc_series.dat (schema PhiGC/1): {x: {'RootPlus': [(re, im), ...], 'RootMinus': [...]}}."""
    f = _file("Phi_gc_series.dat")
    if f is None:
        return None
    text = _join(f.read_text())
    out = {}
    for m in re.finditer(r'"([0-9/]+)" -> <\|"x" -> ', text):
        key = m.group(1)
        rec = {}
        for root in ("RootPlus", "RootMinus"):
            i = text.index(f'"{root}" -> {{', m.end())
            j = i + len(f'"{root}" -> {{')
            depth, k = 1, j
            while depth:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                k += 1
            rec[root] = [parse_complex(t) for t in _top_level_split(text[j:k - 1])]
        mp = re.search(r'"MinimumStoredPrecision" -> \s*([0-9.]+)', text[m.end():])
        rec["MinimumStoredPrecision"] = float(mp.group(1)) if mp else None
        out[key] = rec
    return out


def load_phi1x1d280(nmax: int = None) -> Optional[List[Tuple[Fraction, bool]]]:
    """Phi1x1d280.dat 'res': list of (rational r, has_sqrt3) meaning r or r/sqrt(3)."""
    f = _file("Phi1x1d280.dat")
    if f is None:
        return None
    text = _join(f.read_text())
    body = text[text.index("{") + 1: text.rindex("}")]
    out = []
    for t in _top_level_split(body):
        t = re.sub(r"\s+", "", t)
        m = re.fullmatch(r"(-?\d+)/\((\d+)\*Sqrt\[3\]\)", t)
        if m:
            out.append((Fraction(int(m.group(1)), int(m.group(2))), True))
        else:
            m = re.fullmatch(r"(-?\d+)(?:/(\d+))?", t)
            if not m:
                raise ValueError(f"unparsed entry {t[:60]!r}")
            out.append((Fraction(int(m.group(1)), int(m.group(2) or 1)), False))
        if nmax and len(out) >= nmax:
            break
    return out


def _stream_list(f: Path, nmax: int) -> List[str]:
    """First nmax top-level entries of 'name = {e1, e2, ...}' without reading the whole structure."""
    out, depth, cur, started = [], 0, [], False
    with f.open() as fh:
        prev = ""
        while True:
            chunk = fh.read(1 << 16)
            if not chunk:
                break
            chunk = (prev + chunk)
            prev = ""
            if chunk.endswith("\\"):
                prev, chunk = "\\", chunk[:-1]
            chunk = chunk.replace("\\\n", "")
            for ch in chunk:
                if not started:
                    if ch == "{":
                        started = True
                    continue
                if ch in "{[(":
                    depth += 1
                elif ch in "}])":
                    if depth == 0:
                        out.append("".join(cur).strip())
                        return out
                    depth -= 1
                if ch == "," and depth == 0:
                    out.append("".join(cur).strip())
                    cur = []
                    if len(out) >= nmax:
                        return out
                else:
                    cur.append(ch)
    return out


def load_PDzlist(nmax: int):
    """Phi0x100.dat 'PDzlist' first nmax entries as sympy expressions in z."""
    f = _file("Phi0x100.dat")
    if f is None:
        return None
    import sympy as sp
    z = sp.symbols("z")
    return [sp.sympify(t.replace("^", "**"), locals={"z": z}) for t in _stream_list(f, nmax)]


def load_hpoly2(lmax: int):
    """Phi0n200.dat: {l: sympy polynomial in n} for l <= lmax (h^{2l} coefficient of J_n(e^h) of 4_1)."""
    f = _file("Phi0n200.dat")
    if f is None:
        return None
    import sympy as sp
    n = sp.symbols("n")
    out = {}
    buf = ""
    with f.open() as fh:
        for line in fh:
            buf += line
            if line.strip() == "":
                m = re.match(r"\s*hpoly2\[(\d+)\]\[n_\]\s*:?=\s*(.*)", buf.replace("\\\n", ""), re.S)
                if m:
                    l = int(m.group(1))
                    if l > lmax:
                        break
                    out[l] = sp.sympify(m.group(2).replace("^", "**").replace("\n", " "), locals={"n": n})
                buf = ""
    return out
