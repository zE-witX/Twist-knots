"""Command line interface ``kdb``.

    kdb run perturbative_series --knot 4_1 --sector g --order 12 --x 1 --frame P
    kdb run perturbative_series --twist 3 --sector 0 --order 20 --x symbol
    kdb run stokes_series --knot 4_1 --from-cone I --to-cone II --entry 0g --order 8 --x symbol --frame P
    kdb show 4_1 | kdb modules | kdb check [--quick] [--engine python|sage|wolfram]
    kdb emit <recipe> --engine python|sage|wolfram --params '{"knot": "4_1", ...}' -o file
    kdb tex <bundle-dir> | kdb data build [--knot ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _run(args):
    from .bundle import write_bundle
    from .recipes import run_perturbative_series, run_stokes_series
    params = {k: v for k, v in vars(args).items() if v is not None and k not in ("cmd", "recipe", "func", "no_bundle", "print")}
    if args.recipe == "perturbative_series":
        res = run_perturbative_series(params)
    elif args.recipe == "stokes_series":
        res = run_stokes_series(params)
    else:
        sys.exit(f"unknown recipe {args.recipe}")
    if not args.no_bundle:
        d = write_bundle(res)
        print(f"bundle: {d}")
    if args.print or args.no_bundle:
        print(json.dumps(res, indent=1)[:20000])
    else:
        from .bundle import to_md
        print(to_md(res)[:4000])


def _show(args):
    import yaml
    from .data import load_knot, riley_polynomial
    k = load_knot(args.knot) if not args.knot.lstrip("-").isdigit() else load_knot(twist=int(args.knot))
    raw = dict(k.raw)
    if k.p is not None:
        raw.setdefault("derived", {})["riley_polynomial(Note A formula)"] = str(riley_polynomial(k.p))
    print(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))


def _modules(args):
    from .registry import modules
    for m in modules():
        print(f"{m['kind']:22s} {m['status']:22s} {m['description'][:90]}")


def _check(args):
    from .check import main as check_main
    sys.exit(check_main(quick=args.quick, engines=args.engine))


def _emit(args):
    from .engines.emit import emit
    params = json.loads(args.params) if args.params else {}
    path = emit(args.recipe, args.engine, params, args.output)
    print(f"wrote {path}")


def _tex(args):
    from .bundle import to_tex
    d = Path(args.bundle)
    res = json.loads((d / "result.json").read_text())
    t = to_tex(res)
    (d / "result.tex").write_text(t)
    print(t)


def _data(args):
    from .build_data import build
    for p in build(args.knot):
        print(p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kdb", description="knot resurgence database for hyperbolic twist knots")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a recipe and write a result bundle")
    r.add_argument("recipe", choices=["perturbative_series", "stokes_series"])
    r.add_argument("--knot")
    r.add_argument("--twist", type=int)
    r.add_argument("--sector")
    r.add_argument("--order", type=int)
    r.add_argument("--x")
    r.add_argument("--frame", choices=["standard", "P"])
    r.add_argument("--digits", type=int)
    r.add_argument("--method")
    r.add_argument("--from-cone", dest="from_cone", choices=["I", "II", "III", "IV"])
    r.add_argument("--to-cone", dest="to_cone", choices=["I", "II", "III", "IV"])
    r.add_argument("--entry")
    r.add_argument("--variant", choices=["corrected", "ggmw118"])
    r.add_argument("--no-bundle", action="store_true")
    r.add_argument("--print", action="store_true", help="print the full JSON")
    r.set_defaults(func=_run)

    s = sub.add_parser("show", help="show a knot definition")
    s.add_argument("knot")
    s.set_defaults(func=_show)

    m = sub.add_parser("modules", help="list data modules and their status")
    m.set_defaults(func=_modules)

    c = sub.add_parser("check", help="golden tests of every engine and the identities of CLAUDE.md section 3")
    c.add_argument("--quick", action="store_true", help="skip the slow high-order/private-data cases")
    c.add_argument("--engine", action="append", choices=["python", "sage", "wolfram"])
    c.set_defaults(func=_check)

    e = sub.add_parser("emit", help="write a standalone script for an engine")
    e.add_argument("recipe", choices=["perturbative_series", "stokes_series"])
    e.add_argument("--engine", required=True, choices=["python", "sage", "wolfram"])
    e.add_argument("--params", help="JSON dict of recipe parameters")
    e.add_argument("-o", "--output", required=True)
    e.set_defaults(func=_emit)

    t = sub.add_parser("tex", help="(re)generate result.tex of a bundle")
    t.add_argument("bundle")
    t.set_defaults(func=_tex)

    d = sub.add_parser("data", help="build the permanent data files under data/")
    d.add_argument("action", choices=["build"])
    d.add_argument("--knot", action="append")
    d.set_defaults(func=_data)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
