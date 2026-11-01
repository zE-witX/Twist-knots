# kdb: a knot resurgence database for hyperbolic twist knots

`kdb` records, in executable form, the perturbative series of complex Chern-Simons theory around its flat
connections and the Stokes data that relate them. It covers the hyperbolic twist knots K_p
(4_1 = K_-1, 5_2 = K_2, 6_1 = K_-2, 7_2 = K_3, ...). Conjectural statements are recorded with their status.
They are not tested. The golden tests only check that the code reproduces published numbers.

Two recipes are implemented:

| recipe | output |
|---|---|
| `perturbative_series` | the series phi_sigma(x, h), h = 2 pi i tau, of the trivial sector and of the non-trivial sectors, without prefactor (the exponential and one-loop data are reported separately) |
| `stokes_series` | the Stokes automorphisms between the cones of the radial formula, as exact q~-series whose coefficients are Laurent polynomials in x~ (4_1: all four cones; 7_2: the upper sweep) |

Python is the reference engine. SageMath and Wolfram Language versions of both recipes follow the same
step list (`recipes/<name>/algorithm.md`) and run against the same golden tests.

## Install

```
pip install -e .            # needs python-flint, sympy, mpmath, pyyaml
kdb check --quick           # golden tests of every installed engine + identities
kdb check                   # everything, including the high-order cases (a few minutes)
```

`kdb check` detects `sage` and `wolframscript` automatically. You can also point to them with the
environment variables `KDB_SAGE` and `KDB_WOLFRAMSCRIPT`. A missing engine is reported as *untested*.
`pytest` runs the same tests (`pytest --runslow` adds the slow ones).

## Usage

```
kdb run perturbative_series --knot 4_1 --sector g --order 12 --x 1
kdb run perturbative_series --twist 3 --sector 0 --order 20 --x symbol
kdb run perturbative_series --knot 6_1 --sector all --order 10 --x 6/5 --digits 50
kdb run stokes_series --knot 4_1 --from-cone I --to-cone II --entry 0g --order 8 --x symbol
kdb show 4_1              # knot definition (aliases, classical data, sectors, sources)
kdb modules               # data modules and their status
kdb emit stokes_series --engine wolfram --params '{"order": 6, "x": "symbol"}' -o s.wl
kdb tex out/stokes_series/<hash>
kdb data build            # write the data items under data/knots/...
```

Sectors: `0` (trivial), and `g`, `c` for 4_1. Other knots use `s1, s2, ...`, ordered by the volume of
the sector: geometric, conjugate, the other complex pairs, then real sectors. `all` selects every
non-trivial sector. `--x symbol` gives exact rational functions of x. A rational `--x` gives exact or
high-precision values.

## Result bundles

A run writes `out/<recipe>/<hash>/`:

* `result.json`: the full result. Exact numbers are strings that sympy and Mathematica can parse. Numerics carry their digits. Meta data include the knot, group, side, recipe, params, conventions, version and git commit.
* `result.csv`: one row per coefficient.
* `result.tex`: the formula with its first terms.
* `result.md`: a readable summary.
* `manifest.json`: the parameters and SHA-256 hashes of the files.

`out/` and `data/` are git-ignored by default.

## Layout

```
src/kdb/        cli, registry, data, algebra (exact q-series), blocks, series/, stokes/, bundle, engines/, modules/
src/kdb/knots/  knot definitions (YAML): aliases, classical data, sectors, sources
src/kdb/modules/<kind>/  schema.json, module.yaml, README.md for each data kind
recipes/<name>/ card.yaml, algorithm.md, run.py, run.sage, run.wl, tests/golden.yaml
tests/          pytest wrappers around the golden tests and identities
data/knots/<knot>/<group>/<side>/<kind>.json   generated data (git-ignored)
```

Every datum is indexed by `(knot, group, side, kind)`:

* `group` is `SL2` for now.
* `side` is `cs` for the Chern-Simons side. `dual` is reserved.
* Each item carries `source`, `status` (`known`, `conjectural`, `ours`, `partial` or `unknown`), `digits` and `refs`.

## Adding a knot

1. Create `src/kdb/knots/<id>.yaml` with `schema: knot/1`, `twist_p`, `aliases`, the `classical` items (each with `source` and `status`), the `sectors`, and a `state_integral` description.
2. If the knot is a twist knot, nothing else is needed: the trivial sector uses the Masbaum coefficients, and the non-trivial sectors use the generic two-dimensional saddle engine with Marino's integral for K_p. Any K_p also works without a file (`--twist p`).
3. For another integrand, describe it as a list of Faddeev-dilogarithm factors with a Gaussian and a prefactor (`kdb.series.saddle.Integrand`). The engine finds the formal Gaussian expansion around any critical point.
4. Add golden cases to `recipes/perturbative_series/tests/golden.yaml` and run `kdb check`.

## Adding a module

Create `src/kdb/modules/<kind>/` with `module.yaml` (kind, status, description, recipes), `schema.json`
(versioned, `<kind>/1`) and `README.md`. Then add the recipe that produces the data. Existing modules
do not change. `kdb modules` lists the registry.

These stub modules are reserved for future work: `stokes_curves`, `qdifference_module`, `exponential_network`, `group_slN`, `dual_side`.

## References (labels used in the code)

* GGMW: S. Garoufalidis, J. Gu, M. Marino, C. Wheeler, *Resurgence of Chern-Simons theory at the trivial flat connection*, arXiv:2111.04763.
* GGM: S. Garoufalidis, J. Gu, M. Marino, *Peacock patterns and resurgence in complex Chern-Simons theory*, arXiv:2012.00062.
* Masbaum: G. Masbaum, *Skein-theoretical derivation of some formulas of Habiro*, Algebr. Geom. Topol. 3 (2003).
