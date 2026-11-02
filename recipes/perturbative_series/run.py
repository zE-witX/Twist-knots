#!/usr/bin/env python3
"""Python (reference) engine of the recipe perturbative_series.  Usage: python run.py '{"knot": "4_1", ...}'."""
import json
import sys

from kdb.recipes import run_perturbative_series

if __name__ == "__main__":
    params = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"knot": "4_1", "sector": "0", "order": 4}
    print(json.dumps(run_perturbative_series(params), indent=1))
