#!/usr/bin/env python3
"""Python (reference) engine of the recipe stokes_series.  Usage: python run.py '{"knot": "4_1", ...}'."""
import json
import sys

from kdb.recipes import run_stokes_series

if __name__ == "__main__":
    params = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"knot": "4_1", "from_cone": "I", "to_cone": "II", "order": 4}
    print(json.dumps(run_stokes_series(params), indent=1))
