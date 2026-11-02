#!/usr/bin/env python3
"""Python (reference) engine of the recipe index3d.  Usage: python run.py '{"knot": "7_2", "order": 6, "flux": [0]}'."""
import json
import sys

from kdb.recipes import run_index3d

if __name__ == "__main__":
    params = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"knot": "4_1", "order": 6, "flux": [0]}
    print(json.dumps(run_index3d(params), indent=1))
