"""Doublecheck of wave 4, orchestrator: what 1.1.0 (or the checkout on
PYTHONPATH) does with n_search and n_search_iter values that the ruling
refuses, and with integers beyond 64 bits (runs of at most 60 evaluations,
D = 2)."""
import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__)


def run(options):
    opts = {"display": "off", "max_fun_evals": 60, "random_seed": 0, **options}
    try:
        b = BADS(
            lambda x: float(np.sum(np.ravel(x) ** 2)),
            np.array([1.0, 1.5]),
            -5 * np.ones(2),
            5 * np.ones(2),
            -3 * np.ones(2),
            3 * np.ones(2),
            options=opts,
        )
    except Exception as e:
        return f"refused at creation: {type(e).__name__}: {str(e)[:70]}"
    try:
        r = b.optimize()
        return f"ran: func_count={r['func_count']} fval={r['fval']:.3g}"
    except Exception as e:
        return f"stopped: {type(e).__name__}: {str(e)[:70]}"


for options in [
    {"n_search_iter": 5000},
    {"n_search_iter": 4097},
    {"n_search_iter": 2**70},
    {"n_search": 0},
    {"n_search": -1},
    {"n_search": 2.5},
    {"n_search": 1},
    {"n_search": "4096"},
    {"n_search": np.nan},
    {"n_search": True},
    {"n_search": 10, "n_search_iter": 11},
    {"n_search": 10, "n_search_iter": 10},
    {"max_fun_evals": 2**70},
    {"accelerate_mesh_steps": 2**70},
]:
    print(options, "->", run(options), flush=True)
