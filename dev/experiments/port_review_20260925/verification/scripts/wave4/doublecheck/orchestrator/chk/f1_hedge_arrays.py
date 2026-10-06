"""Doublecheck of wave 4, orchestrator: reviewer (b)'s F1 and F2, the checks
of hedge_gamma, hedge_beta, hedge_decay and n_search_iter on values of other
types, run at 81385ac (runs of at most 80 evaluations, D = 2)."""
import warnings
from decimal import Decimal
from fractions import Fraction

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__)


def run(name, value, evals=80):
    opts = {
        "display": "off",
        "max_fun_evals": evals,
        "random_seed": 0,
        name: value,
    }
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
        return f"refused at creation: {type(e).__name__}"
    stored = b.options[name]
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = b.optimize()
        kinds = sorted({type(x.message).__name__ for x in w})
        return f"accepted, stored {type(stored).__name__}; run completed: func_count={r['func_count']} warnings={kinds}"
    except Exception as e:
        return f"accepted, stored {type(stored).__name__}; run stopped: {type(e).__name__}: {str(e)[:60]}"


cases = [
    ("hedge_gamma", np.array([0.1])),
    ("hedge_gamma", np.array([[0.1]])),
    ("hedge_gamma", np.complex128(0.1 - 5j)),
    ("hedge_gamma", Decimal("0.25")),
    ("hedge_beta", np.array([1.0])),
    ("hedge_beta", np.array([[1.0]])),
    ("hedge_beta", np.array([True])),
    ("hedge_beta", Fraction(1, 4)),
    ("hedge_decay", np.array([0.5])),
    ("hedge_decay", np.array([[0.5]])),
    ("hedge_decay", np.array([True])),
    ("hedge_decay", np.complex128(0.5 + 1j)),
    ("n_search_iter", 2**70),
    ("n_search_iter", 1e308),
    ("n_search_iter", 5000),
    ("max_fun_evals", 2**70),
    ("accelerate_mesh_steps", 2**70),
]
for name, value in cases:
    print(f"{name}={value!r}: {run(name, value)}", flush=True)
