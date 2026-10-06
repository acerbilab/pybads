"""Repeated evaluations in the initial design: x0 at the centre of the
plausible box, and a run on a sphere with its minimum on a bound."""
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)

for D in (2, 3):
    for x0v in (0.0, 1.0):
        calls = []

        def f(x):
            calls.append(np.array(x, dtype=float).ravel().copy())
            return float(np.sum((np.atleast_2d(x) - 0.3) ** 2))

        b = BADS(
            f,
            np.full(D, x0v),
            lower_bounds=np.full(D, -5.0),
            upper_bounds=np.full(D, 5.0),
            plausible_lower_bounds=np.full(D, -4.0),
            plausible_upper_bounds=np.full(D, 4.0),
            options={
                "display": "off",
                "random_seed": 0,
                "max_fun_evals": 60,
                "uncertainty_handling": False,
            },
        )
        b.optimize()
        n_init = (
            int(b.optim_state["eff_starting_points"])
            if "eff_starting_points" in b.optim_state
            else None
        )
        X = np.array(calls)
        init = X[: (n_init if n_init else 0)]
        dup_init = len(init) - len(np.unique(init, axis=0))
        dup_all = len(X) - len(np.unique(X, axis=0))
        print(
            f"D={D} x0={x0v}: initial evaluations {len(init)}, repeated among them {dup_init}; "
            f"whole run {len(X)} evaluations, repeated {dup_all}",
            flush=True,
        )
