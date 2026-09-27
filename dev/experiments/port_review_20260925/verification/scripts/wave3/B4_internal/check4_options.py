"""Runs with the non-default target options: uncertain_incumbent=False (level
0), alternative_incumbent=True, complete_poll=True, tol_poi=0,
min_failed_poll_steps finite, force_poll_mesh=True."""
import traceback

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


D = 3
lb = -5 * np.ones((1, D))
ub = 5 * np.ones((1, D))
plb = -2 * np.ones((1, D))
pub = 2 * np.ones((1, D))
x0 = np.full((1, D), 1.5)
for opts in [
    {"uncertain_incumbent": False},
    {"alternative_incumbent": True},
    {"complete_poll": True},
    {"tol_poi": 0.0},
    {"min_failed_poll_steps": 2},
    {"force_poll_mesh": True},
    {"min_failed_poll_steps": 2, "consecutive_skipping": False},
]:
    o = {"random_seed": 1, "display": "off", "max_fun_evals": 150}
    o.update(opts)
    try:
        r = BADS(rosen, x0, lb, ub, plb, pub, options=o).optimize()
        print(
            opts,
            "OK fval %.3g evals %d iters %d"
            % (r["fval"], r["func_count"], r["iterations"]),
        )
    except Exception as e:
        print(opts, "FAILED", type(e).__name__, e)
        traceback.print_exc(limit=-3)
