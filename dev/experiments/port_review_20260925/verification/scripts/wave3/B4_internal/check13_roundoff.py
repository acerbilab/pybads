"""The poll's division by poll_scale and multiplication back are exact
inverses only up to rounding: (1/s)*s can exceed 1 by an ulp, which puts a
poll point meant to lie on a hard bound just outside it, and the bounds
check drops it."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bm
from pybads import BADS

rng = np.random.default_rng(0)
s = np.exp(rng.uniform(-3, 3, 100000))
d = np.eye(1)
print(
    "fraction of s with (1/s)*s > 1:",
    np.mean((1.0 / s) * s > 1.0),
    " < 1:",
    np.mean((1.0 / s) * s < 1.0),
)


# a run where lb = plb and ub = pub (the plausible box is the hard box): the transformed bounds
def sphere(x):
    return float(np.sum((np.ravel(x) - 0.9) ** 2))


D = 3
b = BADS(
    sphere,
    np.full((1, D), 0.0),
    -np.ones((1, D)),
    np.ones((1, D)),
    options={"random_seed": 0, "display": "off", "max_fun_evals": 150},
)
print("u-space bounds:", b.lower_bounds, b.upper_bounds)
# Count poll points dropped only by rounding: displacement meant to reach a bound exactly
dropped = {"n": 0, "cand": 0}
orig_cc = bm.contraints_check


def cc(U, lb, ub, tol_mesh, fl, proj=True, nbc=None):
    if not proj:
        exact = (
            np.round(U / 2.0**-30) * 2.0**-30
        )  # the points without the rounding of the scaling
        out_exact = np.any(exact > ub, axis=1) | np.any(exact < lb, axis=1)
        out = np.any(U > ub, axis=1) | np.any(U < lb, axis=1)
        dropped["n"] += int(np.sum(out & ~out_exact))
        dropped["cand"] += len(U)
    return orig_cc(U, lb, ub, tol_mesh, fl, proj, nbc)


bm.contraints_check = cc
r = b.optimize()
print(
    "poll points dropped by the rounding alone:",
    dropped["n"],
    "of",
    dropped["cand"],
    "fval %.3g" % r["fval"],
    "x",
    r["x"],
)
