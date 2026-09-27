"""Where x0 = [1, 3] starts in a plausible box [-2048, 2048]."""
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)
seen = []


def f(x):
    seen.append(np.array(x, dtype=float).ravel())
    return float(np.sum(np.atleast_2d(x) ** 2))


for lb in (4096.0, 2048.0, np.inf):
    seen.clear()
    b = BADS(
        f,
        np.array([1.0, 3.0]),
        lower_bounds=np.full(2, -lb),
        upper_bounds=np.full(2, lb),
        plausible_lower_bounds=np.full(2, -2048.0),
        plausible_upper_bounds=np.full(2, 2048.0),
        options={"display": "off", "random_seed": 0, "max_fun_evals": 20},
    )
    print(
        f"hard bounds +-{lb}: u0 as x",
        np.ravel(b.var_transf.inverse_transf(np.atleast_2d(b.u))),
        flush=True,
    )
    b.optimize()
    print("  first evaluated x", seen[0], flush=True)
