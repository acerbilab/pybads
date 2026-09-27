"""Place x0 on a point of the initial Sobol design and count the calls of the
target at the same point during initialization (the noise test's second call
of x0 left out)."""
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)
D = 2


def make(x0, calls):
    def f(x):
        calls.append(np.array(x, dtype=float).ravel().copy())
        return float(np.sum((np.atleast_2d(x) - 0.3) ** 2))

    return BADS(
        f,
        x0,
        lower_bounds=np.full(D, -5.0),
        upper_bounds=np.full(D, 5.0),
        plausible_lower_bounds=np.full(D, -4.0),
        plausible_upper_bounds=np.full(D, 4.0),
        options={"display": "off", "random_seed": 0, "max_fun_evals": 12},
    )


calls = []
b = make(np.full(D, 1.0), calls)
b.optimize()
n0 = int(b.optim_state["eff_starting_points"])
design = np.array(calls[: n0 + 1])
print(
    "initialization calls from x0=[1,1]:", design.round(4).tolist(), flush=True
)
# a design point other than x0
cand = [p for p in design[2:] if not np.allclose(p, 1.0)]
x0 = cand[0]
calls = []
b = make(x0, calls)
b.optimize()
n0 = int(b.optim_state["eff_starting_points"])
init = np.array(calls[: n0 + 1])
print("x0 on a design point", x0, flush=True)
print("initialization calls:", init.round(4).tolist(), flush=True)
same = int(np.sum(np.all(np.isclose(init, x0), axis=1)))
print("calls at x0 during initialization:", same, flush=True)
