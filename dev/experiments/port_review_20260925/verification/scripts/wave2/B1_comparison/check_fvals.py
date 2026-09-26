import logging

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)
calls = []


def sph(x):
    calls.append(1)
    return float(np.sum(np.ravel(x) ** 2))


b = BADS(
    sph,
    x0=np.ones(2),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-2 * np.ones(2),
    plausible_upper_bounds=2 * np.ones(2),
    options={
        "random_seed": 0,
        "display": "off",
        "f_vals": [2.0],
        "max_fun_evals": 30,
    },
)
print("cache_active:", b.optim_state["cache_active"])
try:
    r = b.optimize()
    print("ran", r["func_count"])
except Exception as e:
    print(
        "optimize():",
        type(e).__name__,
        e,
        "; target calls so far:",
        len(calls),
    )
