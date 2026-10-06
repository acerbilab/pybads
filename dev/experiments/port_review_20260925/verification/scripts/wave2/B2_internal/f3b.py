import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


D = 2
try:
    BADS(
        sphere,
        np.full((1, D), 0.7),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options=dict(
            sloppy_improvement=False,
            random_seed=0,
            display="off",
            max_fun_evals=100,
        ),
    ).optimize()
except Exception:
    traceback.print_exc(limit=-2)
