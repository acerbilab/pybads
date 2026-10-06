"""N4 after W2-4: a start with a coordinate of +-inf and finite bounds."""
import logging

import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
logging.getLogger("BADS").setLevel(logging.ERROR)
f = lambda x: float(np.sum(np.ravel(x) ** 2))
for x0, lb, ub in [
    ([np.inf, 0.0], [-2.0, -2.0], [2.0, 2.0]),
    ([-np.inf, 0.0], [-2.0, -2.0], [2.0, 2.0]),
    ([np.nan, 0.0], [-2.0, -2.0], [2.0, 2.0]),
    ([np.inf, 0.0], [-np.inf, -np.inf], [np.inf, np.inf]),
]:
    try:
        b = BADS(
            f,
            np.array(x0),
            np.array(lb),
            np.array(ub),
            -np.ones(2),
            np.ones(2),
            options={"display": "off", "random_seed": 3},
        )
        print(x0, lb, "-> accepted, x0 =", b.x0)
    except Exception as e:
        print(x0, lb, "-> refused:", type(e).__name__, str(e).split("\n")[0])
for x0 in ([np.nan, 100.0], [np.inf, 100.0], [0.0, 100.0]):
    try:
        b = BADS(
            f,
            np.array(x0),
            -2 * np.ones(2),
            2 * np.ones(2),
            -np.ones(2),
            np.ones(2),
            options={"display": "off", "random_seed": 3},
        )
        print(x0, "-> accepted, x0 =", b.x0)
    except Exception as e:
        print(x0, "-> refused:", type(e).__name__, str(e).split("\n")[0])
