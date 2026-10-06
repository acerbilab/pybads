import warnings

import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)


def fun(x):
    x = np.atleast_2d(x)
    return float((np.log(x[0, 0]) - 1) ** 2 + ((x[0, 1] - 800) / 100) ** 2)


warnings.simplefilter("always")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    bads = BADS(
        fun,
        np.array([10.0, 750.0]),
        np.array([1.0, -1000.0]),
        np.array([1000.0, 1000.0]),
        np.array([2.0, -900.0]),
        np.array([500.0, 900.0]),
        options={"display": "off", "max_fun_evals": 30, "random_seed": 1},
    )
    print("init warnings", [(str(x.message), x.filename, x.lineno) for x in w])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r = bads.optimize()
    print(
        "run warnings",
        sorted(
            set(
                (str(x.message), x.filename.split("/")[-1], x.lineno)
                for x in w
            )
        ),
    )
print(r["x"], r["fval"])
