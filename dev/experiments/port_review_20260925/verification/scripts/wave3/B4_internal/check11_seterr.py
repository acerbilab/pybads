"""NumPy's global error state after a BADS run; argmin of an array with
NaN."""
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

print("before:", np.geterr())


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


D = 2
BADS(
    sphere,
    np.full((1, D), 0.5),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    options={"random_seed": 0, "display": "off", "max_fun_evals": 40},
).optimize()
print("after:", np.geterr())
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    np.array([1.0]) / np.array([0.0])
    print("warnings on 1/0 after the run:", len(w))
z = np.array([[3.0], [np.nan], [1.0]])
i = np.argmin(z)
print("argmin with NaN:", i, "size", i.size, "finite", np.isfinite(i))
