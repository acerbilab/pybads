import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.bads.bads as bb
from pybads import BADS

orig = bb.contraints_check
calls = []


def wrapped(U, *a, **k):
    out = orig(U, *a, **k)
    calls.append((np.shape(U), np.shape(out)))
    return out


bb.contraints_check = wrapped
f = lambda x: np.sum(x**2)
b = BADS(
    f,
    np.array([1.0, -1.0, 0.5]),
    np.full(3, -5.0),
    np.full(3, 5.0),
    np.full(3, -3.0),
    np.full(3, 3.0),
    options={"random_seed": 1, "display": "off", "max_fun_evals": 60},
)
r = b.optimize()
print(calls[:6])
