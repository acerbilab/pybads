import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.search.search_hedge as sh
from pybads import BADS

orig = sh.ESSearchHedge.__call__


def wrapped(self, *a, **k):
    us, z = orig(self, *a, **k)
    if not hasattr(wrapped, "done"):
        print(
            "search returns",
            type(us),
            np.shape(us),
            np.shape(z),
            "u shape",
            np.shape(a[0]),
        )
        wrapped.done = True
    return us, z


sh.ESSearchHedge.__call__ = wrapped
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
print(r["fval"], r["func_count"])
