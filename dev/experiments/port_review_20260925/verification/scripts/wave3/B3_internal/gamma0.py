import traceback

import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.search.search_hedge as sh
from pybads import BADS

seen = []
orig = sh.ESSearchHedge.update_hedge


def upd(self, u_search, *a, **k):
    for i in range(self.n_funs):
        seen.append(
            (i, np.shape(u_search[np.minimum(i, len(u_search) - 1) :]))
        )
    return orig(self, u_search, *a, **k)


sh.ESSearchHedge.update_hedge = upd
f = lambda x: float(np.sum(x**2))
D = 3
try:
    b = BADS(
        f,
        np.full(D, 1.3),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options={
            "random_seed": 1,
            "max_fun_evals": 60,
            "display": "off",
            "hedge_gamma": 0,
        },
    )
    r = b.optimize()
    print("finished", r["fval"])
except Exception as e:
    traceback.print_exc(limit=3)
print("u_hedge shapes (strategy, shape):", seen[:4])
