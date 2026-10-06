import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.bads.bads as bm
from pybads import BADS

orig_cc = bm.contraints_check
orig_hedge_call = bm.ESSearchHedge.__call__
calls = []


def cc(U, *a, **k):
    out = orig_cc(U, *a, **k)
    calls.append((np.shape(U), np.shape(out)))
    return out


bm.contraints_check = cc


def hc(self, u, *a, **k):
    us, z = orig_hedge_call(self, u, *a, **k)
    calls.append(("hedge", np.shape(u), np.shape(us), np.shape(z)))
    return us, z


bm.ESSearchHedge.__call__ = hc
f = lambda x: np.sum(x**2)
b = BADS(
    f,
    np.array([1.0, -0.5, 0.3]),
    np.full(3, -5.0),
    np.full(3, 5.0),
    np.full(3, -2.0),
    np.full(3, 2.0),
    options={"random_seed": 1, "max_fun_evals": 60, "display": "off"},
)
r = b.optimize()
print(r["fval"], r["func_count"])
from collections import Counter

print(Counter(map(str, calls)).most_common(10))
