import sys

import numpy as np

import pybads.bads.bads as bm
from pybads import BADS

D = int(sys.argv[1]) if len(sys.argv) > 1 else 3
mfe = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


b = BADS(
    sphere,
    np.ones(D) * 4,
    -100 * np.ones(D),
    100 * np.ones(D),
    -8 * np.ones(D),
    12 * np.ones(D),
    options={"display": "off", "max_fun_evals": mfe, "random_seed": 3},
)
of = bm.local_gp_fitting
os_ = BADS._search_step_
op = BADS._poll_step_
fits = []
steps = []


def fitting(*a, **k):
    fits.append(a[6])
    return of(*a, **k)


def step(orig, kind):
    def w(self, gp):
        u = self.u_best.copy()
        n = len(fits)
        out = orig(self, gp)
        steps.append((kind, not np.array_equal(u, self.u_best), fits[n:]))
        return out

    return w


bm.local_gp_fitting = fitting
BADS._search_step_ = step(os_, "search")
BADS._poll_step_ = step(op, "poll")
r = b.optimize()
for s in steps:
    print(s)
print(r["func_count"], r["iterations"], r["message"])
