import numpy as np

import pybads.bads.bads as bm
from pybads import BADS

D = 3


def sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


b = BADS(
    sphere,
    np.ones(D) * 4,
    -100 * np.ones(D),
    100 * np.ones(D),
    -8 * np.ones(D),
    12 * np.ones(D),
    options={"display": "off", "max_fun_evals": 60, "random_seed": 3},
)
orig = bm.acq_fcn_lcb
log = []


def acq(u, fc, gp):
    log.append((b.optim_state["search_count"], len(u)))
    return orig(u, fc, gp)


bm.acq_fcn_lcb = acq
r = b.optimize()
print(log)
print(r["func_count"], r["iterations"])
