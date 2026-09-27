"""Do the fingerprint's six runs meet ties at the two ES sorts?"""
import numpy as np

import pybads
import pybads.search.es_search as es
from pybads import BADS

print(pybads.__file__)
stats = {
    "calls": 0,
    "pool_diff": 0,
    "pool_ties": 0,
    "y_calls": 0,
    "y_diff": 0,
    "y_ties": 0,
}
orig_lcb = es.acq_fcn_lcb
orig_call = es.ESSearch.__call__
orig_init = es.ESSearchWM._initialize_
cur = []


def lcb(u, *a, **k):
    out = orig_lcb(u, *a, **k)
    cur.append(np.ravel(out[0]).copy())
    z = np.concatenate(cur)
    stats["calls"] += 1
    stats["pool_ties"] += z.size - np.unique(z).size
    stats["pool_diff"] += not np.array_equal(
        np.argsort(z), np.argsort(z, kind="stable")
    )
    return out


def call(self, *a, **k):
    cur.clear()
    return orig_call(self, *a, **k)


def init(self, u, gp, *a, **k):
    y = gp.y.ravel()
    stats["y_calls"] += 1
    stats["y_ties"] += y.size - np.unique(y).size
    stats["y_diff"] += not np.array_equal(
        np.argsort(y), np.argsort(y, kind="stable")
    )
    return orig_init(self, u, gp, *a, **k)


es.acq_fcn_lcb = lcb
es.ESSearch.__call__ = call
es.ESSearchWM._initialize_ = init


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


g = np.random.default_rng(0)


def fn(x):
    return float(np.sum(np.atleast_2d(x) ** 2) + g.standard_normal())


for fun, noisy in [(f, False), (fn, True)]:
    for seed in range(3):
        o = {"display": "off", "max_fun_evals": 80, "random_seed": seed}
        if noisy:
            o["uncertainty_handling"] = True
        BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        ).optimize()
print(stats)
