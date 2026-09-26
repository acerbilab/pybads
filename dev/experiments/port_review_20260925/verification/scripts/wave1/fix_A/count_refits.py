import numpy as np

import pybads.bads.bads as bb
from pybads import BADS

C = {"rec": 0, "done": 0}
orig = bb.local_gp_fitting


def lgf(*a, **k):
    C["done"] += bool(a[6])
    return orig(*a, **k)


bb.local_gp_fitting = lgf
orig_rec = BADS._record_gp_refit_


def rec(self):
    C["rec"] += 1
    return orig_rec(self)


BADS._record_gp_refit_ = rec


def sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


D = 3
for name, fun, x0, lb, ub, plb, pub in [
    (
        "sphere",
        sphere,
        4 * np.ones(D),
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
    ),
    (
        "ell",
        ell,
        0.5 * np.ones(D),
        -5 * np.ones(D),
        5 * np.ones(D),
        -2 * np.ones(D),
        2 * np.ones(D),
    ),
]:
    for seed in (0, 3, 70):
        for mfe in (60, 100, 150):
            C.update(rec=0, done=0)
            r = BADS(
                fun,
                x0,
                lb,
                ub,
                plb,
                pub,
                options={
                    "random_seed": seed,
                    "display": "off",
                    "max_fun_evals": mfe,
                    "poll_training": False,
                },
            ).optimize()
            print(name, seed, mfe, C, r["func_count"], r["iterations"])
