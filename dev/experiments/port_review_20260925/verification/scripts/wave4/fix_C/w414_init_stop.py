import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
rng = np.random.default_rng(0)


def fun(x):
    return float(np.sum(np.ravel(x) ** 2)) + rng.normal()


for tn in (False,):
    b = BADS(
        fun,
        np.ones(3),
        -10 * np.ones(3),
        10 * np.ones(3),
        -5 * np.ones(3),
        5 * np.ones(3),
        options={
            "display": "off",
            "random_seed": 0,
            "uncertainty_handling": True,
            "output_fcn": lambda x, s, st: st == "init",
        },
    )
    r = b.optimize()
    print(
        r["iterations"],
        r["func_count"],
        r["yval_vec"],
        b.yval,
        r["fval"],
        r["fsd"],
        r["ysd_vec"],
    )
