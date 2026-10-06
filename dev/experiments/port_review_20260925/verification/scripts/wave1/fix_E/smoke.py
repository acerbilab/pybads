import logging

import gpyreg as gpr
import numpy as np

import pybads
from pybads import BADS
from pybads.bads import gaussian_process_train as gpt

print(pybads.__file__, gpr.__file__)
logging.getLogger("BADS").setLevel(logging.ERROR)

flags = []
orig = gpt._robust_gp_fit_


def spy(*a, **k):
    out = orig(*a, **k)
    flags.append(out[3])
    return out


gpt._robust_gp_fit_ = spy
fails = {"n": 0}
orig_fit = gpr.GP.fit


def fit(self, *a, **k):
    try:
        return orig_fit(self, *a, **k)
    except np.linalg.LinAlgError:
        fails["n"] += 1
        raise


gpr.GP.fit = fit


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


for opts in ({}, {"use_slice_sampler": True, "double_refit": True}):
    for seed in range(3):
        flags.clear()
        fails["n"] = 0
        o = {"display": "off", "max_fun_evals": 150, "random_seed": seed}
        o.update(opts)
        r = BADS(
            rosen,
            np.zeros(3),
            -5 * np.ones(3),
            5 * np.ones(3),
            -2 * np.ones(3),
            2 * np.ones(3),
            options=o,
        ).optimize()
        print(
            opts,
            seed,
            "fval %.3g" % r["fval"],
            "refits",
            len(flags),
            "flags",
            sorted(set(flags)),
            "fit failures",
            fails["n"],
        )
