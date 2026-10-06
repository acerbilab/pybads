import logging
import sys

import gpyreg as gpr
import numpy as np

import pybads
from pybads import BADS
from pybads.bads import gaussian_process_train as gpt

print(pybads.__file__, gpr.__file__)
logging.getLogger("BADS").setLevel(logging.ERROR)

orig_ss = gpt._get_samples_from_slice_sampler_


def ss(gp, hyp, *a, **k):
    x0 = np.atleast_2d(hyp)
    lo = x0 < gp.lower_bounds
    hi = x0 > gp.upper_bounds
    caller = sys._getframe(1).f_code.co_name
    if lo.any() or hi.any():
        print(
            "  sampler start outside bounds, from",
            caller,
            "rows",
            x0.shape[0],
            "below",
            np.argwhere(lo).tolist(),
            "above",
            np.argwhere(hi).tolist(),
        )
        print("   start", np.round(x0, 3).tolist())
        print("   lb", np.round(gp.lower_bounds, 3).tolist())
        print("   ub", np.round(gp.upper_bounds, 3).tolist())
    return orig_ss(gp, hyp, *a, **k)


gpt._get_samples_from_slice_sampler_ = ss


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


for seed in range(3):
    o = {
        "display": "off",
        "max_fun_evals": 150,
        "random_seed": seed,
        "use_slice_sampler": True,
        "double_refit": True,
    }
    try:
        r = BADS(
            rosen,
            np.zeros(3),
            -5 * np.ones(3),
            5 * np.ones(3),
            -2 * np.ones(3),
            2 * np.ones(3),
            options=o,
        ).optimize()
        print(seed, "fval %.3g" % r["fval"])
    except ValueError as e:
        print(seed, "RAISED", e)
