import math

import numpy as np

import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

seen = {}
orig = gpt._gp_hyp


def spy(optim_state, options, plb, pub, gp, X, y, fl):
    out = orig(optim_state, options, plb, pub, gp, X, y, fl)
    seen["N"] = y.size
    seen["hyp0_mean"] = out[1][-1]
    ys = np.sort(y, axis=None)
    seen["round"] = np.median(ys[: round(0.8 * y.size)])
    seen["ceil"] = np.median(ys[: math.ceil(0.8 * y.size)])
    seen["prior"] = out[0].get_priors()["mean_const"][1][0].item()
    return out


gpt._gp_hyp = spy
for D in [1, 2, 3, 4, 5, 6, 7, 8]:
    b = BADS(
        lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={"display": "off", "random_seed": 3},
    )
    b._init_optimization_()
    print(D, seen)
