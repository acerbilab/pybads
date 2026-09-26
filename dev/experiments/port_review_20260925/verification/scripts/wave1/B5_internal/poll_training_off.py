import collections
import inspect

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.bads as bb
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

cnt = collections.Counter()
orig_rec = bb.BADS._record_gp_refit_


def rec(self):
    cnt[
        "recorded refits ("
        + inspect.currentframe().f_back.f_back.f_code.co_name
        + ")"
    ] += 1
    return orig_rec(self)


bb.BADS._record_gp_refit_ = rec
orig_lgf = gpt.local_gp_fitting


def lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
    if refit_flag:
        cnt[
            "actual refits ("
            + inspect.currentframe().f_back.f_code.co_name
            + ")"
        ] += 1
    return orig_lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)


bb.local_gp_fitting = lgf


def ell(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


for pt in [True, False]:
    cnt.clear()
    b = BADS(
        ell,
        np.full((1, 4), 1.5),
        np.full((1, 4), -10.0),
        np.full((1, 4), 10.0),
        np.full((1, 4), -3.0),
        np.full((1, 4), 3.0),
        options={
            "random_seed": 0,
            "display": "off",
            "max_fun_evals": 200,
            "poll_training": pt,
        },
    )
    r = b.optimize()
    print("poll_training", pt, f"fval {r['fval']:.3g}", dict(cnt))
