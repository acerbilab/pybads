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

reasons = collections.Counter()
orig_lgf = gpt.local_gp_fitting


def lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
    fr = inspect.currentframe().f_back
    fn = fr.f_code.co_name
    self = fr.f_locals.get("self")
    if fn == "_search_step_" and fr.f_locals.get("new_gp") is None:
        if refit_flag:
            r = "search: refit"
        elif gp.temporary_data.get("needs_rebuild"):
            r = "search: needs_rebuild"
        elif optim_state["search_count"] == 0:
            r = "search: first of round"
        else:
            r = "search: only reset_gp (stale after an earlier rebuild?)"
        reasons[r] += 1
    elif fn == "_poll_step_":
        pc = fr.f_locals["poll_count"]
        if refit_flag:
            r = "poll: refit"
        elif pc == 0:
            r = "poll: poll_count 0"
        elif gp.temporary_data.get("needs_rebuild"):
            r = "poll: needs_rebuild"
        else:
            r = "poll: poll_count>0, only reset_gp"
        reasons[r] += 1
    else:
        reasons[fn + " (other)"] += 1
    return orig_lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)


bb.local_gp_fitting = lgf


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ell(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


for name, f, D in [("rosen2", rosen, 2), ("ell4", ell, 4)]:
    for seed in [0, 1]:
        reasons.clear()
        b = BADS(
            f,
            np.full((1, D), 1.5),
            np.full((1, D), -10.0),
            np.full((1, D), 10.0),
            np.full((1, D), -3.0),
            np.full((1, D), 3.0),
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        )
        r = b.optimize()
        print(name, seed, f"fval {r['fval']:.3g}", dict(reasons))
