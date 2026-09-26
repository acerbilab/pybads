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
last = {"u": None}
orig_lgf = gpt.local_gp_fitting


def lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
    fr = inspect.currentframe().f_back
    fn = fr.f_code.co_name
    main = (
        fn == "_search_step_" and fr.f_locals.get("new_gp") is None
    ) or fn == "_poll_step_"
    if main:
        only_reset = (
            not refit_flag
            and not gp.temporary_data.get("needs_rebuild")
            and not (
                fn == "_search_step_" and optim_state["search_count"] == 0
            )
            and not (fn == "_poll_step_" and fr.f_locals["poll_count"] == 0)
        )
        if only_reset:
            same = last["u"] is not None and np.array_equal(
                np.ravel(u), last["u"]
            )
            reasons[
                f"{fn}: reset_gp only, incumbent {'UNCHANGED' if same else 'moved'} since last rebuild"
            ] += 1
        last["u"] = np.ravel(u).copy()
    return orig_lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)


bb.local_gp_fitting = lgf


def ell(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


b = BADS(
    ell,
    np.full((1, 4), 1.5),
    np.full((1, 4), -10.0),
    np.full((1, 4), 10.0),
    np.full((1, 4), -3.0),
    np.full((1, 4), 3.0),
    options={"random_seed": 0, "display": "off", "max_fun_evals": 200},
)
r = b.optimize()
print("ell4 seed 0", f"fval {r['fval']:.3g}")
for k, v in reasons.items():
    print("  ", k, v)
