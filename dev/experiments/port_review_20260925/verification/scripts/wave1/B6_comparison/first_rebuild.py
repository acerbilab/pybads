"""Is the first rebuild of a default run a refit? (If not, the hyperparameters fitted at initialization
under _gp_hyp's priors, a fit MATLAB does not do, are those of the first search.)"""
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bb
from pybads import BADS

orig = bb.local_gp_fitting
calls = []


def wrapped(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
    before = gp.get_hyperparameters(as_array=True).copy()
    out = orig(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)
    calls.append(
        (
            fl.func_count,
            bool(refit_flag),
            np.allclose(before, out[0].get_hyperparameters(as_array=True)),
        )
    )
    return out


bb.local_gp_fitting = wrapped
for D, fun in [
    (3, lambda x: float(np.sum(np.asarray(x) ** 2))),
    (6, lambda x: float(np.sum(np.abs(np.asarray(x)) ** 1.5))),
]:
    calls.clear()
    x0 = np.full(D, 0.5)
    lb = np.full(D, -5.0)
    ub = np.full(D, 5.0)
    plb = np.full(D, -1.0)
    pub = np.full(D, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bads = BADS(
            fun,
            x0,
            lb,
            ub,
            plb,
            pub,
            options={"random_seed": 0, "max_fun_evals": 60, "display": "off"},
        )
        bads.optimize()
    print(
        f"D={D}: first rebuilds (func_count, refit, hyp unchanged):", calls[:6]
    )
