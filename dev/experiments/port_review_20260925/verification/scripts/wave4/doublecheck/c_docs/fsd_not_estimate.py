"""The returned fsd of noisy runs that take no final samples, besides a stop
by output_fcn at "init" (OptimizeResult's description of fsd, W4-30):
max_fun_evals=1, and a max_fun_evals that the start and the design use up."""

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__, flush=True)

D = 2
for label, opts in [
    (
        "max_fun_evals=1, uh=True",
        {"max_fun_evals": 1, "uncertainty_handling": True},
    ),
    (
        "max_fun_evals=1, uh=True, noise_size=2.5",
        {"max_fun_evals": 1, "uncertainty_handling": True, "noise_size": 2.5},
    ),
    (
        "max_fun_evals=33, uh=True",
        {"max_fun_evals": 33, "uncertainty_handling": True},
    ),
    (
        "max_fun_evals=33, uh=True, noise_size=2.5",
        {"max_fun_evals": 33, "uncertainty_handling": True, "noise_size": 2.5},
    ),
    (
        "max_fun_evals=34, uh=True, noise_size=2.5",
        {"max_fun_evals": 34, "uncertainty_handling": True, "noise_size": 2.5},
    ),
    (
        "output_fcn init stop, noise_size=2.5",
        {
            "uncertainty_handling": True,
            "noise_size": 2.5,
            "output_fcn": lambda x, s, st: st == "init",
        },
    ),
]:
    noise_rng = np.random.default_rng(1)

    def fun(x):
        return float(np.sum(np.ravel(x) ** 2)) + 0.5 * noise_rng.normal()

    options = {"display": "off", "random_seed": 0}
    options.update(opts)
    b = BADS(
        fun,
        np.array([1.0, 1.5]),
        -5 * np.ones(D),
        5 * np.ones(D),
        -3 * np.ones(D),
        3 * np.ones(D),
        options=options,
    )
    r = b.optimize()
    print(
        f"{label:44s} iterations={r['iterations']} func_count={r['func_count']} "
        f"fsd={r['fsd']} yval_vec={r['yval_vec']} status={r['status']} "
        f"msg={r['message'][:60]!r}",
        flush=True,
    )
