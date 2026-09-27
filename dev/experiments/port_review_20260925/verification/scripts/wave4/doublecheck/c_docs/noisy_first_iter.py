"""Noisy runs that end within their first iteration (W4-14's changelog entry):
max_iter=1 and a max_fun_evals that the initial design nearly uses up, at
levels 1 and 2, at whichever pybads PYTHONPATH selects."""

import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__, flush=True)

D = 2
cases = [
    ("max_iter=1, level 1", {"max_iter": 1}, False),
    ("max_fun_evals=38, level 1", {"max_fun_evals": 38}, False),
    ("max_fun_evals=45, level 1", {"max_fun_evals": 45}, False),
    ("max_iter=1, level 2", {"max_iter": 1}, True),
    ("max_fun_evals=38, level 2", {"max_fun_evals": 38}, True),
    ("max_iter=2, level 1", {"max_iter": 2}, False),
]
for label, opts, stn in cases:
    noise_rng = np.random.default_rng(1)

    def fun(x):
        f = float(np.sum(np.ravel(x) ** 2)) + 0.5 * noise_rng.normal()
        return (f, 0.5) if stn else f

    options = {
        "display": "off",
        "random_seed": 0,
        "uncertainty_handling": True,
        "specify_target_noise": stn,
    }
    options.update(opts)
    try:
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
        yv = r["yval_vec"]
        ysd = r["ysd_vec"]
        print(
            f"{label:28s} iterations={r['iterations']} func_count={r['func_count']} "
            f"fval={r['fval']:.4f} fsd={r['fsd']:.4f} "
            f"yval_vec size={None if yv is None else np.size(yv)} "
            f"ysd_vec size={None if ysd is None else np.size(ysd)} "
            f"x={np.round(r['x'], 4)} status={r['status']}",
            flush=True,
        )
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(
            f"{label:28s} {type(e).__name__}: {e} [{tb.filename.split('/pybads/')[-1]}:{tb.lineno}]",
            flush=True,
        )
