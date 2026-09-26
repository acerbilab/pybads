"""uncertain_incumbent=False on a deterministic target: the target branch of
_get_target_from_gp_ (bads.py:2696-2699) returns Python floats, and the
callers call .item() on them (bads.py:1748, 2245)."""
import traceback

import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
from pybads import BADS

f = lambda x: float(np.sum(np.ravel(x) ** 2))
g = lambda x: np.sum(np.ravel(x) ** 2)
for fun, name in [(f, "returns float"), (g, "returns np.float64")]:
    try:
        b = BADS(
            fun,
            np.ones(3) * 2,
            -10 * np.ones(3),
            10 * np.ones(3),
            -5 * np.ones(3),
            5 * np.ones(3),
            options=dict(
                uncertain_incumbent=False,
                random_seed=0,
                display="off",
                max_fun_evals=60,
            ),
        )
        r = b.optimize()
        print(name, "-> fval", r["fval"], "nfev", r["func_count"])
    except Exception as e:
        print(name, "->", type(e).__name__, e)
        tb = traceback.extract_tb(e.__traceback__)[-2:]
        for fr in tb:
            print(
                "   ",
                fr.filename.split("pybads-review/")[-1],
                fr.lineno,
                fr.line,
            )
