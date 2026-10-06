import logging
import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.WARNING)
D = 2
f = lambda x: float(np.sum(x**2))
x0 = np.full(D, 0.5)
lb = np.full(D, -5.0)
ub = np.full(D, 5.0)
plb = np.full(D, -2.0)
pub = np.full(D, 2.0)
for label, opts in [
    ("f_vals", dict(f_vals=[f(x0)])),
    (
        "fun_values",
        dict(
            fun_values={
                "X": np.array([[0.1, 0.2], [0.3, -0.1]]),
                "Y": np.array([[0.05], [0.1]]),
            }
        ),
    ),
]:
    try:
        b = BADS(
            f,
            x0,
            lb,
            ub,
            plb,
            pub,
            options=dict(
                display="off", random_seed=0, max_fun_evals=40, **opts
            ),
        )
        print(label, "cache_active:", b.optim_state["cache_active"])
        r = b.optimize()
        print(label, "ran:", r["fval"], r["func_count"])
    except Exception as e:
        print(label, "raises", type(e).__name__, ":", e)
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(
            "   at",
            tb.filename.split("pybads-review/")[-1],
            tb.lineno,
            tb.line,
        )
