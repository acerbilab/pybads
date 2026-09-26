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
try:
    b = BADS(
        f,
        np.full(D, 1.0),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options=dict(
            display="off",
            random_seed=0,
            max_fun_evals=60,
            sloppy_improvement=False,
        ),
    )
    r = b.optimize()
    print("ran", r["fval"], r["func_count"], r["message"])
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)[-1]
    print("raises", type(e).__name__, e, "at", tb.lineno, tb.line)
