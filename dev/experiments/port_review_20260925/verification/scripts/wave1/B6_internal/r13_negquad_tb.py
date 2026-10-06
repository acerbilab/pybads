import traceback
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

D = 2
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        b = BADS(
            lambda x: float(10 + np.sum(np.atleast_1d(x) ** 2)),
            np.full(D, 1.5),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options=dict(
                display="off",
                random_seed=0,
                max_fun_evals=200,
                gp_mean_fun="negquad",
            ),
        )
        b.optimize()
    except Exception:
        tb = traceback.format_exc().splitlines()
        print("\n".join(tb[-12:]))
