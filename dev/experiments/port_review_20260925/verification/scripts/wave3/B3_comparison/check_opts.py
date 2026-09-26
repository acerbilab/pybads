import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads import BADS

f = lambda x: float(np.sum(np.ravel(x) ** 2))
for opts in [
    {"search_acq_fcn": ("acq_LCB", 1.0)},
    {"search_acq_fcn": ("acq_LCB", np.float64(1.0))},
    {"hedge_gamma": 0.0},
]:
    o = {"random_seed": 0, "display": "off", "max_fun_evals": 60}
    o.update(opts)
    try:
        r = BADS(
            f,
            np.full(3, 3.0),
            np.full(3, -20.0),
            np.full(3, 20.0),
            np.full(3, -5.0),
            np.full(3, 5.0),
            options=o,
        ).optimize()
        print(opts, "-> ran, fval", r["fval"])
    except Exception as e:
        print(opts, "->", type(e).__name__, str(e).splitlines()[0][:100])
