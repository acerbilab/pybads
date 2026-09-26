import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


D = 2
x0 = np.full((1, D), 0.7)
for disp in ("iter", "off"):
    try:
        r = BADS(
            sphere,
            x0,
            -5 * np.ones((1, D)),
            5 * np.ones((1, D)),
            -2 * np.ones((1, D)),
            2 * np.ones((1, D)),
            options=dict(
                f_vals=[sphere(x0)],
                random_seed=0,
                display=disp,
                max_fun_evals=30,
            ),
        ).optimize()
        print(disp, "ok", r["func_count"])
    except Exception as e:
        print(f"display={disp}: raised {type(e).__name__}: {e}")
        traceback.print_exc(limit=-2)
