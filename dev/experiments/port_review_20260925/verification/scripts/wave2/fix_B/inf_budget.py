import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


for mfe in [np.inf, 0, -5, 30.5, 30.0, "200*D", None]:
    try:
        r = BADS(
            f,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options={
                "display": "off",
                "max_fun_evals": mfe,
                "random_seed": 0,
                "max_iter": 10,
            },
        ).optimize()
        print(repr(mfe), "ran", r["func_count"], r["message"])
    except Exception as e:
        print(repr(mfe), type(e).__name__, e)
