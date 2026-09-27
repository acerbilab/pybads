import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
for v in [0, -1, 0.5, 2.5, np.inf, True, "2"]:
    try:
        b = BADS(
            lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
            np.array([0.5, 0.0]),
            -5 * np.ones(2),
            5 * np.ones(2),
            -3 * np.ones(2),
            3 * np.ones(2),
            options={
                "display": "off",
                "random_seed": 1,
                "n_search_iter": v,
                "max_fun_evals": 60,
            },
        )
        r = b.optimize()
        print(repr(v), "ran:", r["func_count"], r["fval"])
    except Exception as e:  # noqa: BLE001
        print(
            repr(v),
            type(e).__name__,
            str(e)[:80],
            "at func_count",
            b.function_logger.func_count,
        )
