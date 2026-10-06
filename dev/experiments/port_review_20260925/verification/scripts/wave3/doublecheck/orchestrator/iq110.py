import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads import BADS

for noisy in [False, True]:
    for q in [0.0, 1.0]:
        g = np.random.default_rng(0)
        f = (
            (lambda x: float(np.sum(np.asarray(x) ** 2) + g.standard_normal()))
            if noisy
            else (lambda x: float(np.sum(np.asarray(x) ** 2)))
        )
        o = {
            "display": "off",
            "max_fun_evals": 100,
            "random_seed": 0,
            "improvement_quantile": q,
        }
        if noisy:
            o["uncertainty_handling"] = True
        b = BADS(
            f,
            np.ones(2) * 3,
            -10 * np.ones(2),
            10 * np.ones(2),
            -5 * np.ones(2),
            5 * np.ones(2),
            options=o,
        )
        n = [0]
        orig = b._update_incumbent_

        def wrap(*a, **k):
            n[0] += 1
            return orig(*a, **k)

        b._update_incumbent_ = wrap
        try:
            r = b.optimize()
            print(
                "noisy" if noisy else "det",
                q,
                "moves",
                n[0],
                "x",
                np.round(r["x"], 3),
                "iters",
                r["iterations"],
                "fevals",
                r["func_count"],
            )
        except Exception as e:
            print("noisy" if noisy else "det", q, type(e).__name__, e)
