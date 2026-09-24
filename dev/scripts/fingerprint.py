"""Hash of six seeded PyBADS runs: equal before and after a change that
must not move results. Run from the repository root."""
import hashlib

import numpy as np

import pybads
from pybads import BADS


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


g = np.random.default_rng(0)


def fn(x):
    return float(np.sum(np.atleast_2d(x) ** 2) + g.standard_normal())


h = hashlib.sha256()
for fun, noisy in [(f, False), (fn, True)]:
    for seed in range(3):
        o = {"display": "off", "max_fun_evals": 80, "random_seed": seed}
        if noisy:
            o["uncertainty_handling"] = True
        r = BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        ).optimize()
        h.update(np.asarray(r["x"], dtype=float).tobytes())
        h.update(np.float64(r["fval"]).tobytes())
        h.update(np.int64(r["func_count"]).tobytes())
        if r["yval_vec"] is not None:
            h.update(np.asarray(r["yval_vec"], dtype=float).tobytes())
print(pybads.__file__, h.hexdigest()[:16])
