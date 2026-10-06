"""The runs of dev/scripts/fingerprint.py, each hashed on its own, with
func_count, fval and the init_N of the first refits."""
import hashlib

import numpy as np

import pybads
from pybads import BADS


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


g = np.random.default_rng(0)


def fn(x):
    return float(np.sum(np.atleast_2d(x) ** 2) + g.standard_normal())


print(pybads.__file__, flush=True)
h_all = hashlib.sha256()
for fun, noisy in [(f, False), (fn, True)]:
    for seed in range(3):
        o = {"display": "off", "max_fun_evals": 81, "random_seed": seed}
        if noisy:
            o["uncertainty_handling"] = True
        b = BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        )
        r = b.optimize()
        h = hashlib.sha256()
        for hh in (h, h_all):
            hh.update(np.asarray(r["x"], dtype=float).tobytes())
            hh.update(np.float64(r["fval"]).tobytes())
            hh.update(np.int64(r["func_count"]).tobytes())
            if r["yval_vec"] is not None:
                hh.update(np.asarray(r["yval_vec"], dtype=float).tobytes())
        init_N = [v for v in b.iteration_history["init_N"] if v is not None]
        print(
            "noisy" if noisy else "det",
            seed,
            h.hexdigest()[:16],
            r["func_count"],
            repr(r["fval"]),
            "init_N:",
            list(init_N)[:8],
            flush=True,
        )
print("all", h_all.hexdigest()[:16], flush=True)
