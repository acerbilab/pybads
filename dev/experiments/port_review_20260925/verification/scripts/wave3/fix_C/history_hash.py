"""Per-run hashes of the whole evaluation history of the fingerprint's runs."""
import hashlib

import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


g = np.random.default_rng(0)


def fn(x):
    return float(np.sum(np.atleast_2d(x) ** 2) + g.standard_normal())


for fun, noisy in [(f, False), (fn, True)]:
    for seed in range(3):
        o = {"display": "off", "max_fun_evals": 80, "random_seed": seed}
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
        fl = b.function_logger
        X = fl.X[: fl.Xn + 1]
        h = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()[:12]
        hf = hashlib.sha256(
            np.asarray(r["x"], float).tobytes()
            + np.float64(r["fval"]).tobytes()
        ).hexdigest()[:12]
        print(noisy, seed, "evals", X.shape[0], "history", h, "final", hf)
