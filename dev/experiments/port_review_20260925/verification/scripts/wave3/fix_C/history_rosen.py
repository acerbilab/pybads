import hashlib

import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


for seed in range(3):
    b = BADS(
        rosen,
        np.zeros(3),
        np.full(3, -5.0),
        np.full(3, 5.0),
        np.full(3, -2.0),
        np.full(3, 2.0),
        options={"display": "off", "random_seed": seed, "max_fun_evals": 120},
    )
    r = b.optimize()
    fl = b.function_logger
    X = fl.X[: fl.Xn + 1]
    print(
        seed,
        X.shape[0],
        hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()[:12],
        r["fval"],
    )
