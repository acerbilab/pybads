import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads import BADS

for D in [2, 3, 4]:
    fun = lambda x: float(np.sum(np.asarray(x).ravel() ** 2))
    plb, pub = np.full(D, -2.0), np.full(D, 2.0)
    b = BADS(
        fun,
        (plb + pub) / 2 + 0.0,
        np.full(D, -5.0),
        np.full(D, 5.0),
        plb,
        pub,
        options={
            "random_seed": 1,
            "max_fun_evals": 2 * D + 20,
            "display": "off",
        },
    )
    b._init_optimization_()
    fl = b.function_logger
    X = fl.X[: fl.Xn + 1]
    rows = [i for i in range(1, len(X)) if np.all(np.abs(X[i] - X[0]) < 1e-12)]
    print(
        f"D={D}: {len(X)} points after the initial design; rows equal to x0 (row 0): {rows}"
    )
