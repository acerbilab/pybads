"""B7 verifier: does a deterministic default run depend on random_seed at
all once the design is fixed? Same x0, three seeds; and random x0."""
import warnings

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)


def quad(x):
    x = np.atleast_2d(x)
    return float(np.sum((x - 0.1) ** 2 * np.arange(1, x.size + 1)))


def rosen(x):
    x = np.atleast_2d(x).ravel()
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


for fname, fun in (("quad", quad), ("rosenbrock", rosen)):
    for D in (3,):
        res = []
        for seed in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b = BADS(
                    fun,
                    0.4 * np.ones((1, D)),
                    -5 * np.ones((1, D)),
                    5 * np.ones((1, D)),
                    -2 * np.ones((1, D)),
                    2 * np.ones((1, D)),
                    options={
                        "display": "off",
                        "random_seed": seed,
                        "max_fun_evals": 200,
                    },
                )
                r = b.optimize()
            lg = b.function_logger
            res.append((r["func_count"], lg.X[: lg.Xn + 1].copy(), r["x"]))
            print(
                f"{fname} D={D} seed {seed}: func_count {r['func_count']}, "
                f"x {np.round(r['x'], 6).tolist()}, fval {r['fval']:.3g}",
                flush=True,
            )
        n0 = min(len(res[0][1]), len(res[1][1]))
        same = [
            np.array_equal(res[0][1][:k], res[1][1][:k])
            for k in range(1, n0 + 1)
        ]
        first_diff = same.index(False) if False in same else None
        print(
            f"  seeds 0 and 1: evaluated points identical up to row "
            f"{first_diff} (None = all {n0} rows identical)",
            flush=True,
        )
