import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)


def ridge(x):
    x = np.ravel(x)
    return float(10 * np.sum(np.abs(np.diff(x))) + abs(np.sum(x)))


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


for name, f, D in [
    ("ridge D2", ridge, 2),
    ("ridge D4", ridge, 4),
    ("rosen D3", rosen, 3),
]:
    out = []
    for seed in [1, 2, 3]:
        r = BADS(
            f,
            np.full((1, D), 1.5),
            -5 * np.ones((1, D)),
            5 * np.ones((1, D)),
            -2 * np.ones((1, D)),
            2 * np.ones((1, D)),
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        ).optimize()
        out.append("%.3g/%d" % (r["fval"], r["func_count"]))
    print(name, " ".join(out))
