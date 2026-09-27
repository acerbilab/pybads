import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import pybads
from pybads import BADS

print(pybads.__file__)
D = 3


def noisy(seed):
    rng = np.random.default_rng(seed)
    return (
        lambda x: float(np.sum(np.atleast_2d(x) ** 2)) + rng.standard_normal()
    )


for evals in (100, 150):
    for seed in (7, 0, 1, 2, 3):
        run = {}
        polls = []

        def out(x, s, state):
            if state == "iter":
                b = run["b"]
                polls.append(
                    np.array_equal(
                        [s[k] for k in ("yval", "fval", "fsd")],
                        [b.yval, b.fval, b.fsd],
                        equal_nan=True,
                    )
                )
            return False

        run["b"] = BADS(
            noisy(0),
            np.ones(D) * 4,
            -100 * np.ones(D),
            100 * np.ones(D),
            -8 * np.ones(D),
            12 * np.ones(D),
            options={
                "display": "off",
                "max_fun_evals": evals,
                "random_seed": seed,
                "uncertainty_handling": True,
                "output_fcn": out,
            },
        )
        run["b"].optimize()
        print(evals, seed, len(polls), polls.count(False))
