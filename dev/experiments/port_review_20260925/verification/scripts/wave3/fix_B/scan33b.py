import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
D = 3


def noisy_sd(seed):
    rng = np.random.default_rng(seed)

    def fun(x):
        y = float(np.sum(np.atleast_2d(x) ** 2))
        sd = 1.0 + 0.1 * np.sqrt(y)
        sd_estimate = sd * np.exp(0.3 * rng.standard_normal())
        return y + sd * rng.standard_normal(), sd_estimate

    return fun


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
            noisy_sd(0),
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
                "specify_target_noise": True,
                "output_fcn": out,
            },
        )
        run["b"].optimize()
        print(evals, seed, len(polls), polls.count(False))
