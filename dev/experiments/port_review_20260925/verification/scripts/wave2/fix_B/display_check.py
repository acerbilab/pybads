import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
rng = np.random.default_rng(0)


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def fn(x):
    return f(x) + rng.standard_normal()


for disp, fun, extra in [
    ("notify", f, {}),
    ("final", fn, {"uncertainty_handling": True}),
    ("final", f, {}),
]:
    print(f"==== display={disp} {fun.__name__}")
    BADS(
        fun,
        np.ones(3) * 4,
        -100 * np.ones(3),
        100 * np.ones(3),
        -8 * np.ones(3),
        12 * np.ones(3),
        options={
            "display": disp,
            "max_fun_evals": 50,
            "random_seed": 0,
            **extra,
        },
    ).optimize()
