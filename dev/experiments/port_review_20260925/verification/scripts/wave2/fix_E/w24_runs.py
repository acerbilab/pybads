"""Sanity runs after W2-4 (not a gate): problems that reach the old margin."""
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.ERROR)


def logsph(x):
    return float(np.sum((np.log10(np.ravel(x)) + 2) ** 2))


cases = [
    (
        "sphere [-5,5]^2, plb omitted",
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        [1.0, 1.0],
        [-5.0] * 2,
        [5.0] * 2,
        None,
        None,
        {},
    ),
    (
        "optimum at lb, x0 on lb, plb omitted",
        lambda x: float(np.sum((np.ravel(x) + 5) ** 2)),
        [-5.0, 0.0],
        [-5.0] * 2,
        [5.0] * 2,
        None,
        None,
        {},
    ),
    (
        "log 3-D, plb omitted",
        logsph,
        [1.0] * 3,
        [1e-3] * 3,
        [1e3] * 3,
        None,
        None,
        {},
    ),
    (
        "log 1-D, x0 = optimum = plb",
        logsph,
        [1e-2],
        [1e-3],
        [1e3],
        [1e-2],
        [1e2],
        {},
    ),
    (
        "log 2-D noisy, plb omitted",
        lambda x: logsph(x)
        + 0.1
        * np.random.default_rng(
            int(1e6 * np.sum(x)) % 2**32
        ).standard_normal(),
        [1.0] * 2,
        [1e-3] * 2,
        [1e3] * 2,
        None,
        None,
        {"uncertainty_handling": True},
    ),
]
a = lambda v: None if v is None else np.array(v, float)
for name, fun, x0, lb, ub, plb, pub, opt in cases:
    for seed in range(3):
        b = BADS(
            fun,
            a(x0),
            a(lb),
            a(ub),
            a(plb),
            a(pub),
            options={
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": 200,
                **opt,
            },
        )
        r = b.optimize()
        print(
            f"{name:40s} seed {seed}: x {np.round(r['x'], 5)} fval {r['fval']:.3g} evals {r['func_count']} | {r['message'][:60]}"
        )
