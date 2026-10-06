"""The cases of test_half_bounded_variables over seeds 0-19: the error of
the result against the test's tolerance (0.05) and the evaluations."""
import logging

import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
logging.getLogger("BADS").setLevel(logging.ERROR)
inf = np.inf


def q(x):
    x = np.asarray(x).ravel()
    return float((x[0] - 0.3) ** 2 + 0.1 * (x[1] - 2.0) ** 2)


cases = {
    "bounded_below": ([0.0, -inf], [inf, inf], [0.1, -3.0], [0.9, 3.0]),
    "bounded_above": ([-inf, -inf], [1.0, 10.0], [0.1, -3.0], [0.9, 3.0]),
    "bounded_below_log": ([1e-3, 0.5], [inf, inf], [1e-2, 1.0], [1.0, 20.0]),
}
for name, bounds in cases.items():
    errs, evals = [], []
    for seed in range(20):
        b = BADS(
            q,
            np.array([0.5, 1.0]),
            *[np.array(v) for v in bounds],
            options={
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": 100,
            },
        )
        r = b.optimize()
        errs.append(np.max(np.abs(np.ravel(r["x"]) - [0.3, 2.0])))
        evals.append(r["func_count"])
        if seed == 1:
            print(f"  {name} seed 1: x {r['x']} evals {r['func_count']}")
    print(
        f"{name:18s} max|err| over seeds 0-19: max {max(errs):.3g}, median {np.median(errs):.3g}; evals {min(evals)}-{max(evals)}"
    )
