"""Fix agent D's note: a 4-D ridge 10*sum|dx| + |sum x| started at
x0 = 1.5 * ones stalls at x0 (f = 6, 84 evaluations) for three seeds."""
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)


def ridge(x):
    x = np.ravel(x)
    return float(10 * np.sum(np.abs(np.diff(x))) + abs(np.sum(x)))


for seed in (1, 2, 3):
    b = BADS(
        ridge,
        1.5 * np.ones(4),
        -5 * np.ones(4),
        5 * np.ones(4),
        -3 * np.ones(4),
        3 * np.ones(4),
        options={"display": "off", "random_seed": seed, "max_fun_evals": 200},
    )
    r = b.optimize()
    print(
        f"seed {seed}: fval {r['fval']:.4g}, {r['func_count']} evaluations, "
        f"x {np.round(r['x'], 4)}",
        flush=True,
    )
