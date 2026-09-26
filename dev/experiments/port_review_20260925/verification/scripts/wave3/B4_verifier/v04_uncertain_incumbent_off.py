"""I-F4 / C-F3: uncertain_incumbent=False on a deterministic target."""
import traceback

import numpy as np
from vhdr import box, sphere

from pybads import BADS

D = 2
lb, ub, plb, pub = box(D)
for label, f in (
    ("float", sphere),
    ("np.float64", lambda x: np.float64(sphere(x))),
):
    try:
        r = BADS(
            f,
            np.ones(D),
            lb,
            ub,
            plb,
            pub,
            options=dict(
                random_seed=0,
                display="off",
                max_fun_evals=100,
                uncertain_incumbent=False,
            ),
        ).optimize()
        print(label, "ran:", r["fval"])
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        print(
            label,
            "->",
            type(e).__name__,
            e,
            "| at",
            [f"{t.name}:{t.lineno}" for t in tb[-2:]],
        )
# the same with uncertainty_handling=False (level 0 declared)
try:
    BADS(
        sphere,
        np.ones(D),
        lb,
        ub,
        plb,
        pub,
        options=dict(
            random_seed=0,
            display="off",
            max_fun_evals=100,
            uncertain_incumbent=False,
            uncertainty_handling=False,
        ),
    ).optimize()
except Exception as e:
    print("uncertainty_handling=False ->", type(e).__name__, e)
# level 1 is unaffected: the branch is not taken
rng = np.random.default_rng(0)
r = BADS(
    lambda x: sphere(x) + 0.1 * rng.normal(),
    np.ones(D),
    lb,
    ub,
    plb,
    pub,
    options=dict(
        random_seed=0,
        display="off",
        max_fun_evals=100,
        uncertain_incumbent=False,
        uncertainty_handling=True,
    ),
).optimize()
print("level 1 with uncertain_incumbent=False ran, fval =", r["fval"])
