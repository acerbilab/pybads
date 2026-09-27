import warnings

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)
from pybads.init_functions import init_sobol

warnings.simplefilter("error")
print(
    "D : design size at fun_eval_start = 1..2D (default fun_eval_start = D marked *)"
)
for D in [1, 2, 3, 4, 5, 8, 16]:
    row = []
    for fes in range(1, 2 * D + 1):
        u, m = init_sobol(
            np.zeros(D),
            None,
            None,
            -np.ones((1, D)),
            np.ones((1, D)),
            fes,
            rng=np.random.default_rng(0),
        )
        row.append(f"{fes}{'*' if fes == D else ''}:{u.shape[0]}")
    print(D, " ".join(row), flush=True)
for D in [1, 2, 4, 16, 20, 32]:
    fes = max(20, D)
    u, m = init_sobol(
        np.zeros(D),
        None,
        None,
        -np.ones((1, D)),
        np.ones((1, D)),
        fes,
        rng=np.random.default_rng(0),
    )
    print(
        f"noisy, D={D}: fun_eval_start {fes} -> {u.shape[0]} points; second return value {m}",
        flush=True,
    )
