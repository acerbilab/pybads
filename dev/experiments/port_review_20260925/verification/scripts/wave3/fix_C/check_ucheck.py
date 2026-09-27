"""contraints_check against the verifier's transcription of uCheck.m, on
random candidate sets with evaluated points among them."""
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(
    0,
    "/home/user/pybads-fix-C/dev/experiments/port_review_20260925/verification/scripts/wave3/B3_verifier",
)
from v_ucheck import ucheck_matlab  # noqa: E402  (runs its own prints first)

import pybads
from pybads.function_logger.constraints_check import contraints_check

print(pybads.__file__)
rng = np.random.default_rng(1)
agree = total = 0
removed = 0
for trial in range(500):
    D = rng.integers(1, 5)
    grid = 2.0 ** -rng.integers(2, 7)
    U = (
        np.round(rng.uniform(-1.2, 1.2, size=(rng.integers(1, 60), D)) / grid)
        * grid
    )
    n_eval = rng.integers(0, 20)
    X_eval = np.vstack(
        [
            U[rng.integers(0, len(U), size=n_eval // 2)],
            np.round(rng.uniform(-1, 1, size=(n_eval - n_eval // 2, D)) / grid)
            * grid,
        ]
    )
    fl = SimpleNamespace(
        X=np.vstack([X_eval, np.full((3, D), np.nan)]),
        X_max_idx=len(X_eval) - 1,
    )
    lb_s, ub_s = -np.ones((1, D)), np.ones((1, D))
    tol_mesh = 2.0 ** -rng.integers(8, 20)
    for proj in (True, False):
        p = contraints_check(U, lb_s, ub_s, tol_mesh, fl, proj)
        m = ucheck_matlab(U, tol_mesh, X_eval, lb_s, ub_s, lb_s, ub_s, proj)
        total += 1
        agree += p.shape == m.shape and np.array_equal(p, m)
print(f"agree {agree}/{total}")
