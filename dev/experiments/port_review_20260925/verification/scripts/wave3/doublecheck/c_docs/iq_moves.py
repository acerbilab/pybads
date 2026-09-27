"""Count the moves of the incumbent after initialization in 1.1.0 at
improvement_quantile 0 and 1, deterministic and noisy."""
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)

moves = []
orig = BADS._update_incumbent_


def counting(self, *a, **k):
    moves.append(1)
    return orig(self, *a, **k)


BADS._update_incumbent_ = counting


def sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


for noisy in (False, True):
    for q in (0, 1, 0.5):
        moves.clear()
        rng = np.random.default_rng(1)
        fun = (lambda x: sphere(x) + rng.normal()) if noisy else sphere
        opts = dict(
            display="off",
            random_seed=0,
            max_fun_evals=150 if noisy else 100,
            improvement_quantile=q,
        )
        if noisy:
            opts.update(uncertainty_handling=True, noise_size=1.0)
        D = 2
        b = BADS(
            fun,
            np.full(D, 2.0),
            lower_bounds=np.full(D, -5.0),
            upper_bounds=np.full(D, 5.0),
            plausible_lower_bounds=np.full(D, -4.0),
            plausible_upper_bounds=np.full(D, 4.0),
            options=opts,
        )
        r = b.optimize()
        fl = b.function_logger
        n = fl.X_max_idx + 1 if hasattr(fl, "X_max_idx") else None
        print(
            f"noisy={noisy} q={q}: incumbent updates after init {len(moves)}, "
            f"x {np.round(r['x'], 4)}, iterations {r['iterations']}",
            flush=True,
        )
