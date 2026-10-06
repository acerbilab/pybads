import logging
import traceback

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.ERROR)
D = 2
det = lambda x: float(np.sum(x**2) + 0.3 * np.sum(np.cos(3 * x)))


def noisy_maker():
    r = np.random.default_rng(0)
    return lambda x: float(np.sum(x**2) + 0.3 * r.normal())


cases = [
    ("search_mesh_expand=1", dict(search_mesh_expand=1), False),
    ("search_size_locked=False", dict(search_size_locked=False), False),
    (
        "skip_poll_after_search=False",
        dict(skip_poll_after_search=False),
        False,
    ),
    ("accelerate_mesh=False", dict(accelerate_mesh=False), False),
    (
        "improvement_quantile=0.2 noisy",
        dict(improvement_quantile=0.2, uncertainty_handling=True),
        True,
    ),
    (
        "final_quantile=0.5 noisy",
        dict(final_quantile=0.5, uncertainty_handling=True),
        True,
    ),
    (
        "noise_final_samples=0 noisy",
        dict(noise_final_samples=0, uncertainty_handling=True),
        True,
    ),
    ("restarts=1", dict(restarts=1), False),
    ("init_mesh_size_integer=-2", dict(init_mesh_size_integer=-2), False),
    ("max_poll_grid_number=2", dict(max_poll_grid_number=2), False),
    ("tol_stall_iters=1", dict(tol_stall_iters=1), False),
    ("complete_poll=True", dict(complete_poll=True), False),
    ("fun_eval_start=0", dict(fun_eval_start=0), False),
]
for label, opts, noisy in cases:
    f = noisy_maker() if noisy else det
    try:
        b = BADS(
            f,
            np.full(D, 1.0),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options=dict(
                display="off", random_seed=0, max_fun_evals=80, **opts
            ),
        )
        r = b.optimize()
        print(
            f"{label}: ok, func_count {r['func_count']}, iterations {r['iterations']}, fval {float(r['fval']):.4g}, msg: {r['message'][:60]}"
        )
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(
            f"{label}: raises {type(e).__name__}: {e} at {tb.filename.split('/')[-1]}:{tb.lineno}"
        )
