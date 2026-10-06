import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.WARNING)
D = 2


def mk(seed, het):
    nrng = np.random.default_rng(seed)
    if het:

        def f(x):
            sd = 0.5 + 0.1 * np.sum(x**2)
            return float(np.sum(x**2) + sd * nrng.normal()), float(sd)

    else:
        f = lambda x: float(np.sum(x**2) + 0.5 * nrng.normal())
    return f


for label, het, extra in [
    ("level1 nfs=10", False, {}),
    ("level1 nfs=1", False, dict(noise_final_samples=1)),
    ("level2 nfs=10", True, dict(specify_target_noise=True)),
    (
        "level2 nfs=1",
        True,
        dict(specify_target_noise=True, noise_final_samples=1),
    ),
]:
    b = BADS(
        mk(0, het),
        np.full(D, 1.0),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options=dict(
            display="off",
            random_seed=0,
            max_fun_evals=150,
            uncertainty_handling=True,
            **extra,
        ),
    )
    r = b.optimize()
    yv = r["yval_vec"]
    ys = r["ysd_vec"]
    hist_f = b.iteration_history.get("fval").astype(float)
    print(
        f"{label}: func_count {r['func_count']} iterations {r['iterations']} fval {r['fval']:.4f} fsd {r['fsd']:.4f} "
        f"yval_vec shape {None if yv is None else np.shape(yv)} ysd_vec {None if ys is None else np.shape(ys)}; "
        f"x {np.round(r['x'],3)}"
    )
    if yv is not None and np.size(yv) > 1 and ys is not None:
        w = 1 / np.asarray(ys) ** 2
        print(
            "   check precision-weighted mean:",
            np.sum(np.ravel(yv) * w) / np.sum(w),
            1 / np.sqrt(np.sum(w)),
        )
    elif yv is not None and np.size(yv) > 1:
        v = np.ravel(yv)
        print("   check mean/SEM:", v.mean(), v.std(ddof=1) / np.sqrt(v.size))
