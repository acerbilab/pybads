"""Array lengths after FunctionLogger.finalize and reset_fun_eval_time
(the changelog's "FunctionLogger.finalize" entry)."""
import gpyreg
import numpy as np

import pybads
from pybads.function_logger import FunctionLogger

print(pybads.__file__, gpyreg.__file__, flush=True)


def lengths(fl):
    names = [
        "X_orig",
        "Y_orig",
        "X",
        "Y",
        "X_flag",
        "fun_eval_time",
        "n_evals",
    ]
    return {n: getattr(fl, n).shape[0] for n in names}


for n_points, label in (
    (5, "5 points, cache 500"),
    (8, "8 points, cache 4 (grown)"),
):
    fl = FunctionLogger(
        lambda x: float(np.sum(x**2)),
        2,
        False,
        0,
        cache_size=500 if n_points == 5 else 4,
    )
    for i in range(n_points):
        fl(np.array([0.1 * i, 0.2]))
    print(label, "before:", lengths(fl))
    fl.finalize()
    print(label, "finalize:", lengths(fl))
    fl.reset_fun_eval_time()
    print(label, "reset:", lengths(fl))
    try:
        print(label, "n_evals[X_flag]:", fl.n_evals[fl.X_flag].ravel())
    except Exception as e:
        print(label, "n_evals[X_flag]:", type(e).__name__, e)
