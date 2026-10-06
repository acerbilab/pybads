"""C-F12: fun_values."""
import traceback

import common
import numpy as np

from pybads import BADS

f = lambda x: float(np.sum(np.ravel(x) ** 2))
for fv in [
    dict(X=np.array([[0.2, 0.1], [0.3, 0.3]]), Y=np.array([[0.05], [0.18]])),
    dict(X=np.array([[0.2, 0.1]]), Y=np.array([[0.05]])),
]:
    try:
        b = BADS(
            f,
            np.array([1.0, 1.0]),
            np.full(2, -5.0),
            np.full(2, 5.0),
            np.full(2, -2.0),
            np.full(2, 2.0),
            options=dict(
                random_seed=0, max_fun_evals=30, display="off", fun_values=fv
            ),
        )
        r = b.optimize()
        print("ran", r["fval"])
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(
            len(fv["X"]),
            "points ->",
            type(e).__name__,
            e,
            "| at",
            tb.filename.split("pybads-review/")[-1],
            tb.lineno,
        )
