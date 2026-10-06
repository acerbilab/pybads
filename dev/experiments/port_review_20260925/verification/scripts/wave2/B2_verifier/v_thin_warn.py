import traceback
import warnings

import common
import numpy as np

from pybads import BADS


def show(message, category, filename, lineno, file=None, line=None):
    st = [
        f for f in traceback.extract_stack() if "pybads-review" in f.filename
    ]
    print(
        "WARNING",
        category.__name__,
        message,
        "| from",
        [
            (f.filename.split("pybads-review/")[-1], f.lineno, f.name)
            for f in st[-3:]
        ],
    )


warnings.showwarning = show
f = lambda x: float(np.sum((np.ravel(x) - 0.3) ** 2))
nbc = lambda X: np.abs(np.atleast_2d(X)[:, 0] - np.atleast_2d(X)[:, 1]) - 0.005
b = BADS(
    f,
    np.full(2, 0.5),
    np.full(2, -5.0),
    np.full(2, 5.0),
    np.full(2, -2.0),
    np.full(2, 2.0),
    nbc,
    options=dict(display="off", random_seed=0, max_fun_evals=200),
)
r = b.optimize()
print(r["message"], r["func_count"], b.iteration_history.get("mesh_size"))
