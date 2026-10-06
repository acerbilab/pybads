import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)


def repeats(fl, tol_mesh):
    X = fl.X[: fl.Xn + 1]
    bins = np.round(X / (tol_mesh / 2))
    _, first = np.unique(bins, axis=0, return_index=True)
    return X.shape[0] - first.size


for D in (1, 2):
    for seed in (0, 1):
        b = BADS(
            lambda x: float(np.sum((np.atleast_2d(x) + 1) ** 2)),
            np.full(D, 2.5),
            np.zeros(D),
            np.full(D, 5.0),
            np.full(D, 0.5),
            np.full(D, 4.5),
            options={"display": "off", "random_seed": seed},
        )
        r = b.optimize()
        print(
            f"bound optimum D={D} seed={seed}: evals {r['func_count']} fval {r['fval']:.6g} repeats {repeats(b.function_logger, b.optim_state['tol_mesh'])}"
        )
