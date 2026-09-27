import inspect

import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads import BADS
from pybads.function_logger.function_logger import FunctionLogger

orig = FunctionLogger.__call__
B = [None]


def wrapped(self, x, record_duplicate_data=True):
    caller = inspect.stack()[1].function
    n = self.X_max_idx + 1
    x2 = np.atleast_2d(x)
    tol = B[0].optim_state["tol_mesh"] / 2
    hit = np.flatnonzero(
        np.all(np.round(self.X[:n] / tol) == np.round(x2 / tol), axis=1)
    )
    if caller == "_search_step_" and hit.size:
        st = B[0].optim_state
        print(
            f"  repeat at eval {self.func_count+1}: u={x2.ravel()}, same as row {hit.tolist()} (exact: {np.all(self.X[hit[0]] == x2)}), "
            f"incumbent={np.all(B[0].u_best == x2)}, mesh={st['mesh_size']:.3g}, search mesh={st['search_mesh_size']:.3g}, search_factor={st['search_factor']:.3g}"
        )
    return orig(self, x, record_duplicate_data)


FunctionLogger.__call__ = wrapped
for seed in range(2):
    b = BADS(
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        np.array([3.0]),
        np.array([-20.0]),
        np.array([20.0]),
        np.array([-5.0]),
        np.array([5.0]),
        options={"random_seed": seed, "display": "off", "max_fun_evals": 200},
    )
    B[0] = b
    r = b.optimize()
    print(
        f"seed={seed} fval={r['fval']:.3g} evals={r['func_count']} msg={r['message'][:60]}"
    )
