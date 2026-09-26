import inspect

import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads import BADS
from pybads.function_logger.function_logger import FunctionLogger

orig = FunctionLogger.__call__
stats = {}


def wrapped(self, x, record_duplicate_data=True):
    x2 = np.atleast_2d(x)
    caller = inspect.stack()[1].function
    tol = TOL[0] / 2
    n = self.X_max_idx + 1
    dup = False
    if n > 0:
        dup = np.any(
            np.all(np.round(self.X[:n] / tol) == np.round(x2 / tol), axis=1)
        )
    s = stats.setdefault(caller, [0, 0])
    s[0] += 1
    s[1] += int(dup)
    return orig(self, x, record_duplicate_data)


FunctionLogger.__call__ = wrapped


def rosen(x):
    x = np.atleast_2d(x)
    return np.sum(
        100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2
    )


def ellip(x):
    x = np.ravel(x)
    D = x.size
    return np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2)


def sphere(x):
    return np.sum(np.ravel(x) ** 2)


TOL = [None]
for name, f, D in [
    ("sphere", sphere, 4),
    ("rosen", rosen, 2),
    ("rosen", rosen, 4),
    ("ellip", ellip, 4),
]:
    for seed in range(3):
        stats.clear()
        lb = np.full(D, -20.0)
        ub = np.full(D, 20.0)
        plb = np.full(D, -5.0)
        pub = np.full(D, 5.0)
        x0 = np.full(D, 3.0)
        b = BADS(
            f,
            x0,
            lb,
            ub,
            plb,
            pub,
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        )
        TOL[0] = b.optim_state["tol_mesh"]
        r = b.optimize()
        print(
            f"{name} D={D} seed={seed} fval={r['fval']:.3g} evals={r['func_count']} ",
            {k: tuple(v) for k, v in stats.items()},
        )
