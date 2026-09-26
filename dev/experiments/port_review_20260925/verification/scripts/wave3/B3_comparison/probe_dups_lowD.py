import inspect
import sys

import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.function_logger.constraints_check as ccm
import pybads.search.es_search as es
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

# count evaluated points surviving contraints_check inside ESSearch
orig_cc = ccm.contraints_check
cstats = [0, 0]


def cc_wrapped(U, lb, ub, tol_mesh, fl, proj=True, nbc=None):
    out = orig_cc(U, lb, ub, tol_mesh, fl, proj, nbc)
    tol = tol_mesh / 2
    n = fl.X_max_idx + 1
    u2 = {tuple(r) for r in np.round(fl.X[:n] / tol)}
    k = sum(tuple(r) in u2 for r in np.round(np.atleast_2d(out) / tol))
    cstats[0] += len(np.atleast_2d(out))
    cstats[1] += k
    return out


es.contraints_check = cc_wrapped


def sphere(x):
    return np.sum(np.ravel(x) ** 2)


TOL = [None]
import sys

DIM = int(sys.argv[1])
for level in ["det"]:
    for seed in range(4):
        stats.clear()
        cstats[:] = [0, 0]
        D = DIM
        rng = np.random.default_rng(100 + seed)
        if level == "noisy":
            f = lambda x: sphere(x) + rng.normal()
            opts = {
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
                "uncertainty_handling": True,
                "noise_size": 1.0,
            }
        else:
            f = sphere
            opts = {
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            }
        lb = np.full(D, -20.0)
        ub = np.full(D, 20.0)
        plb = np.full(D, -5.0)
        pub = np.full(D, 5.0)
        b = BADS(f, np.full(D, 3.0), lb, ub, plb, pub, options=opts)
        TOL[0] = b.optim_state["tol_mesh"]
        r = b.optimize()
        print(
            f"{level} seed={seed} fval={r['fval']:.3g} evals={r['func_count']} ",
            {k: tuple(v) for k, v in stats.items()},
            "ES candidates kept / of which evaluated:",
            cstats,
        )
