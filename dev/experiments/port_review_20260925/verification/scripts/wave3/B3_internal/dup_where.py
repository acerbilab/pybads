import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.bads.bads as bm
import pybads.function_logger.constraints_check as ccm
import pybads.search.es_search as es
from pybads import BADS

orig = ccm.contraints_check
log = []


def make(site):
    def wrapped(U, lb, ub, tol_mesh, fl, proj=True, nbc=None):
        out = orig(U, lb, ub, tol_mesh, fl, proj, nbc)
        tol = tol_mesh / 2
        u1 = np.round(out / tol)
        u2 = np.round(fl.X[: fl.X_max_idx + 1] / tol)
        m = (
            (u1[:, None, :] == u2[None, :, :]).all(-1).any(1)
            if len(out)
            else np.zeros(0, bool)
        )
        if m.any():
            j = [
                int(np.flatnonzero((u1[i] == u2).all(-1))[0])
                for i in np.flatnonzero(m)
            ]
            log.append((site, fl.func_count, int(m.sum()), len(out), j[:3]))
        return out

    return wrapped


bm.contraints_check = make("step/poll")
es.contraints_check = make("es")
f = lambda x: float(np.sum(x**2))
D = 3
for seed in [1, 3]:
    log.clear()
    b = BADS(
        f,
        np.full(D, 1.3),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options={"random_seed": seed, "max_fun_evals": 150, "display": "off"},
    )
    r = b.optimize()
    print(
        "seed",
        seed,
        "n",
        r["func_count"],
        "calls with evaluated points kept:",
        log[:10],
    )
