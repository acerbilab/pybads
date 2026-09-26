import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.bads.bads as bm
import pybads.function_logger.constraints_check as ccm
import pybads.search.es_search as es
from pybads import BADS

orig = ccm.contraints_check


def fixed(U, lb, ub, tol_mesh, fl, proj=True, non_box_cons=None):
    # same as the port, but a candidate matching an evaluated point is removed
    if proj:
        U_new = np.maximum(np.minimum(U, ub), lb)
    else:
        idx = np.any(U > ub, axis=1) | np.any(U < lb, axis=1)
        U_new = U[~idx].copy()
    _, i = np.unique(U_new, axis=0, return_index=True)
    U_new = U_new[np.sort(i), :]
    if U_new.size > 0:
        tol = tol_mesh / 2.0
        u1 = np.round(U_new / tol)
        u2 = np.round(fl.X[: fl.X_max_idx + 1] / tol)
        keep = (
            ~(u1[:, None, :] == u2[None, :, :]).all(-1).any(1)
            if len(u2)
            else np.ones(len(u1), bool)
        )
        U_new = U_new[keep]
    if non_box_cons is not None:
        X = fl.variable_transformer.inverse_transf(U_new)
        U_new = U_new[np.ravel(non_box_cons(X)) <= 0]
    return U_new


def run(fun, D, lb, ub, plb, pub, x0, seed, patched):
    f = fixed if patched else orig
    bm.contraints_check = f
    es.contraints_check = f
    b = BADS(
        fun,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"random_seed": seed, "max_fun_evals": 150, "display": "off"},
    )
    r = b.optimize()
    fl = b.function_logger
    X = fl.X_orig[: fl.Xn + 1]
    _, cnt = np.unique(X, axis=0, return_counts=True)
    ndup = int(np.sum(cnt - 1))
    return r["fval"], r["func_count"], ndup


for name, D in [("corner2", 2), ("corner1", 1), ("sphere3", 3)]:
    if name.startswith("corner"):
        fun = lambda x: float(np.sum((x + 1.0) ** 2))
        lb, ub = np.zeros(D), np.full(D, 5.0)
        plb, pub = np.full(D, 0.5), np.full(D, 4.0)
        x0 = np.full(D, 2.0)
    else:
        fun = lambda x: float(np.sum(x**2))
        lb, ub = np.full(D, -5.0), np.full(D, 5.0)
        plb, pub = np.full(D, -2.0), np.full(D, 2.0)
        x0 = np.full(D, 1.3)
    for seed in [1, 2, 3]:
        a = run(fun, D, lb, ub, plb, pub, x0, seed, False)
        b = run(fun, D, lb, ub, plb, pub, x0, seed, True)
        print(
            name,
            seed,
            "port: fval=%.3g n=%d dup=%d" % a,
            "| removal: fval=%.3g n=%d dup=%d" % b,
        )
