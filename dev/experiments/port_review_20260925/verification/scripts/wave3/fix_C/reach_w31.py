"""Does the fingerprint reach W3-1? And the bound-optimum runs' repeats."""
import numpy as np

import pybads
import pybads.bads.bads as bads_module
import pybads.search.es_search as es_module
from pybads import BADS
from pybads.function_logger.constraints_check import contraints_check as new_cc

print(pybads.__file__)


def old_cc(U, lb, ub, tol_mesh, fl, proj=True, non_box_cons=None):
    U_new = (
        np.maximum(np.minimum(U, ub), lb)
        if proj
        else U[~(np.any(U > ub, axis=1) | np.any(U < lb, axis=1))].copy()
    )
    _, i = np.unique(U_new, axis=0, return_index=True)
    U_new = U_new[np.sort(i), :]
    if U_new.size > 0:
        tol = tol_mesh / 2.0
        u1 = np.round(U_new / tol)
        u2 = np.round(fl.X[: fl.X_max_idx + 1] / tol)
        _, i = np.unique(np.vstack((u1, u2)), axis=0, return_index=True)
        U_new = U_new[i[i < len(u1)]]
    return U_new


stats = {}


def wrap(where):
    def cc(U, lb, ub, tol_mesh, fl, proj=True, non_box_cons=None):
        a = new_cc(U, lb, ub, tol_mesh, fl, proj, non_box_cons)
        b = old_cc(U, lb, ub, tol_mesh, fl, proj, non_box_cons)
        s = stats.setdefault(where, [0, 0])
        s[0] += 1
        s[1] += not (a.shape == b.shape and np.array_equal(a, b))
        return a

    return cc


bads_module.contraints_check = wrap("bads.py (init, search step, poll)")
es_module.contraints_check = wrap("es_search")


def f(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


g = np.random.default_rng(0)


def fn(x):
    return float(np.sum(np.atleast_2d(x) ** 2) + g.standard_normal())


for fun, noisy in [(f, False), (fn, True)]:
    for seed in range(3):
        o = {"display": "off", "max_fun_evals": 80, "random_seed": seed}
        if noisy:
            o["uncertainty_handling"] = True
        BADS(
            fun,
            np.ones(3) * 4,
            -100 * np.ones(3),
            100 * np.ones(3),
            -8 * np.ones(3),
            12 * np.ones(3),
            options=o,
        ).optimize()
print("fingerprint runs: calls, calls where old != new:", stats)


def repeats(fl, tol_mesh):
    X = fl.X[: fl.Xn + 1]
    bins = np.round(X / (tol_mesh / 2))
    _, first = np.unique(bins, axis=0, return_index=True)
    return X.shape[0] - first.size


stats.clear()
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
print("bound-optimum runs: calls, calls where old != new:", stats)
