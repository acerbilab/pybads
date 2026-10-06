"""F1 / B3-K3 consequence: repeated evaluations in seeded runs, by stage;
optionally with a MATLAB-like uCheck (evaluated bins removed)."""

import numpy as np
import vhdr  # noqa

import pybads.bads.bads as bads_mod
import pybads.search.es_search as es_mod
from pybads import BADS
from pybads.function_logger import FunctionLogger
from pybads.function_logger import constraints_check as cc_mod

orig_cc = cc_mod.contraints_check


def cc_matlab(
    U, lb, ub, tol_mesh, function_logger, proj=True, non_box_cons=None
):
    U = np.atleast_2d(U)
    if proj:
        U_new = np.maximum(np.minimum(U, ub), lb)
    else:
        idx = np.any(U > ub, axis=1) | np.any(U < lb, axis=1)
        U_new = U[~idx].copy()
    _, first = np.unique(U_new, axis=0, return_index=True)
    U_new = U_new[np.sort(first), :]
    if U_new.size > 0:
        tol = tol_mesh / 2.0
        u1 = np.round(U_new / tol)
        u2 = np.round(function_logger.X[: function_logger.X_max_idx + 1] / tol)
        tmp = np.vstack((u1, u2))
        _, first = np.unique(tmp, axis=0, return_index=True)
        keep = first[first < len(u1)]
        ev = {tuple(r) for r in u2}
        keep = np.array([k for k in keep if tuple(u1[k]) not in ev], dtype=int)
        U_new = U_new[keep]
    if non_box_cons is not None:
        X = function_logger.variable_transformer.inverse_transf(U_new)
        U_new = U_new[np.ravel(non_box_cons(X)) <= 0]
    return U_new


stage = {"s": "init"}
log = []
orig_call = FunctionLogger.__call__
orig_search = BADS._search_step_
orig_poll = BADS._poll_step_


def fl_call(self, x, record_duplicate_data=True):
    log.append(
        (
            stage["s"],
            np.array(x, dtype=float).ravel().copy(),
            record_duplicate_data,
        )
    )
    return orig_call(self, x, record_duplicate_data)


def search(self, gp):
    stage["s"] = "search"
    try:
        return orig_search(self, gp)
    finally:
        stage["s"] = "other"


def poll(self, gp):
    stage["s"] = "poll"
    try:
        return orig_poll(self, gp)
    finally:
        stage["s"] = "other"


FunctionLogger.__call__ = fl_call
BADS._search_step_ = search
BADS._poll_step_ = poll


def run(name, fun, x0, lb, ub, plb, pub, opts, matlab_cc=False):
    log.clear()
    stage["s"] = "init"
    if matlab_cc:
        es_mod.contraints_check = cc_matlab
        bads_mod.contraints_check = cc_matlab
    else:
        es_mod.contraints_check = orig_cc
        bads_mod.contraints_check = orig_cc
    o = {"display": "off"}
    o.update(opts)
    res = BADS(fun, x0, lb, ub, plb, pub, options=o).optimize()
    tol = res["x"] is not None and 2.0**-20
    seen = set()
    rep = {"search": 0, "poll": 0, "init": 0, "other": 0}
    n = {"search": 0, "poll": 0, "init": 0, "other": 0}
    for st, u, rec in log:
        if not rec:
            continue  # final re-evaluation samples, not recorded
        k = tuple(np.round(u / tol))
        n[st] += 1
        if k in seen:
            rep[st] += 1
        seen.add(k)
    print(
        f"{name:34s} cc={'MATLAB' if matlab_cc else 'PyBADS':6s} fval={res['fval']:.6g} "
        f"evals={res['func_count']} repeats: search {rep['search']}/{n['search']}, poll {rep['poll']}/{n['poll']}, init {rep['init']}/{n['init']}"
    )
    return res


def shifted_sphere(x):
    x = np.atleast_2d(x)
    return float(np.sum((x + 1) ** 2))


def sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


for D in (1, 2):
    for seed in (0, 1):
        for mcc in (False, True):
            run(
                f"bound-opt D={D} seed={seed}",
                shifted_sphere,
                np.full(D, 2.5),
                np.zeros(D),
                np.full(D, 5.0),
                np.full(D, 0.1),
                np.full(D, 4.9),
                {"random_seed": seed, "max_fun_evals": 60},
                mcc,
            )

for seed in (0, 1):
    run(
        f"sphere D=1 seed={seed}",
        sphere,
        np.full(1, 1.5),
        np.full(1, -5.0),
        np.full(1, 5.0),
        np.full(1, -2.0),
        np.full(1, 2.0),
        {"random_seed": seed, "max_fun_evals": 30},
    )
    run(
        f"sphere D=3 seed={seed}",
        sphere,
        np.full(3, 1.5),
        np.full(3, -5.0),
        np.full(3, 5.0),
        np.full(3, -2.0),
        np.full(3, 2.0),
        {"random_seed": seed, "max_fun_evals": 100},
    )


def hetero_ellipsoid(seed):
    rng = np.random.default_rng(seed)
    w = np.array([1.0, 10.0, 100.0])

    def f(x):
        x = np.atleast_2d(x)
        fx = float(np.sum(w * x**2))
        sd = 1.0 + 0.1 * fx**0.5
        return fx + sd * rng.standard_normal(), sd

    return f


for seed in (0,):
    for mcc in (False, True):
        run(
            f"hetero ellipsoid D=3 L2 seed={seed}",
            hetero_ellipsoid(seed),
            np.full(3, 2.0),
            np.full(3, -10.0),
            np.full(3, 10.0),
            np.full(3, -3.0),
            np.full(3, 3.0),
            {
                "random_seed": seed,
                "max_fun_evals": 200,
                "specify_target_noise": True,
            },
            mcc,
        )
