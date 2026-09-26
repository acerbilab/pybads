"""The starting points of a refit: the port's (gpyreg design of init_N
points plus the given rows, best opts_N optimized) against MATLAB's policy
(one local optimization from the previous hyperparameters, a second one
from the second-fit point), on the same data, at every refit of default
runs."""
import copy
import sys

import gpyreg
import gpyreg as gpr
import gpyreg.gaussian_process as gpmod
import numpy as np
import s4_runs as R

import pybads
import pybads.bads.gaussian_process_train as gpt
from pybads.bads.bads import BADS

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)

orig_rf = gpt._robust_gp_fit_
orig_fmf = gpmod.f_min_fill
recs = []
cur = {}


def fmf(f, x0, LB, UB, PLB, PUB, hprior, N, design=None, rng=None):
    X, y = orig_fmf(f, x0, LB, UB, PLB, PUB, hprior, N, design, rng=rng)
    if cur.get("active"):
        x0c = np.minimum(np.maximum(x0, LB), UB)
        cur["N"] = max(N, x0.shape[0])
        cur["rows"] = x0.shape[0]
        cur["best_is_old"] = bool(np.allclose(X[0], x0c[0]))
        cur["old_rank"] = int(
            np.flatnonzero(np.all(np.isclose(X, x0c[0]), axis=1))[0]
        )
    return X, y


def rf(gp, X, Y, s2, hyp_gp, gp_train, optim_state, options, rng=None):
    # MATLAB-like policy on a copy: no design, optimize from the given rows
    g = copy.deepcopy(gp)
    tr = dict(gp_train)
    tr["init_N"] = 0
    tr["opts_N"] = hyp_gp.shape[0]
    try:
        h_m, res_m, _ = g.fit(
            X,
            Y,
            s2,
            hyp0=hyp_gp.copy(),
            options=tr,
            rng=np.random.default_rng(0),
        )
        nll_m = res_m.fun
    except Exception as e:
        print("matlab-policy fit raised", type(e).__name__, e)
        h_m, nll_m = None, np.nan
    cur.clear()
    cur["active"] = True
    out = orig_rf(gp, X, Y, s2, hyp_gp, gp_train, optim_state, options, rng)
    cur["active"] = False
    g2 = copy.deepcopy(gp)
    g2.X, g2.y, g2.s2 = X, Y, s2
    obj = lambda h: float(
        np.ravel(g2._GP__gp_obj_fun(np.ravel(h), False, False))[0]
    )
    nll_p = obj(np.atleast_2d(out[1])[0])
    nll_m = obj(h_m[0]) if h_m is not None else np.nan
    D = X.shape[1]
    dl = (
        np.nan
        if h_m is None
        else float(np.max(np.abs(np.atleast_2d(out[1])[0, :D] - h_m[0, :D])))
    )
    recs.append(
        dict(
            N=cur.get("N"),
            rows=cur.get("rows"),
            best_is_old=cur.get("best_is_old"),
            old_rank=cur.get("old_rank"),
            opts_N=gp_train["opts_N"],
            nll_port=nll_p,
            nll_matlab_policy=nll_m,
            max_dloglen=dl,
        )
    )
    return out


gpt._robust_gp_fit_ = rf
gpmod.f_min_fill = fmf
D = 3
for name, f, x0 in (("rosen3", R.rosen, -1.5), ("ellip3", R.ellip, 2.0)):
    for seed in range(2 if len(sys.argv) < 2 else int(sys.argv[1])):
        recs.clear()
        b = BADS(
            f,
            np.full((1, D), x0),
            np.full((1, D), -5.0),
            np.full((1, D), 5.0),
            np.full((1, D), -3.0),
            np.full((1, D), 3.0),
            options={
                "display": "off",
                "random_seed": seed,
                "max_fun_evals": 200,
            },
        )
        b.optimize()
        best_old = sum(r["best_is_old"] for r in recs)
        worse = sum(
            r["nll_port"] > r["nll_matlab_policy"] + 1e-6 for r in recs
        )
        better = sum(
            r["nll_port"] < r["nll_matlab_policy"] - 1e-6 for r in recs
        )
        big = sum(r["max_dloglen"] > 0.1 for r in recs)
        print(
            f"{name} seed={seed}: refits={len(recs)}, design sizes={[r['N'] for r in recs]}, "
            f"second fits={sum(r['opts_N'] == 2 for r in recs)}"
        )
        print(
            f"   previous hyp was the best design point (so the optimization starts there, as in MATLAB): {best_old}/{len(recs)}; "
            f"its ranks: {[r['old_rank'] for r in recs]}"
        )
        print(
            f"   final nll: port lower in {better}, MATLAB-policy lower in {worse}, "
            f"|d log lengthscale| > 0.1 in {big}; nll diffs "
            f"{np.round([r['nll_port'] - r['nll_matlab_policy'] for r in recs], 3).tolist()}"
        )
