"""F5 (comparison): refit from gpyreg's design (as is) vs from the previous hyperparameters only (MATLAB-like)."""
import copy
import logging

import common  # noqa
import numpy as np

import pybads.bads.bads as bb
from pybads import BADS
from pybads.bads import gaussian_process_train as gpt

logging.getLogger("BADS").setLevel(logging.ERROR)
orig_rf = gpt._robust_gp_fit_
ROWS = []


def rf(gp, X, Y, s2, hyp, gp_train, os_, o, rng=None):
    out = orig_rf(gp, X, Y, s2, hyp, gp_train, os_, o, rng)
    g = copy.deepcopy(gp)
    r = {}
    for mode, opts in (
        ("design", dict(gp_train)),
        (
            "matlab-like",
            dict(gp_train, init_N=0, opts_N=np.shape(np.atleast_2d(hyp))[0]),
        ),
    ):
        gg = copy.deepcopy(g)
        try:
            h, res, _ = gg.fit(
                X,
                Y,
                s2,
                hyp0=np.atleast_2d(hyp),
                options=opts,
                rng=np.random.default_rng(0),
            )
            r[mode] = res.fun
        except np.linalg.LinAlgError:
            r[mode] = np.nan
    ROWS.append((gp_train["init_N"], r["design"], r["matlab-like"]))
    return out


gpt._robust_gp_fit_ = rf


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


for name, fun, D, seed in [("rosen3", rosen, 3, 80), ("ell3", ell, 3, 80)]:
    ROWS.clear()
    BADS(
        fun,
        0.5 * np.ones((1, D)) + 0.1 * np.arange(D),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options={"random_seed": seed, "display": "off", "max_fun_evals": 150},
    ).optimize()
    a = np.array(ROWS)
    d = a[:, 2] - a[:, 1]
    print(f"{name}: {len(a)} refits; init_N {a[:,0].astype(int).tolist()}")
    print(f"   nlZ(matlab-like) - nlZ(design): {np.round(d, 2).tolist()}")
    print(
        f"   design better (>0.01) in {int(np.nansum(d > 0.01))}, equal in {int(np.nansum(abs(d) <= 0.01))}, "
        f"matlab-like better in {int(np.nansum(d < -0.01))}, matlab-like LinAlgError {int(np.sum(np.isnan(a[:,2])))}, design LinAlgError {int(np.sum(np.isnan(a[:,1])))}"
    )
