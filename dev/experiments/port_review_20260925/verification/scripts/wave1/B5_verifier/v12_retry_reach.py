"""B5-R1 / F11 / C-F10: how often local_gp_fitting's posterior update fails (exit flag -2) at default options."""
import logging

import common  # noqa
import numpy as np

import pybads.bads.bads as bb
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)
orig_lgf = bb.local_gp_fitting
C = {"calls": 0, "refit": 0, "m2": 0, "m2_refit": 0, "restored": 0}


def lgf(gp, u, fl, o, os_, ih, refit_flag, rng=None):
    g, flag = orig_lgf(gp, u, fl, o, os_, ih, refit_flag, rng=rng)
    C["calls"] += 1
    C["refit"] += bool(refit_flag)
    if flag == -2:
        C["m2"] += 1
        C["m2_refit"] += bool(refit_flag)
        C["restored"] += "needs_rebuild" in g.temporary_data
    return g, flag


bb.local_gp_fitting = lgf


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


for name, fun, D, seed in [
    ("rosen3", rosen, 3, 41),
    ("ell4", ell, 4, 40),
    ("ell4", ell, 4, 41),
    ("ell2", ell, 2, 40),
]:
    for k in C:
        C[k] = 0
    BADS(
        fun,
        0.5 * np.ones((1, D)) + 0.1 * np.arange(D),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options={"random_seed": seed, "display": "off", "max_fun_evals": 200},
    ).optimize()
    print(name, seed, dict(C))
