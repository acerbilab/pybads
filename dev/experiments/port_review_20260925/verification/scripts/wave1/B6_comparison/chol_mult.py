"""How often gpyreg's training Cholesky needs the noise multiplied (sn2_mult > 1) in default runs,
where MATLAB (CholAttempts = 0) raises 'Cannot compute Cholesky decomposition.'"""
import sys
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import gpyreg.gaussian_process as gpm

from pybads import BADS

orig = gpm.GP.__dict__["_GP__training_cholesky"].__func__
stats = {"calls": 0, "mult>1": 0, "max_mult": 1, "failed": 0}


def wrapped(K, sn2, L_chol, sn2_mult=1):
    stats["calls"] += 1
    try:
        L, sl, m = orig(K, sn2, L_chol, sn2_mult)
    except Exception:
        stats["failed"] += 1
        raise
    if m > sn2_mult:
        stats["mult>1"] += 1
        stats["max_mult"] = max(stats["max_mult"], m)
    return L, sl, m


gpm.GP._GP__training_cholesky = staticmethod(wrapped)
name = sys.argv[1]
D = int(sys.argv[2])
seed = int(sys.argv[3])
if name == "sphere":
    fun = lambda x: float(np.sum(np.asarray(x) ** 2))
    x0 = np.full(D, 0.5)
    lb = np.full(D, -5.0)
    ub = np.full(D, 5.0)
    plb = np.full(D, -1.0)
    pub = np.full(D, 1.0)
elif name == "rosen":
    fun = lambda x: float(
        np.sum(
            100 * (np.asarray(x)[1:] - np.asarray(x)[:-1] ** 2) ** 2
            + (1 - np.asarray(x)[:-1]) ** 2
        )
    )
    x0 = np.zeros(D)
    lb = np.full(D, -5.0)
    ub = np.full(D, 5.0)
    plb = np.full(D, -2.0)
    pub = np.full(D, 2.0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bads = BADS(
        fun,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"random_seed": seed, "max_fun_evals": 200, "display": "off"},
    )
    res = bads.optimize()
print(name, D, seed, "fval", res["fval"], "cholesky:", stats)
mults = [gp.posteriors[0].sn2_mult for gp in bads.iteration_history["gp"]]
noise = [
    round(float(gp.get_hyperparameters()[0]["noise_log_scale"][0]), 2)
    for gp in bads.iteration_history["gp"]
]
print("stored GPs: posterior sn2_mult", mults, "log noise", noise)
