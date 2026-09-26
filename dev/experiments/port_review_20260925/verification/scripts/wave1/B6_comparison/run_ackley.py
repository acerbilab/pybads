"""Run BADS on a 6-D Ackley function (200 evals) and inspect the GP mean prior/bounds."""
import sys
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

D = 6
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def ackley(x):
    x = np.asarray(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        - np.exp(np.mean(np.cos(2 * np.pi * x)))
        + 20
        + np.e
    )


x0 = np.full(D, 3.3)
lb = np.full(D, -32.0)
ub = np.full(D, 32.0)
plb = np.full(D, -5.0)
pub = np.full(D, 5.0)
# count NaN/inf log priors seen by gpyreg during fits
import gpyreg.gaussian_process as gpm

orig = gpm.GP._GP__compute_log_priors
stats = {"calls": 0, "nan": 0, "inf": 0}


def wrapped(self, hyp, compute_grad):
    out = orig(self, hyp, compute_grad)
    lp = out[0] if compute_grad else out
    stats["calls"] += 1
    if np.isnan(lp):
        stats["nan"] += 1
    elif np.isinf(lp):
        stats["inf"] += 1
    return out


gpm.GP._GP__compute_log_priors = wrapped
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bads = BADS(
        ackley,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"random_seed": seed, "max_fun_evals": 200, "display": "off"},
    )
    res = bads.optimize()
print(
    "fval",
    res["fval"],
    "x",
    np.round(res["x"], 4),
    "func_count",
    res["func_count"],
)
print("log-prior evaluations:", stats)
for it, gp in enumerate(bads.iteration_history["gp"]):
    pri = gp.get_priors()["mean_const"]
    b = gp.get_bounds()["mean_const"]
    hyp = gp.get_hyperparameters()[0]
    print(
        f"it {it:2d} ymin {np.min(gp.y):.3g} prior mu {pri[1][0][0]:.3g} sd {pri[1][1][0]:.3g} "
        f"bounds [{b[0][0]:.3g},{b[1][0]:.3g}] mean_hyp {hyp['mean_const'][0]:.4g} "
        f"norm_const(mean) {gp.normalization_constants[-1]:.3g}"
    )
