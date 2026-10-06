"""How often, in default runs, a poll step meets a remaining point with a
zero predictive SD (gamma_z not finite: p_less = 0 and the GP declared
unreliable), and a stop after a good poll that follows from it."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bm
from pybads import BADS

ctx = {"poll": False}
orig_poll = bm.BADS._poll_step_


def poll(self, gp):
    ctx["poll"] = True
    ctx["b"] = self
    try:
        return orig_poll(self, gp)
    finally:
        ctx["poll"] = False


bm.BADS._poll_step_ = poll
stats = {}
orig_lcb = bm.acq_fcn_lcb


def lcb(xi, fc, gp, sqrt_beta=None):
    z, f_mu, fs = orig_lcb(xi, fc, gp, sqrt_beta)
    if ctx["poll"]:
        stats["steps"] = stats.get("steps", 0) + 1
        if np.any(fs == 0):
            stats["zero_sd"] = stats.get("zero_sd", 0) + 1
        if np.any(~np.isfinite(z)):
            stats["nonfinite_lcb"] = stats.get("nonfinite_lcb", 0) + 1
    return z, f_mu, fs


bm.acq_fcn_lcb = lcb
orig_stop = bm.BADS._is_poll_stop_


def stop(self, good, cal, p_less, count):
    r = orig_stop(self, good, cal, p_less, count)
    if good:
        k = (
            "good_stop_unreliable"
            if (r and cal)
            else ("good_stop_poi" if r else "good_continue")
        )
        stats[k] = stats.get(k, 0) + 1
    return r


bm.BADS._is_poll_stop_ = stop


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ellip(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (3 * np.arange(D) / (D - 1)) * x) ** 2))


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


for name, f, D in [
    ("rosen", rosen, 3),
    ("ellip", ellip, 4),
    ("sphere", sphere, 3),
]:
    for seed in [1, 2, 3]:
        stats.clear()
        lb = -5 * np.ones((1, D))
        ub = 5 * np.ones((1, D))
        plb = -2 * np.ones((1, D))
        pub = 2 * np.ones((1, D))
        r = BADS(
            f,
            np.full((1, D), 1.5),
            lb,
            ub,
            plb,
            pub,
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        ).optimize()
        print(name, seed, dict(sorted(stats.items())), "fval %.3g" % r["fval"])
