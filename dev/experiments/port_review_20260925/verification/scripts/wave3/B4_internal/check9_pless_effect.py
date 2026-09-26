"""Stop decisions after a good poll under the coded p_less and under the
product over the D+1 largest PoI; and the runs with the intended product
substituted (monkeypatched _is_poll_stop_), default options otherwise."""

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from scipy.special import erfc

import pybads.bads.bads as bm
from pybads import BADS

MODE = {"fix": False}
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
    ctx["intended"] = None
    if ctx["poll"]:
        b = ctx["b"]
        D = b.D
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (
                b.optim_state["f_target"] - b.sufficient_improvement - f_mu
            ) / fs
        if np.all(np.isfinite(g)):
            p = 0.5 * erfc(-g / np.sqrt(2))
            ctx["intended"] = np.prod(
                1 - np.sort(p.ravel())[::-1][0 : min(D + 1, len(p))]
            )
    return z, f_mu, fs


bm.acq_fcn_lcb = lcb
orig_stop = bm.BADS._is_poll_stop_


def stop(self, good, cal, p_less, count):
    pi = ctx.get("intended")
    if good and pi is not None:
        thr = 1 - self.options["tol_poi"]
        stats["good_steps"] = stats.get("good_steps", 0) + 1
        if (p_less > thr) != (pi > thr):
            k = (
                "coded_stops_intended_continues"
                if p_less > thr
                else "coded_continues_intended_stops"
            )
            stats[k] = stats.get(k, 0) + 1
    if MODE["fix"] and pi is not None:
        p_less = pi
    return orig_stop(self, good, cal, p_less, count)


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


def ackley(x):
    x = np.ravel(x)
    D = x.size
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / D))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / D)
        + 20
        + np.e
    )


for name, f, D in [
    ("rosen", rosen, 3),
    ("ellip", ellip, 4),
    ("ellip", ellip, 6),
    ("ackley", ackley, 4),
]:
    for seed in [1, 2, 3]:
        out = []
        for fix in [False, True]:
            MODE["fix"] = fix
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
            out.append(
                "%s: fval %.3g evals %d %s"
                % (
                    "intended" if fix else "coded",
                    r["fval"],
                    r["func_count"],
                    dict(stats),
                )
            )
        print(name, D, seed, " | ".join(out))
