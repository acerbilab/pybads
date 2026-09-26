"""The p_less of the poll: np.sort on the (N, 1) array of the probabilities
of improvement sorts along the axis of length 1, so the product takes the
last D+1 points in the order of u_poll, not the D+1 largest. Measured in
default runs: the shape, and the stop decisions after a good poll under
the coded and the intended product."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from scipy.special import erfc

import pybads.bads.bads as bm
from pybads import BADS

D_ = 3
f_pi = np.array([[0.9], [0.01], [0.02], [0.03], [0.04]])
print(
    "toy: np.sort(f_pi)[::-1] ->",
    np.ravel(np.sort(f_pi)[::-1]),
    " coded p_less %.4f, with the D+1 largest %.4f"
    % (
        np.prod(1 - np.sort(f_pi)[::-1][0 : D_ + 1]),
        np.prod(1 - np.sort(f_pi.ravel())[::-1][0 : D_ + 1]),
    ),
)

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
        b = ctx["b"]
        D = b.D
        stats.setdefault("shape", set()).add(f_mu.shape[1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (
                b.optim_state["f_target"] - b.sufficient_improvement - f_mu
            ) / fs
        if np.all(np.isfinite(g)) and len(f_mu) > D + 1:
            p = 0.5 * erfc(-g / np.sqrt(2))
            coded = np.prod(1 - np.sort(p)[::-1][0 : min(D + 1, len(p))])
            intended = np.prod(
                1 - np.sort(p.ravel())[::-1][0 : min(D + 1, len(p))]
            )
            stats["n_more_than_D+1"] = stats.get("n_more_than_D+1", 0) + 1
            if not np.isclose(coded, intended, rtol=0, atol=1e-15):
                stats["differ"] = stats.get("differ", 0) + 1
            thr = 1 - b.options["tol_poi"]
            if (coded > thr) != (intended > thr):
                stats["decision_differs"] = (
                    stats.get("decision_differs", 0) + 1
                )
    return z, f_mu, fs


bm.acq_fcn_lcb = lcb


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ellip(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (3 * np.arange(D) / (D - 1)) * x) ** 2))


for name, f, D in [
    ("rosen", rosen, 3),
    ("ellip", ellip, 4),
    ("ellip", ellip, 6),
]:
    for seed in [1, 2]:
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
        print(name, D, seed, stats, "fval %.3g" % r["fval"])
