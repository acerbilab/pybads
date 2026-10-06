"""Ratio of predictive to latent SD at the points whose stats are saved."""
import logging

import common  # noqa
import numpy as np

import pybads.bads.bads as bb
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)

orig_acq = bb.acq_fcn_lcb
LAST = {}


def acq(xi, fc, gp, sqrt_beta=None):
    z, fmu, fs = orig_acq(xi, fc, gp, sqrt_beta)
    _, ys2 = gp.predict(xi, add_noise=True)
    LAST["fs"] = np.ravel(fs).copy()
    LAST["ys"] = np.sqrt(np.ravel(ys2)).copy()
    return z, fmu, fs


bb.acq_fcn_lcb = acq
ALL = []


class B(BADS):
    def _save_gp_stats_(self, fval, ymu, ys):
        idx = np.flatnonzero(LAST["fs"] == ys)
        ALL.append((ys, LAST["ys"][idx[0]], self.function_logger.func_count))
        return super()._save_gp_stats_(fval, ymu, ys)


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


for name, fun, D, seed, noisy in [
    ("rosen3", rosen, 3, 50, False),
    ("ell3", ell, 3, 50, False),
    ("noisy sphere3", None, 3, 50, True),
]:
    ALL.clear()
    opts = {"random_seed": seed, "display": "off", "max_fun_evals": 200}
    if noisy:
        r_ = np.random.default_rng(seed + 7)
        fun = lambda x, r_=r_: float(
            np.sum(np.ravel(x) ** 2) + r_.standard_normal()
        )
        opts["uncertainty_handling"] = True
    B(
        fun,
        0.5 * np.ones((1, D)) + 0.1 * np.arange(D),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options=opts,
    ).optimize()
    a = np.array(ALL)
    lat, pr = a[:, 0], a[:, 1]
    with np.errstate(divide="ignore"):
        ratio = pr / lat
    q = np.nanpercentile(ratio[np.isfinite(ratio)], [10, 50, 90])
    print(
        f"{name}: {len(a)} stats; predictive/latent SD ratio 10/50/90%: {np.round(q, 2)}; latent SD <= 1e-8 in {int(np.sum(lat <= 1e-8))}; "
        f"median ratio in the last 50 evals {np.median(ratio[-50:][np.isfinite(ratio[-50:])]):.3g}"
    )
