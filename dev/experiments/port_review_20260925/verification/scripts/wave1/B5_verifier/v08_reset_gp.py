"""F8 (internal) / F1 (comparison): reset_gp is never cleared by a rebuild."""
import logging

import common  # noqa
import numpy as np

import pybads.bads.bads as bb
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)

orig_lgf = bb.local_gp_fitting
CUR = {}


def lgf(gp, u, fl, options, os_, ih, refit_flag, rng=None):
    b = CUR["b"]
    only_reset = False
    if u is b.u:  # the search/poll rebuild of the incumbent's GP
        only_reset = (
            (not refit_flag)
            and b.reset_gp
            and not gp.temporary_data.get("needs_rebuild", False)
            and CUR["phase_first"] is False
        )
        CUR["n"] += 1
        CUR["only_reset"] += only_reset
        CUR["n_train_in"].append(gp.X.shape[0] if gp.X is not None else 0)
    out = orig_lgf(gp, u, fl, options, os_, ih, refit_flag, rng=rng)
    if u is b.u and CUR["mode"] == "clear":
        b.reset_gp = False
    return out


class B(BADS):
    def _search_step_(self, gp):
        CUR["phase_first"] = self.optim_state["search_count"] == 0
        return super()._search_step_(gp)

    def _poll_step_(self, gp):
        # poll_count == 0 at the first poll iteration: mark via a flag toggled after first lgf call
        CUR["phase_first"] = True
        orig = bb.local_gp_fitting

        def first_then(*a, **k):
            r = lgf(*a, **k)
            CUR["phase_first"] = False
            return r

        bb.local_gp_fitting = first_then
        try:
            return super()._poll_step_(gp)
        finally:
            bb.local_gp_fitting = lgf


bb.local_gp_fitting = lgf


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


for name, fun, D, seed in [
    ("rosen3", rosen, 3, 60),
    ("rosen3", rosen, 3, 61),
    ("ell4", ell, 4, 60),
    ("ell4", ell, 4, 61),
]:
    res = {}
    for mode in ("as is", "clear"):
        CUR.update(
            mode=mode, n=0, only_reset=0, n_train_in=[], phase_first=False
        )
        b = B(
            fun,
            0.5 * np.ones((1, D)) + 0.1 * np.arange(D),
            -5 * np.ones((1, D)),
            5 * np.ones((1, D)),
            -2 * np.ones((1, D)),
            2 * np.ones((1, D)),
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        )
        CUR["b"] = b
        r = b.optimize()
        res[mode] = (CUR["n"], CUR["only_reset"], r["fval"], r["func_count"])
    print(
        f"{name} seed {seed}: as is {res['as is'][0]} rebuilds, {res['as is'][1]} triggered only by reset_gp, fval {res['as is'][2]:.3e} | "
        f"cleared after rebuild: {res['clear'][0]} rebuilds, fval {res['clear'][2]:.3e} | log10 ratio {np.log10(res['as is'][2]/res['clear'][2]):+.2f}"
    )
