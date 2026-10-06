"""F7 (internal) / F6, F7 (comparison) / B5-R3 part 1: the SD stored in gp_stats, and the refit verdicts."""
import logging

import common  # noqa
import numpy as np
from matlab_transcriptions import is_refit_time

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


class B(BADS):
    def _save_gp_stats_(self, fval, ymu, ys):
        idx = np.flatnonzero(LAST["fs"] == ys)
        self._pred.append(LAST["ys"][idx[0]])
        self._lat.append(ys)
        self._fv.append(fval)
        self._mu.append(ymu)
        return super()._save_gp_stats_(fval, ymu, ys)

    def _record_gp_refit_(self):
        self._pred, self._lat, self._fv, self._mu = [], [], [], []
        return super()._record_gp_refit_()

    def _is_gp_refit_time_(self, alpha):
        fc = self.function_logger.func_count
        lf = self.optim_state["lastfitgp"]
        fv, mu, lat, pr = (
            list(self._fv),
            list(self._mu),
            list(self._lat),
            list(self._pred),
        )
        r, u = super()._is_gp_refit_time_(alpha)
        ml = is_refit_time(
            fc, self.D, lf, self.options["min_refit_time"], fv, mu, lat, alpha
        )
        mp = is_refit_time(
            fc, self.D, lf, self.options["min_refit_time"], fv, mu, pr, alpha
        )
        self._log.append(
            (len(fv), (r, u), ml, mp, min(lat) if lat else np.nan)
        )
        return r, u


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


runs = [
    ("rosen3", rosen, 3, 50, False),
    ("rosen3", rosen, 3, 51, False),
    ("ell3", ell, 3, 50, False),
    ("noisy sphere3", None, 3, 50, True),
]
for name, fun, D, seed, noisy in runs:
    opts = {"random_seed": seed, "display": "off", "max_fun_evals": 200}
    if noisy:
        r_ = np.random.default_rng(seed + 7)
        fun = lambda x, r_=r_: float(
            np.sum(np.ravel(x) ** 2) + r_.standard_normal()
        )
        opts["uncertainty_handling"] = True
    b = B(
        fun,
        0.5 * np.ones((1, D)) + 0.1 * np.arange(D),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options=opts,
    )
    b._pred, b._lat, b._fv, b._mu, b._log = [], [], [], [], []
    # The stats created in _init_optimization_ are empty; keep ours in step
    b.optimize()
    L = b._log
    n_checks = len(L)
    py_ref = sum(x[1][0] for x in L)
    ml_ref = sum(x[2][0] for x in L)
    mp_ref = sum(x[3][0] for x in L)
    d_py_mp = sum(x[1] != x[3] for x in L)
    d_ml_mp = sum(x[2] != x[3] for x in L)
    late = sum(
        1 for x in L if (not x[1][0]) and x[3][0] and x[0] == max(10, 2 * D)
    )
    zero_lat = sum(1 for x in L if x[1][0] and not x[3][0] and x[4] < 1e-8)
    print(
        f"{name} seed {seed}: checks {n_checks}; refits python {py_ref}, MATLAB-rule+latent {ml_ref}, MATLAB-rule+predictive {mp_ref}; "
        f"verdicts (refit,unrel) python!=MATLAB {d_py_mp}; latent!=predictive under MATLAB rule {d_ml_mp}; "
        f"periodic refit late by one {late}; python refits with a latent SD<1e-8 that MATLAB does not make {zero_lat}"
    )
    per = max(10, 2 * D)
    early = [x for x in L if x[1][0] and x[0] < per]
    print(
        f"    python refits before the period (test fired): {len(early)}, of which with a latent SD < 1e-8 among the stats: {sum(1 for x in early if x[4] < 1e-8)}; n at those: {[x[0] for x in early]}"
    )
    # ratio of predictive to latent SD at saved points (all stats of the run are not kept; use the log's last window)
