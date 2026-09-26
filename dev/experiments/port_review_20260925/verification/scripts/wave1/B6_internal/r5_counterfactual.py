"""Q2: at each refit where the mean prior's normalization constant is 0, redo the same
rebuild (same GP, same data, a copy of the same generator state) with that constant
replaced by 1, which leaves the MAP unchanged in exact arithmetic, and compare the
fitted hyperparameters, the objective reached and the predictions."""
import copy
import time
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from gpyreg.gaussian_process import GP

import pybads.bads.bads as bmod
from pybads import BADS

PATCH = {"on": False}
orig_norm = GP._GP__recompute_normalization_constants


def patched_norm(self):
    orig_norm(self)
    if PATCH["on"]:
        nc = self.normalization_constants
        nc[nc == 0] = 1.0


GP._GP__recompute_normalization_constants = patched_norm

orig_lgf = bmod.local_gp_fitting
out_rows = []


def wrapped(gp, u, fl, options, optim_state, ih, refit_flag, rng=None):
    if not refit_flag:
        return orig_lgf(
            gp, u, fl, options, optim_state, ih, refit_flag, rng=rng
        )
    gp_c = copy.deepcopy(gp)
    os_c = copy.deepcopy(optim_state)
    rng_c = copy.deepcopy(rng)
    t0 = time.time()
    res = orig_lgf(gp, u, fl, options, optim_state, ih, refit_flag, rng=rng)
    t_real = time.time() - t0
    i = res[0].D + 3
    if res[0].normalization_constants[i] == 0:
        PATCH["on"] = True
        try:
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cf, _ = orig_lgf(
                    gp_c, u, fl, options, os_c, ih, refit_flag, rng=rng_c
                )
            t_cf = time.time() - t0
        finally:
            PATCH["on"] = False
        g = res[0]
        h_real = g.get_hyperparameters(as_array=True)[0]
        h_cf = cf.get_hyperparameters(as_array=True)[0]
        # objective of each at the patched (finite) normalization: compare MAP quality
        PATCH["on"] = True
        cf.set_priors(cf.get_priors())
        f_real = cf._GP__gp_obj_fun(h_real, False, False)
        f_cf = cf._GP__gp_obj_fun(h_cf, False, False)
        PATCH["on"] = False
        cf.set_priors(cf.get_priors())
        Xt = g.X[:5] + 0.01
        m1, v1 = g.predict(Xt)
        m2, v2 = cf.predict(Xt)
        out_rows.append(
            (
                t_real,
                t_cf,
                h_real,
                h_cf,
                f_real,
                f_cf,
                np.max(np.abs(m1 - m2)),
                np.max(np.abs(np.sqrt(v1) - np.sqrt(v2))),
            )
        )
    return res


bmod.local_gp_fitting = wrapped


def rosen(z):
    z = np.atleast_1d(z)
    return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2))


for D in (2, 3):
    out_rows.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            lambda x: rosen(np.atleast_1d(x) - 4),
            np.zeros(D),
            np.full(D, -10.0),
            np.full(D, 10.0),
            np.full(D, -1.0),
            np.full(D, 1.0),
            options=dict(display="off", random_seed=0, max_fun_evals=200),
        )
        r = b.optimize()
    print(
        f"\nrosen shifted by 4, D={D}: fval {r['fval']:.4g}, evals {r['func_count']}; refits with mean normalization 0: {len(out_rows)}"
    )
    for t_real, t_cf, h_real, h_cf, f_real, f_cf, dm, ds in out_rows:
        print(
            f"  time {t_real:.2f}s vs {t_cf:.2f}s | neg log post (finite norm) real {f_real:.6g} vs counterfactual {f_cf:.6g} | max|dmu| {dm:.3g} max|dsd| {ds:.3g}"
        )
        print(f"     hyp real {np.round(h_real, 3)}")
        print(f"     hyp cf   {np.round(h_cf, 3)}")
