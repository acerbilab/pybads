"""F1 (internal): optim_state plb/pub swapped -> poll_scale with infinite bounds."""
import common  # noqa
import numpy as np

import pybads.bads.bads as bb
from pybads import BADS

orig_init = bb.BADS._init_optim_state_
orig_lgf = bb.local_gp_fitting
REC = []


def rec_lgf(gp, u, fl, options, os_, ih, refit_flag, rng=None):
    gp, ef = orig_lgf(gp, u, fl, options, os_, ih, refit_flag, rng=rng)
    if refit_flag:
        ll = gp.get_hyperparameters()[0]["covariance_log_lengthscale"]
        REC.append(
            (
                np.exp(ll).copy(),
                np.array(gp.temporary_data["poll_scale"]).copy(),
            )
        )
    return gp, ef


def fixed_init(self):
    os_ = orig_init(self)
    os_["plb"], os_["pub"] = os_["pub"].copy(), os_["plb"].copy()
    return os_


bb.local_gp_fitting = rec_lgf


def f(x):
    x = np.ravel(x)
    return float((x[0]) ** 2 + (5 * x[1]) ** 2 + (30 * x[2]) ** 2)


D = 3
x0 = np.array([[1.0, -1.2, 0.8]])
plb, pub = -2 * np.ones((1, D)), 2 * np.ones((1, D))
cases = {
    "unbounded (as is)": (None, None, orig_init),
    "unbounded (names fixed)": (None, None, fixed_init),
    "bounded +-20 (as is)": (
        -20 * np.ones((1, D)),
        20 * np.ones((1, D)),
        orig_init,
    ),
    "x1 unbounded only (as is)": (
        np.array([[-np.inf, -20, -20]]),
        np.array([[np.inf, 20, 20]]),
        orig_init,
    ),
}
for name, (lb, ub, init) in cases.items():
    for seed in (11, 12):
        REC.clear()
        bb.BADS._init_optim_state_ = init
        b = BADS(
            f,
            x0,
            lb,
            ub,
            plb,
            pub,
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 120,
            },
        )
        r = b.optimize()
        print(
            f"{name:28s} seed {seed} optim_state plb={b.optim_state['plb']} pub={b.optim_state['pub']}"
        )
        for i, (ls, ps) in enumerate(REC[-3:]):
            print(
                f"   refit {len(REC)-3+i}: lengthscale={np.round(ls,3)} poll_scale={np.round(ps,3)}"
            )
        print(
            f"   fval={r['fval']:.3e} func_count={r['func_count']} refits={len(REC)}"
        )
bb.BADS._init_optim_state_ = orig_init
