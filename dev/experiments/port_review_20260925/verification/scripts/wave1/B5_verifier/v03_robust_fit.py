"""F3/F4 (internal), F2/F3/F12 (comparison): _robust_gp_fit_ under injected fit failures."""
import copy

import common  # noqa
import gpyreg as gpr
import numpy as np

from pybads import BADS
from pybads.bads import gaussian_process_train as gpt


def f(x):
    x = np.ravel(x)
    return float(np.sum((x - 0.3) ** 2 * np.array([1.0, 4.0, 9.0])))


D = 3
b = BADS(
    f,
    np.array([[1.0, -1.0, 0.5]]),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    options={"random_seed": 5, "display": "off", "max_fun_evals": 70},
)
b.optimize()
gp0 = b.iteration_history["gp"][b.optim_state["iter"]]
opts, os_ = b.options, b.optim_state
print(
    "training points:",
    gp0.X.shape[0],
    " noise bounds at entry:",
    gp0.get_bounds()["noise_log_scale"],
)

orig_fit = gpr.GP.fit
LOG = []


def run(n_fail, nudge=None, label=""):
    gp = copy.deepcopy(gp0)
    o = copy.deepcopy(opts)
    if nudge is not None:
        o["noise_nudge"] = nudge
    calls = {"n": 0}

    def fit(self, X=None, y=None, s2=None, hyp0=None, options=None, rng=None):
        calls["n"] += 1
        LOG.append(
            (
                calls["n"],
                float(
                    self.lower_bounds[gp.covariance.hyperparameter_count(D)]
                ),
                X.shape[0],
                float(np.ravel(hyp0)[gp.covariance.hyperparameter_count(D)]),
            )
        )
        if calls["n"] <= n_fail:
            raise np.linalg.LinAlgError("injected")
        return orig_fit(self, X, y, s2, hyp0=hyp0, options=options, rng=rng)

    LOG.clear()
    gpr.GP.fit = fit
    hyp = gp.get_hyperparameters(as_array=True)
    gp_train = gpt._get_gp_training_options(
        os_, b.iteration_history, o, hyp, 0, b.function_logger
    )
    try:
        gp_out, hyp_out, res, flag = gpt._robust_gp_fit_(
            gp,
            gp.X,
            gp.y,
            gp.s2,
            hyp,
            gp_train,
            os_,
            o,
            np.random.default_rng(1),
        )
        outcome = (
            f"returned flag={flag}, noise hyp={hyp_out[0, gp.covariance.hyperparameter_count(D)]:.3f}, "
            f"gp.X rows={gp_out.X.shape[0]}, gp noise lb={gp_out.get_bounds()['noise_log_scale'][0]}"
        )
    except Exception as e:
        outcome = f"RAISED {type(e).__name__}: {e}"
    finally:
        gpr.GP.fit = orig_fit
    print(
        f"--- {label}: {n_fail} injected failures, noise_nudge={o['noise_nudge']}"
    )
    lb0 = LOG[0][1]
    for k, (n, lb, nX, h0) in enumerate(LOG):
        print(
            f"   try {n}: noise lb={lb:7.3f} (MATLAB lb={lb0:7.3f}), X rows={nX}, noise start={h0:7.3f}"
        )
    print("   ->", outcome)


import sys

if len(sys.argv) > 1:
    run(10, nudge=np.array([0, 0]), label="ten failures, nudge [0,0]")
    run(9, nudge=np.array([0, 0]), label="nine failures, nudge [0,0]")
else:
    run(0, label="no failure")
    run(1, label="one failure")
    run(3, label="three failures")
    run(4, label="four failures")
    run(5, label="five failures")
    run(10, label="ten failures")
