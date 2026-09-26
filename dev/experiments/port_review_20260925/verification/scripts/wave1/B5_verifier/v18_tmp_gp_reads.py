"""B5-R2: what the retry of _robust_gp_fit_ reads from tmp_gp after a failed fit (gpyreg 1.3.3)."""
import copy
import logging

import common  # noqa
import gpyreg as gpr
import numpy as np

from pybads import BADS
from pybads.bads import gaussian_process_train as gpt

logging.getLogger("BADS").setLevel(logging.ERROR)


def f(x):
    x = np.ravel(x)
    return float(np.sum((x - 0.3) ** 2 * np.array([1.0, 4.0, 9.0])))


D = 3
for mean in ("const", "negquad"):
    b = BADS(
        f,
        np.array([[1.0, -1.0, 0.5]]),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options={
            "random_seed": 5,
            "display": "off",
            "max_fun_evals": 70,
            "gp_mean_fun": mean,
        },
    )
    b.optimize()
    gp = b.iteration_history["gp"][b.optim_state["iter"]]
    print(
        f"gp_mean_fun={mean}: NaN bounds held by the GP at a refit: lower {int(np.isnan(gp.lower_bounds).sum())}, upper {int(np.isnan(gp.upper_bounds).sum())}"
    )

# slice-sampler branch: data the sampler reads vs data the next fit is given
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
gp = copy.deepcopy(b.iteration_history["gp"][b.optim_state["iter"]])
o = copy.deepcopy(b.options)
o["use_slice_sampler"] = True
orig_fit = gpr.GP.fit
orig_ss = gpt._get_samples_from_slice_sampler_
LOG = []
C = {"n": 0}


def fit(self, X=None, y=None, s2=None, hyp0=None, options=None, rng=None):
    C["n"] += 1
    LOG.append(
        f"fit {C['n']}: given X rows {X.shape[0]}, tmp_gp holds {self.X.shape[0]}"
    )
    if C["n"] <= 3:
        raise np.linalg.LinAlgError("injected")
    return orig_fit(self, X, y, s2, hyp0=hyp0, options=options, rng=rng)


def ss(g, hyp, os_, opts, rng=None):
    x0 = np.atleast_2d(hyp)[-1]
    LOG.append(
        f"   slice sampler reads the data of tmp_gp: {g.X.shape[0]} rows; start - LB min {np.min(x0 - g.lower_bounds):.3g} at {np.argmin(x0 - g.lower_bounds)}, UB - start min {np.min(g.upper_bounds - x0):.3g} at {np.argmin(g.upper_bounds - x0)}"
    )
    try:
        return orig_ss(g, hyp, os_, opts, rng)
    except ValueError as e:
        LOG.append(f"   slice sampler raised ValueError: {e}")
        raise


gpr.GP.fit = fit
gpt._get_samples_from_slice_sampler_ = ss
try:
    hyp = gp.get_hyperparameters(as_array=True)
    gp_train = gpt._get_gp_training_options(
        b.optim_state, b.iteration_history, o, hyp, 0, b.function_logger
    )
    gpt._robust_gp_fit_(
        gp,
        gp.X,
        gp.y,
        gp.s2,
        hyp,
        gp_train,
        b.optim_state,
        o,
        np.random.default_rng(1),
    )
except ValueError as e:
    LOG.append(f"_robust_gp_fit_ raised ValueError: {e}")
finally:
    gpr.GP.fit = orig_fit
    gpt._get_samples_from_slice_sampler_ = orig_ss
print("\n".join(LOG))
