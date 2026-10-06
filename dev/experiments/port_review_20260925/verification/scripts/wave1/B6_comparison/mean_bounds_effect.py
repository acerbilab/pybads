"""The constant mean's bounds from the initial design vs MATLAB's (-inf, inf):
refit the last GP of an Ackley run (a) as PyBADS holds it, (b) with MATLAB's bounds on the mean,
(c) with the prior moved 40 SDs below the lower bound (underflowing normalization)."""
import copy
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

D = 6


def ackley(x):
    x = np.asarray(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        - np.exp(np.mean(np.cos(2 * np.pi * x)))
        + 20
        + np.e
    )


x0 = np.full(D, 3.3)
lb = np.full(D, -32.0)
ub = np.full(D, 32.0)
plb = np.full(D, -5.0)
pub = np.full(D, 5.0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bads = BADS(
        ackley,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"random_seed": 0, "max_fun_evals": 200, "display": "off"},
    )
    res = bads.optimize()
gp0 = bads.iteration_history["gp"][-1]
opts = {
    "init_method": "rand",
    "tol_opt": 1e-5,
    "init_N": 8,
    "opts_N": 1,
    "n_samples": 0,
}
xc = gp0.X[np.argmin(gp0.y)]


def far_points(gp, k=2.0):
    ell = np.exp(gp.get_hyperparameters()[0]["covariance_log_lengthscale"])
    rng = np.random.default_rng(1)
    d = rng.normal(size=(20, D))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return xc + k * d * ell


def refit(gp, label):
    rng = np.random.default_rng(123)
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        hyp, r, _ = gp.fit(
            options=opts, rng=rng, hyp0=gp0.get_hyperparameters(as_array=True)
        )
    h = gp.hyperparameters_to_dict(hyp)[0]
    lp = gp.log_posterior(hyp[0]) - gp.log_likelihood(hyp[0])
    mu, s2 = gp.predict(far_points(gp0))
    print(
        f"{label}: mean_hyp {h['mean_const'][0]:.4g}  log-prior {lp:.4g}  nlZ {r.fun:.4g}  "
        f"opt '{r.message}' nit {r.nit}  norm_const(mean) {gp.normalization_constants[-1]:.3g}  "
        f"pred 2 ell away: median {np.median(mu):.3g}  warnings {sorted(set(str(w.message)[:50] for w in W))}"
    )


print(
    "local training set: N",
    gp0.y.size,
    "min",
    gp0.y.min(),
    "median",
    np.median(gp0.y),
    "p90",
    np.percentile(gp0.y, 90, method="hazen"),
)
pr = gp0.get_priors()["mean_const"]
print("prior on mean:", pr, "bounds:", gp0.get_bounds()["mean_const"])
a = copy.deepcopy(gp0)
refit(a, "(a) PyBADS bounds")
b = copy.deepcopy(gp0)
bd = b.get_bounds()
bd["mean_const"] = (np.array([-np.inf]), np.array([np.inf]))
b.set_bounds(bd)
refit(b, "(b) MATLAB bounds (-inf,inf)")
c = copy.deepcopy(gp0)
p = c.get_priors()
lbm = c.get_bounds()["mean_const"][0][0]
p["mean_const"] = ("gaussian", (np.array([lbm - 40 * 0.1]), np.array([0.1])))
c.set_priors(p)
refit(c, "(c) PyBADS bounds, prior 40 SD below LB")
c2 = copy.deepcopy(c)
bd = c2.get_bounds()
bd["mean_const"] = (np.array([-np.inf]), np.array([np.inf]))
c2.set_bounds(bd)
refit(c2, "(c') same prior, MATLAB bounds")
