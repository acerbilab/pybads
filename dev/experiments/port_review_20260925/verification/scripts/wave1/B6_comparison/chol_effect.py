"""For the stored GPs of a default Rosenbrock D=4 run: posterior noise multiplier, effective noise SD,
residuals at the best training points, and whether the fitted hyperparameters admit an unmodified
factorization (MATLAB's CholAttempts = 0 raises where they do not)."""
import warnings

import gpyreg
import numpy as np
import scipy.linalg as sla

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

D = 4
fun = lambda x: float(
    np.sum(
        100 * (np.asarray(x)[1:] - np.asarray(x)[:-1] ** 2) ** 2
        + (1 - np.asarray(x)[:-1]) ** 2
    )
)
x0 = np.zeros(D)
lb = np.full(D, -5.0)
ub = np.full(D, 5.0)
plb = np.full(D, -2.0)
pub = np.full(D, 2.0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bads = BADS(
        fun,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"random_seed": 0, "max_fun_evals": 200, "display": "off"},
    )
    res = bads.optimize()
for it, gp in enumerate(bads.iteration_history["gp"]):
    p = gp.posteriors[0]
    h = gp.get_hyperparameters()[0]
    cov_N = D + 2
    K = gp.covariance.compute(p.hyp[:cov_N], gp.X)
    sn2 = np.exp(2 * h["noise_log_scale"][0])
    Lchol = sn2 >= 1e-6
    A = K / sn2 + np.eye(len(K)) if Lchol else K + sn2 * np.eye(len(K))
    try:
        sla.cholesky(A)
        ok = True
    except sla.LinAlgError:
        ok = False
    mu, s2 = gp.predict(gp.X)
    best = np.argsort(gp.y.ravel())[:10]
    print(
        f"it {it:2d} sn2_mult {p.sn2_mult:>4} unmodified chol ok {ok!s:5} sn {np.sqrt(sn2):.2e} "
        f"effective sn {np.sqrt(sn2*p.sn2_mult):.2e} y-range(best10) {np.ptp(gp.y[best]):.2e} "
        f"max|resid|(best10) {np.max(np.abs(mu[best]-gp.y[best])):.2e} ymin {gp.y.min():.2e}"
    )
