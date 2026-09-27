"""Transcriptions of MATLAB BADS formulas, compared with PyBADS on the same
inputs: EvalImprovement, acqLCB's default schedule, acqPortfolio 'upd',
searchHedge's probabilities, gpupdate's geometry, udist, the RQ effective
radius, the final quantile."""

import gpyreg
import numpy as np
from scipy.special import erfc, erfcinv
from scipy.stats import norm

import pybads

print("pybads:", pybads.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)

from pybads.bads.bads import BADS
from pybads.search.grid_functions import udist
from pybads.search.search_hedge import ESSearchHedge

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- EvalImprovement
def matlab_eval_improvement(fbase, fnew, sbase, snew, q):
    mu = fbase - fnew
    sigma = np.sqrt(sbase**2 + snew**2)
    x0 = -np.sqrt(2) * erfcinv(2 * q)
    return sigma * x0 + mu


obj = BADS.__new__(BADS)  # _eval_improvement_ uses no state
maxdiff = 0.0
for q in [1e-3, 0.1, 0.25, 0.5, 0.75, 0.9]:
    fb, fn = rng.normal(size=5), rng.normal(size=5)
    sb, sn = rng.uniform(0, 2, 5), rng.uniform(0, 2, 5)
    zp = obj._eval_improvement_(fb, fn, sb, sn, q)
    zm = matlab_eval_improvement(fb, fn, sb, sn, q)
    maxdiff = max(maxdiff, np.max(np.abs(zp - zm)))
    # Derivation: q-quantile of fb_true - fn_true under independent Gaussians
    zd = norm.ppf(q, loc=fb - fn, scale=np.sqrt(sb**2 + sn**2))
    maxdiff = max(maxdiff, np.max(np.abs(zp - zd)))
print("EvalImprovement: max |py - matlab|, |py - quantile| =", maxdiff)
# Monte Carlo check of the quantile of the difference at q = 0.1
fb, fn, sb, sn, q = 1.0, 0.7, 0.3, 0.4, 0.1
samp = (fb + sb * rng.standard_normal(10**6)) - (
    fn + sn * rng.standard_normal(10**6)
)
print(
    "  q=0.1: formula",
    obj._eval_improvement_(fb, fn, sb, sn, q),
    "MC quantile",
    np.quantile(samp, q),
)
print(
    "  level 0 (SDs 0):",
    obj._eval_improvement_(1.0, 0.7, 0.0, 0.0, 0.3),
    "q=0.5:",
    obj._eval_improvement_(1.0, 0.7, 0.3, 0.4, 0.5),
)


# ---------------------------------------------------------------- final quantile
fq = 1e-3
sm = np.sqrt(2) * erfcinv(2 * fq)
print(
    "final quantile multiplier",
    sm,
    "= Phi^-1(1 - q):",
    norm.ppf(1 - fq),
)
# np.nanargmin on object arrays with NaN, as the final choice uses
fv = np.array([0.0, 1.0, np.nan, 0.5, 2.0], dtype=object)
fs = np.array([0.1, 0.1, np.nan, 0.3, 0.0], dtype=object)
qb = fv + sm * fs
print(
    "  q_beta (object):",
    qb,
    "nanargmin(q_beta[1:]) + 1 =",
    np.nanargmin(qb[1:]) + 1,
)


# ---------------------------------------------------------------- acqLCB schedule
def matlab_sqrtbeta(t, nvars):
    delta, nu = 0.1, 0.2
    return np.sqrt(nu * 2 * np.log(nvars * t**2 * np.pi**2 / (6 * delta)))


class FakeGP:
    def __init__(self, mu, s2):
        self.mu, self.s2 = mu, s2

    def predict(self, x):
        return self.mu.copy(), self.s2.copy()


from pybads.acquisition_functions.acq_fcn_lcb import acq_fcn_lcb

for D in [1, 3, 10]:
    for fc in [0, 10, 499]:
        mu = rng.normal(size=(7, 1))
        s2 = rng.uniform(0, 1, (7, 1))
        z, fm, fsd = acq_fcn_lcb(np.zeros((7, D)), fc, FakeGP(mu, s2))
        zm = mu - matlab_sqrtbeta(fc + 1, D) * np.sqrt(s2)
        assert np.allclose(z, zm), (D, fc)
print("acqLCB default schedule: equal (D in 1,3,10; funccount 0,10,499)")
print(
    "  sqrt_beta at D=2: t=1",
    matlab_sqrtbeta(1, 2),
    "t=100",
    matlab_sqrtbeta(100, 2),
    "t=1000",
    matlab_sqrtbeta(1000, 2),
)


# ---------------------------------------------------------------- Hedge
def matlab_hedge_update(
    g, n, chosen, gamma, phat, decay, u, f, fs, fvalold, MeshSize
):
    g = g.copy()
    for i in range(n):
        if i == chosen:
            fH, fsH = f, fs
        elif gamma == 0:
            raise NotImplementedError
        else:
            fH, fsH = 0.0, 1.0
        if fsH == 0:
            er = max(0, fvalold - fH)
        elif np.isfinite(fH) and np.isfinite(fsH) and fsH > 0:
            gz = (fvalold - fH) / fsH
            fpi = 0.5 * erfc(-gz / np.sqrt(2))
            er = fsH * (gz * fpi + np.exp(-0.5 * gz**2) / np.sqrt(2 * np.pi))
        else:
            er = 0
        g[i] = decay * g[i] + er / phat[i] / MeshSize
    return g


def matlab_hedge_probs(g, beta, gamma, n):
    p = np.exp(beta * (g - g.max())) / np.sum(np.exp(beta * (g - g.max())))
    return p * (1 - n * gamma) + gamma


D = 3
opts = {
    "hedge_gamma": 0.125,
    "hedge_beta": 1.0,
    "hedge_decay": 0.1 ** (1 / (2 * D)),
    "n_search_iter": 2,
    "n_search": 2**12,
}
h = ESSearchHedge(
    [("ES-wcm", 1), ("ES-ell", 1)], opts, rng=np.random.default_rng(1)
)
maxd = 0.0
for trial in range(200):
    g_before = h.g.copy()
    # replicate __call__'s probabilities and choice without running a search
    p_py = np.exp(h.beta * (h.g - np.max(h.g)))
    p_py = p_py / np.sum(np.exp(h.beta * (h.g - np.max(h.g))))
    p_py = p_py * (1 - h.n_funs * h.gamma) + h.gamma
    p_m = matlab_hedge_probs(h.g, h.beta, h.gamma, h.n_funs)
    maxd = max(maxd, np.max(np.abs(p_py - p_m)))
    chosen = int(rng.integers(0, 2))
    h.prob = p_py
    h.chosen_hedge = np.array([chosen])
    h.phat = np.full(2, np.inf)
    h.phat[chosen] = p_py[chosen]
    fvalold = rng.normal()
    if trial % 2:
        f, fs = fvalold + rng.normal(), 0.0
    else:
        f, fs = fvalold + rng.normal(), abs(rng.normal())
    mesh = 2.0 ** -int(rng.integers(0, 8))
    g_m = matlab_hedge_update(
        g_before,
        2,
        chosen,
        h.gamma,
        h.phat,
        h.decay,
        None,
        f,
        fs,
        fvalold,
        mesh,
    )
    h.update_hedge(np.zeros(D), fvalold, f, fs, None, mesh)
    maxd = max(maxd, np.max(np.abs(h.g - g_m)))
print("Hedge probabilities and update: max |py - matlab| =", maxd)
# Expected reward = E[max(0, fvalold - F)], F ~ N(f, fs^2): Monte Carlo
f, fs, fvalold = 0.3, 0.5, 0.1
F = f + fs * rng.standard_normal(10**6)
gz = (fvalold - f) / fs
er = fs * (
    gz * 0.5 * erfc(-gz / np.sqrt(2))
    + np.exp(-0.5 * gz**2) / np.sqrt(2 * np.pi)
)
print("  EI formula", er, "MC", np.mean(np.maximum(0, fvalold - F)))


# ---------------------------------------------------------------- geometry
def matlab_geometry(
    logell, logalpha, rescale, lb, ub, plb, pub, searchmesh, scale=1.0
):
    D = logell.size
    lenscale = np.exp(logell) if D > 1 else 1.0
    ll = rescale * logell
    ll = np.exp(1.0 * (ll - np.mean(ll)))
    ubb = ub.copy()
    ubb[~np.isfinite(ubb)] = pub[~np.isfinite(ubb)]
    lbb = lb.copy()
    lbb[~np.isfinite(lbb)] = plb[~np.isfinite(lbb)]
    ll = np.minimum(np.maximum(ll, searchmesh), (ubb - lbb) / scale)
    alpha = np.exp(logalpha)
    effr = np.sqrt(alpha * (np.exp(1 / alpha) - 1))
    return lenscale, ll, effr


for alpha in [1e-2, 0.3, 1.0, np.e, 30.0, 1e4]:
    effr = np.sqrt(alpha * (np.exp(1 / alpha) - 1))
    r = np.sqrt(2) * effr
    k = (1 + r**2 / (2 * alpha)) ** (-alpha)
    print(
        f"  alpha={alpha:g}: effective radius {effr:.6g}, kernel at sqrt(2)*R = {k:.6g} (e^-1 = {np.exp(-1):.6g})"
    )

# udist vs MATLAB's udist.m
U = rng.normal(size=(20, 4))
u2 = rng.normal(size=(1, 4))
ls = np.exp(rng.normal(size=4))
d_py = udist(
    U, u2, ls, -np.ones(4), np.ones(4), 1.0, np.zeros(4, bool)
).ravel()
d_m = np.sum(((U - u2) / ls) ** 2, axis=1)
print("udist: max |py - matlab| =", np.max(np.abs(d_py - d_m)))
