"""W3-6, W3-7, W3-11: ESSearchHedge.update_hedge against a transcription of
acqPortfolio.m's 'upd' branch (74919c0), with MATLAB's intent at
HedgeGamma = 0 (gppred's latent mean and variance at the search point, the
line that uses the undefined gpstructnew read as gpstruct)."""
import hdr  # noqa: F401
import numpy as np
from scipy.special import erfc

from pybads import BADS
from pybads.function_examples import rosenbrocks_fcn
from pybads.search.search_hedge import ESSearchHedge


def upd_matlab(g, chosen, gamma, phat, decay, u, f, fs, gp, fvalold, MeshSize):
    g = g.copy()
    n = g.size
    u = np.atleast_2d(u)
    for i in range(n):
        uH = u[min(i, u.shape[0] - 1)][None, :]
        if i == chosen:
            fH, fsH = f, fs
        elif gamma == 0:
            m, s2 = gp.predict(uH)
            fH, fsH = m.item(), np.sqrt(s2).item()
        else:
            fH, fsH = 0.0, 1.0
        if fsH == 0:
            er = max(0.0, fvalold - fH)
        elif (
            np.isfinite(fH) and np.isfinite(fsH) and np.isreal(fsH) and fsH > 0
        ):
            gz = (fvalold - fH) / fsH
            fpi = 0.5 * erfc(-gz / np.sqrt(2))
            er = fsH * (gz * fpi + np.exp(-0.5 * gz**2) / np.sqrt(2 * np.pi))
        else:
            er = 0.0
        g[i] = decay * g[i] + er / phat[i] / MeshSize
    return g


D = 3
b = BADS(
    rosenbrocks_fcn,
    np.zeros((1, D)),
    -20 * np.ones((1, D)),
    20 * np.ones((1, D)),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    options={"random_seed": 0, "display": "off"},
)
b.options["fun_eval_start"] = 10
gp, _, _, _ = b._init_optimization_()
rng = np.random.default_rng(0)
worst = 0.0
n = 0
for gamma in (0.125, 0.0, 0.3):
    for trial in range(200):
        opts = dict(b.options)
        opts["hedge_gamma"] = gamma
        h = ESSearchHedge(
            b.options["search_method"], opts, rng=np.random.default_rng(trial)
        )
        h.g = rng.normal(size=2) * 5
        # a choice as __call__ makes it
        h.prob = np.exp(h.beta * (h.g - h.g.max()))
        h.prob /= h.prob.sum()
        h.prob = h.prob * (1 - h.n_funs * h.gamma) + h.gamma
        h.chosen_hedge = np.array([int(rng.integers(0, 2))])
        h.phat = np.ones(2) if gamma == 0 else np.full(2, np.inf)
        if gamma != 0:
            h.phat[h.chosen_hedge] = h.prob[h.chosen_hedge]
        u = rng.uniform(-1, 1, size=D)
        fval_old = float(rng.normal())
        f = float(rng.normal())
        fs = float(abs(rng.normal())) if trial % 3 else 0.0
        mesh = 2.0 ** -int(rng.integers(0, 10))
        g0 = h.g.copy()
        ref = upd_matlab(
            g0,
            h.chosen_hedge.item(),
            gamma,
            h.phat,
            h.decay,
            u,
            f,
            fs,
            gp,
            fval_old,
            mesh,
        )
        h.update_hedge(u, fval_old, f, fs, gp, mesh)
        d = np.max(np.abs(h.g - ref) / np.maximum(1e-300, np.abs(ref)))
        worst = max(worst, d)
        n += 1
print(
    f"update_hedge against acqPortfolio 'upd' on {n} states (gamma 0.125, 0, 0.3): worst relative difference {worst:.2e}"
)

# W3-11: an empty set. MATLAB: the chosen search gets f = fval = fvalold and
# fs = 0 at the stale usearch, the others 0 and 1 (gamma > 0), divided by Inf
worst = 0.0
for trial in range(200):
    opts = dict(b.options)
    h = ESSearchHedge(
        b.options["search_method"], opts, rng=np.random.default_rng(trial)
    )
    h.g = rng.normal(size=2) * 5
    h.prob = np.array([0.6, 0.4])
    h.chosen_hedge = np.array([int(rng.integers(0, 2))])
    h.phat = np.full(2, np.inf)
    h.phat[h.chosen_hedge] = h.prob[h.chosen_hedge]
    fval = float(rng.normal())
    stale = rng.uniform(-1, 1, size=D)
    ref = upd_matlab(
        h.g,
        h.chosen_hedge.item(),
        h.gamma,
        h.phat,
        h.decay,
        stale,
        fval,
        0.0,
        gp,
        fval,
        2.0**-3,
    )
    h.update_hedge(None, fval, fval, 0.0, gp, 2.0**-3)
    worst = max(worst, np.max(np.abs(h.g - ref)))
print(
    f"empty set (gamma 0.125): worst absolute difference from MATLAB's update at the stale point {worst:.2e}"
)

# At gamma = 0 MATLAB's intent would score the search not chosen at the stale point
h = ESSearchHedge(
    b.options["search_method"],
    {**b.options, "hedge_gamma": 0},
    rng=np.random.default_rng(0),
)
h.g = np.array([3.0, 1.0])
h.chosen_hedge = np.array([0])
h.phat = np.ones(2)
m, _ = gp.predict(gp.X[:1])
ref = upd_matlab(h.g, 0, 0.0, h.phat, h.decay, gp.X[0], 5.0, 0.0, gp, 5.0, 1.0)
h.update_hedge(None, 5.0, 5.0, 0.0, gp, 1.0)
print(
    f"empty set at gamma 0: port gains {h.g.tolist()}, MATLAB's intent at a stale point {ref.tolist()}"
)

# W3-6: the reward at gamma values, port formula vs phi
from scipy.stats import norm

for gz in (0.0, -1.0, -3.0, 2.0):
    old = gz * norm.cdf(gz) + np.exp(-0.5 * gz**2 / np.sqrt(2 * np.pi))
    new = gz * norm.cdf(gz) + norm.pdf(gz)
    print(
        f"gamma {gz:+.1f}: old reward/sigma {old:.6g}, MATLAB's {new:.6g}, ratio {old / new:.3g}"
    )
