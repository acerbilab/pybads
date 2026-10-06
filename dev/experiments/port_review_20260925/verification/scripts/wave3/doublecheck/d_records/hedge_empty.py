"""KD-B3-5: MATLAB's acqPortfolio 'upd' on an empty search set (bads.m:667-672: fsearch = fval,
fsearchsd = 0, stale usearch; bads.m:722-724) against PyBADS's update_hedge(None)."""
import gpyreg
import numpy as np
from scipy.special import erfc

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)
from pybads.search.search_hedge import ESSearchHedge


def matlab_upd(g, chosen, gamma, prob, decay, f, fs, fvalold, mesh):
    n = len(g)
    g = g.copy()
    phat = np.ones(n) if gamma == 0 else np.full(n, np.inf)
    if gamma != 0:
        phat[chosen] = prob[chosen]
    for i in range(n):
        if i == chosen:
            fH, fsH = f, fs
        elif gamma == 0:
            raise RuntimeError("gpstructnew undefined (acqPortfolio.m:47)")
        else:
            fH, fsH = 0.0, 1.0
        if fsH == 0:
            er = max(0.0, fvalold - fH)
        elif np.isfinite(fH) and np.isfinite(fsH) and fsH > 0:
            gz = (fvalold - fH) / fsH
            fpi = 0.5 * erfc(-gz / np.sqrt(2))
            er = fsH * (gz * fpi + np.exp(-0.5 * gz**2) / np.sqrt(2 * np.pi))
        else:
            er = 0.0
        g[i] = decay * g[i] + er / phat[i] / mesh
    return g


rng = np.random.default_rng(0)
opts = {
    "hedge_gamma": 0.125,
    "hedge_beta": 1.0,
    "hedge_decay": 0.1 ** (1 / (2 * 3)),
    "n_search_iter": 2,
    "n_search": 2**12,
}
worst = 0.0
for t in range(200):
    h = ESSearchHedge(
        [("ES-wcm", 1), ("ES-ell", 1)],
        opts,
        None,
        rng=np.random.default_rng(t),
    )
    h.g = rng.normal(size=2) * 10
    h.chosen_hedge = np.array([rng.integers(0, 2)])
    h.prob = np.array([0.4, 0.6])
    fval = rng.normal()
    mesh = 2.0 ** rng.integers(-10, 1)
    gm = matlab_upd(
        h.g,
        h.chosen_hedge.item(),
        0.125,
        h.prob,
        h.decay,
        fval,
        0.0,
        fval,
        mesh,
    )
    h.update_hedge(None, fval, None, None, None, mesh)
    worst = max(worst, np.max(np.abs(gm - h.g)))
print("max |g_matlab - g_pybads| over 200 empty-set updates:", worst)
