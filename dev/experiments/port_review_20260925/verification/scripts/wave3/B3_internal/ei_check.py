import gpyreg
import numpy as np
from scipy.special import erfc
from scipy.stats import norm

import pybads

print(pybads.__file__, gpyreg.__file__)
from pybads.search.search_hedge import ESSearchHedge

opts = {
    "hedge_gamma": 0.125,
    "hedge_beta": 1.0,
    "hedge_decay": 1.0,
    "n_search_iter": 2,
    "n_search": 4096,
}
for gz in [-3, -2, -1, 0, 1, 2]:
    h = ESSearchHedge(options_dict=opts, rng=np.random.default_rng(0))
    h.g[:] = 0
    h.chosen_hedge = np.array([0])
    h.phat = np.array([1.0, np.inf])
    fs = 1.0
    f = -gz * fs  # fval_old = 0
    h.update_hedge(np.zeros(3), 0.0, f, fs, None, 1.0)
    ei = fs * (gz * norm.cdf(gz) + norm.pdf(gz))
    print(
        f"gamma_z={gz:+d}: port reward={h.g[0]:.4f}  expected improvement={ei:.4f}  ratio={h.g[0]/ei:.1f}"
    )
