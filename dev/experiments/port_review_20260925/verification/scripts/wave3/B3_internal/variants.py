import sys
import time

import gpyreg
import numpy as np
from scipy.special import erfc

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.search.es_search as es
import pybads.search.search_hedge as sh
from pybads import BADS

orig_ucov = es.ucov
orig_mask = es.ESSearch._get_selection_idx_mask_
orig_upd = sh.ESSearchHedge.update_hedge


def ucov_fixed(U, u, w, ub, lb, scale, periodic_vars=None):
    # weighted scatter about u over the len(w) best rows, weights applied per row
    d = (U - u)[: len(w)] if w.size else (U - u)
    if w.size == 0:
        return d.T @ d
    return d.T @ (w[:, None] * d)


def mask_fixed(self, mu, lamb):
    tot = mu + lamb
    sq = np.sqrt(np.arange(1, tot + 1))
    w = np.ceil((1.0 / sq) / np.sum(1.0 / sq) * lamb).astype(int)
    nonzero = np.sum(w > 0)
    while (np.sum(w) - lamb) > nonzero:
        w = np.maximum(0, w - 1)
        nonzero = np.sum(w > 0)
    delta = np.sum(w) - lamb
    last = np.argwhere(w > 0)[-1].item()
    st = max(0, last - int(delta) + 1)
    w[st : last + 1] -= 1
    return np.repeat(np.arange(len(w)), w)  # parent k gets w[k] offspring


def upd_fixed(self, u_search, fval_old, f, fs, gp, mesh_size):
    for i in range(self.n_funs):
        if i == self.chosen_hedge:
            fh, sh_ = f, fs
        else:
            fh, sh_ = 0, 1
        if sh_ == 0:
            er = np.maximum(0, fval_old - fh)
        elif np.isfinite(fh) and np.isfinite(sh_) and sh_ > 0:
            g = (fval_old - fh) / sh_
            er = sh_ * (
                g * 0.5 * erfc(-g / np.sqrt(2))
                + np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi)
            )
        else:
            er = 0
        self.g[i] = self.decay * self.g[i] + er / self.phat[i] / mesh_size


def set_variant(v):
    es.ucov = ucov_fixed if "ucov" in v else orig_ucov
    es.ESSearch._get_selection_idx_mask_ = (
        mask_fixed if "mask" in v else orig_mask
    )
    sh.ESSearchHedge.update_hedge = upd_fixed if "ei" in v else orig_upd


rng0 = np.random.default_rng(12345)
Q4, _ = np.linalg.qr(rng0.normal(size=(4, 4)))
Q3, _ = np.linalg.qr(rng0.normal(size=(3, 3)))


def ellipsoid4(x):
    z = Q4 @ (np.asarray(x).ravel() - 0.7)
    return float(np.sum(10.0 ** (4 * np.arange(4) / 3) * z**2))


def rosen3(x):
    x = Q3 @ np.asarray(x).ravel() + 1.0
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def noisy_sphere3(seed):
    r = np.random.default_rng(1000 + seed)
    return (
        lambda x: float(np.sum((np.asarray(x).ravel() - 0.5) ** 2))
        + r.normal()
    )


problems = {
    "ellipsoid4": (lambda s: ellipsoid4, 4, {}),
    "rosen3": (lambda s: rosen3, 3, {}),
    "noisy_sphere3": (noisy_sphere3, 3, {"uncertainty_handling": True}),
}
which = sys.argv[1]
variants = sys.argv[2].split(",")
seeds = range(int(sys.argv[3]))
mk, D, extra = problems[which]
for v in variants:
    set_variant(v)
    vals = []
    t0 = time.time()
    for s in seeds:
        opts = {"random_seed": s, "max_fun_evals": 200, "display": "off"}
        opts.update(extra)
        b = BADS(
            mk(s),
            np.full(D, -1.5),
            np.full(D, -8.0),
            np.full(D, 8.0),
            np.full(D, -4.0),
            np.full(D, 4.0),
            options=opts,
        )
        r = b.optimize()
        if which.startswith("noisy"):
            xr = np.asarray(r["x"]).ravel()
            vals.append(float(np.sum((xr - 0.5) ** 2)))
        else:
            vals.append(r["fval"])
    vals = np.array(vals)
    print(
        f"{which} {v:10s} n={len(vals)} median={np.median(vals):.3g} log10-mean={np.mean(np.log10(vals+1e-300)):.2f} vals={np.array2string(vals, precision=2)} ({time.time()-t0:.0f}s)",
        flush=True,
    )
