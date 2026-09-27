import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
from scipy.special import erfc

import pybads.search.search_hedge as sh
from pybads import BADS


def er_py(fold, f, s):
    g = (fold - f) / s
    fpi = 0.5 * erfc(-g / np.sqrt(2))
    return s * (g * fpi + np.exp(-0.5 * (g**2) / np.sqrt(2 * np.pi)))


def er_ml(fold, f, s):  # acqPortfolio.m:112
    g = (fold - f) / s
    fpi = 0.5 * erfc(-g / np.sqrt(2))
    return s * (g * fpi + np.exp(-0.5 * (g**2)) / np.sqrt(2 * np.pi))


for g in [-3, -2, -1, 0, 1, 2]:
    print(
        f"gamma_z={g:+d}: Python er/sd = {er_py(g, 0, 1):.4f}, MATLAB er/sd = {er_ml(g, 0, 1):.4f}"
    )

MODE = [None]
log = []
orig = sh.ESSearchHedge.update_hedge


def upd(self, u_search, fval_old, f, fs, gp, mesh_size):
    if np.isfinite(f) and np.isfinite(fs) and fs > 0:
        log.append(
            (
                (fval_old - f) / fs,
                er_py(fval_old, f, fs),
                er_ml(fval_old, f, fs),
            )
        )
    if MODE[0] == "ml":
        # MATLAB's formula for the chosen strategy (the others get er/Inf = 0 either way)
        for i in range(self.n_funs):
            if i == self.chosen_hedge.item():
                if fs == 0:
                    er = max(0, fval_old - f)
                elif np.isfinite(f) and np.isfinite(fs) and fs > 0:
                    er = er_ml(fval_old, f, fs)
                else:
                    er = 0
                self.g[i] = (
                    self.decay * self.g[i] + er / self.phat[i] / mesh_size
                )
            else:
                self.g[i] = self.decay * self.g[i]
        return
    return orig(self, u_search, fval_old, f, fs, gp, mesh_size)


sh.ESSearchHedge.update_hedge = upd

probs = []
orig_call = sh.ESSearchHedge.__call__


def call(self, *a):
    out = orig_call(self, *a)
    probs.append((self.prob[0], self.chosen_hedge.item()))
    return out


sh.ESSearchHedge.__call__ = call

D = 3
for mode in ["py", "ml"]:
    for seed in range(3):
        MODE[0] = mode
        log.clear()
        probs.clear()
        rng = np.random.default_rng(1000 + seed)
        f = lambda x: np.sum(np.ravel(x) ** 2) + rng.normal()
        b = BADS(
            f,
            np.full(D, 3.0),
            np.full(D, -20.0),
            np.full(D, 20.0),
            np.full(D, -5.0),
            np.full(D, 5.0),
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
                "uncertainty_handling": True,
                "noise_size": 1.0,
            },
        )
        r = b.optimize()
        L = np.array(log)
        P = np.array(probs)
        ratio = L[:, 1] / np.maximum(L[:, 2], 1e-300)
        print(
            f"formula={mode} seed={seed}: fval={r['fval']:.3f} updates with sd>0: {len(L)}, median gamma_z {np.median(L[:,0]):.2f}, "
            f"median er ratio Py/ML {np.median(ratio):.2f}, P(ES-wcm) mean {P[:,0].mean():.2f}, "
            f"share of ES-wcm choices {np.mean(P[:,1]==0):.2f}"
        )
