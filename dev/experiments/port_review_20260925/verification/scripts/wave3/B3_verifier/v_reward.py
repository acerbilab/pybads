"""F5 / B3-K1: the hedge's expected reward, port vs acqPortfolio.m:64; in-run at level 1."""
import numpy as np
import vhdr  # noqa
from scipy.special import erfc

from pybads import BADS
from pybads.search.search_hedge import ESSearchHedge


def er_port(g, s):
    fpi = 0.5 * erfc(-g / np.sqrt(2))
    return s * (g * fpi + np.exp(-0.5 * (g**2) / np.sqrt(2 * np.pi)))


def er_matlab(g, s):
    fpi = 0.5 * erfc(-g / np.sqrt(2))
    return s * (g * fpi + np.exp(-0.5 * (g**2)) / np.sqrt(2 * np.pi))


for g in (-3, -2, -1, 0, 1, 2):
    print(
        f"gamma={g:+d}: port er/sigma={er_port(g, 1):.4f}  MATLAB er/sigma={er_matlab(g, 1):.4f}  ratio={er_port(g, 1) / er_matlab(g, 1):.2f}"
    )

# In a level-1 run: the chosen strategy's reward under both formulas
rec = []
orig = ESSearchHedge.update_hedge


def upd(self, u_search, fval_old, f, fs, gp, mesh_size):
    fs_ = float(np.ravel(fs)[0]) if np.size(fs) else fs
    if fs_ != 0 and np.isfinite(f) and np.isfinite(fs_) and fs_ > 0:
        g = (fval_old - f) / fs_
        rec.append(
            (
                g,
                er_port(g, fs_),
                er_matlab(g, fs_),
                self.prob[self.chosen_hedge].item(),
            )
        )
    else:
        rec.append((np.nan, np.nan, np.nan, np.nan))
    return orig(self, u_search, fval_old, f, fs, gp, mesh_size)


ESSearchHedge.update_hedge = upd
rng = np.random.default_rng(0)
D = 3
res = BADS(
    lambda x: float(np.sum(np.atleast_2d(x) ** 2)) + rng.standard_normal(),
    np.full(D, 2.0),
    np.full(D, -10.0),
    np.full(D, 10.0),
    np.full(D, -3.0),
    np.full(D, 3.0),
    options={
        "display": "off",
        "random_seed": 0,
        "max_fun_evals": 150,
        "uncertainty_handling": True,
    },
).optimize()
a = np.array(rec)
ok = np.isfinite(a[:, 0])
print(
    f"level-1 run: {len(a)} hedge updates, {ok.sum()} through the SD>0 branch"
)
print(
    f"  gamma quartiles {np.percentile(a[ok, 0], [10, 25, 50, 75, 90]).round(2).tolist()}"
)
print(
    f"  port/MATLAB reward ratio quartiles {np.percentile(a[ok, 1] / a[ok, 2], [10, 25, 50, 75, 90]).round(2).tolist()}"
)
