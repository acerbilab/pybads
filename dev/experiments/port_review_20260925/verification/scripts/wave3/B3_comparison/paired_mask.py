import copy

import gpyreg
import numpy as np

import pybads

print(pybads.__file__, gpyreg.__file__)
import pybads.search.search_hedge as sh
from pybads import BADS
from pybads.search.es_search import ESSearch, ESSearchELL, ESSearchWM

captured = []
orig_call = sh.ESSearchHedge.__call__


def cap(self, u, lb, ub, fl, gp, optim_state):
    if len(captured) < 400:
        captured.append(
            (
                u.copy(),
                copy.deepcopy(fl),
                copy.deepcopy(gp),
                copy.deepcopy(optim_state),
                self.options_dict,
                self.non_box_cons,
            )
        )
    return orig_call(self, u, lb, ub, fl, gp, optim_state)


sh.ESSearchHedge.__call__ = cap


def rosen(x):
    x = np.atleast_2d(x)
    return np.sum(
        100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2
    )


D = 4
b = BADS(
    rosen,
    np.full(D, -1.5),
    np.full(D, -20.0),
    np.full(D, 20.0),
    np.full(D, -5.0),
    np.full(D, 5.0),
    options={"random_seed": 3, "display": "off", "max_fun_evals": 150},
)
b.optimize()
print("captured", len(captured))

orig_mask = ESSearch._get_selection_idx_mask_


def matlab_mask(self, mu, lamb):
    m = orig_mask(self, mu, lamb)
    return m[1:] - 1  # MATLAB 0-based parents: Python[1:] - 1


res = {"ES-wcm": [], "ES-ell": []}
from_iter1 = {"py": 0, "ml": 0, "n": 0}
for k, (u, fl, gp, st, opts, nbc) in enumerate(captured[::6]):
    for cls, name in [(ESSearchWM, "ES-wcm"), (ESSearchELL, "ES-ell")]:
        for seed in range(5):
            out = {}
            for tag in ["py", "ml"]:
                ESSearch._get_selection_idx_mask_ = (
                    orig_mask if tag == "py" else matlab_mask
                )
                s = cls(2048, 2048, opts, np.random.default_rng(seed))
                us, z = s(u, None, None, fl, gp, st, 1, nbc)
                out[tag] = float(z)
            res[name].append(out["ml"] - out["py"])
ESSearch._get_selection_idx_mask_ = orig_mask
for name, d in res.items():
    d = np.array(d)
    print(
        f"{name}: n={d.size}, best LCB (MATLAB mask - Python mask): mean {d.mean():.3g}, median {np.median(d):.3g}, "
        f"MATLAB better in {np.mean(d < 0):.2f}, Python better in {np.mean(d > 0):.2f}"
    )
# sanity: the patched mask equals the MATLAB transcription
from check_esupdate_lib import esupdate_matlab

s = ESSearchWM(2048, 2048, captured[0][4])
for mu in [2048, 1500, 37]:
    ESSearch._get_selection_idx_mask_ = matlab_mask
    a = s._get_selection_idx_mask_(mu, 2048)[: min(mu, 2048)]
    ESSearch._get_selection_idx_mask_ = orig_mask
    bm = esupdate_matlab(mu, 2048)[0][: min(mu, 2048)] - 1
    print("patched mask == MATLAB transcription:", mu, np.array_equal(a, bm))
