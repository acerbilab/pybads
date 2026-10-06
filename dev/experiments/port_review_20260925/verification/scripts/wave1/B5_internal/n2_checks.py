import collections
import inspect

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
from scipy.stats import chi2

import pybads.bads.bads as bb
from pybads import BADS

dec = collections.Counter()
orig_irt = bb.BADS._is_gp_refit_time_


def irt(self, alpha):
    it = self.gp_stats.get("iter_gp")
    n = 0 if it is None else len(it)
    caller = inspect.currentframe().f_back.f_code.co_name
    out = orig_irt(self, alpha)
    key = f"{caller} n={min(n,3) if n < 3 else '>=3'}"
    dec[key + " calls"] += 1
    dec[key + " do_gp_calibration"] += bool(out[1])
    dec[key + " refit"] += bool(out[0])
    if n == 2:
        f = self.gp_stats.get("fval").astype(float)
        m = self.gp_stats.get("ymu").astype(float)
        s = self.gp_stats.get("ys").astype(float)
        s[np.isclose(0, s)] = 1e-6
        dec["n=2 flagged by correct chi2(2) at alpha"] += (
            chi2.sf(np.sum(((f - m) / s) ** 2), 2) < alpha / 2
        )
    return out


bb.BADS._is_gp_refit_time_ = irt


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ell(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


for name, f, D in [("rosen2", rosen, 2), ("ell4", ell, 4)]:
    dec.clear()
    b = BADS(
        f,
        np.full((1, D), 1.5),
        np.full((1, D), -10.0),
        np.full((1, D), 10.0),
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        options={"random_seed": 0, "display": "off", "max_fun_evals": 200},
    )
    b.optimize()
    print(name)
    for k in sorted(dec):
        print("   ", k, int(dec[k]))
