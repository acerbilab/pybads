import collections

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

cur = {"n": 0}
rows = []
orig_fit = gpyreg.GP.fit


def counted_fit(self, *a, **k):
    try:
        return orig_fit(self, *a, **k)
    except np.linalg.LinAlgError:
        cur["n"] += 1
        raise


gpyreg.GP.fit = counted_fit
orig_rob = gpt._robust_gp_fit_


def rob(gp, *a, **k):
    cur["n"] = 0
    lb0 = float(np.ravel(gp.get_bounds()["noise_log_scale"][0])[0])
    out = orig_rob(gp, *a, **k)
    fitted = float(out[0].get_hyperparameters()[0]["noise_log_scale"][0])
    rows.append((cur["n"], lb0, fitted))
    return out


gpt._robust_gp_fit_ = rob


def rosen(x):
    x = np.atleast_2d(x)
    return float(
        np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2)
    )


def ell(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


for name, f, D in [("rosen3", rosen, 3), ("ell6", ell, 6)]:
    for seed in [0, 1]:
        b = BADS(
            f,
            np.full((1, D), 1.5),
            np.full((1, D), -10.0),
            np.full((1, D), 10.0),
            np.full((1, D), -3.0),
            np.full((1, D), 3.0),
            options={
                "random_seed": seed,
                "display": "off",
                "max_fun_evals": 200,
            },
        )
        b.optimize()
by = collections.defaultdict(list)
for n, lb0, fit in rows:
    by[n].append(fit - lb0)
for n in sorted(by):
    v = np.array(by[n])
    nudge = {0: 0, 1: 1, 2: 3}.get(n, None)
    at_bound = (
        np.mean(np.isclose(v, nudge, atol=1e-3))
        if nudge is not None
        else np.nan
    )
    print(
        f"{n} failures: {len(v)} refits; fitted log noise minus original lower bound: median {np.median(v):.2f}, min {v.min():.2f}; share at the nudged bound (+{nudge}): {at_bound:.2f}"
    )
