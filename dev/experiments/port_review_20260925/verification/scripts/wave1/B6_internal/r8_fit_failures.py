"""Where the GP fits of a default run fail (LinAlgError), with which hyperparameters,
and what the retries of _robust_gp_fit_ start from."""
import traceback
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from gpyreg.gaussian_process import GP

import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

fails = []
orig_fit = GP.fit


def fit(self, X=None, y=None, s2=None, hyp0=None, options=None, rng=None):
    try:
        return orig_fit(self, X, y, s2, hyp0=hyp0, options=options, rng=rng)
    except np.linalg.LinAlgError as e:
        tb = traceback.extract_tb(e.__traceback__)
        where = [
            f"{f.name}:{f.lineno}"
            for f in tb
            if "gaussian_process.py" in f.filename
            or "f_min_fill" in f.filename
        ][:4]
        fails.append(
            dict(
                where=where,
                hyp0=None if hyp0 is None else np.array(hyp0).copy(),
                lb=self.lower_bounds.copy(),
                ub=self.upper_bounds.copy(),
                N=len(self.y) if self.y is not None else None,
                ystd=float(np.std(y)) if y is not None else None,
            )
        )
        raise


GP.fit = fit
robust = []
orig_r = gpt._robust_gp_fit_


def rwrap(*a, **k):
    out = orig_r(*a, **k)
    robust.append(out[3])
    return out


gpt._robust_gp_fit_ = rwrap


def rosen(z):
    z = np.atleast_1d(z)
    return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2))


for D in (2, 4):
    fails.clear()
    robust.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            rosen,
            np.full(D, 1.5),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options=dict(display="off", random_seed=1, max_fun_evals=200),
        )
        r = b.optimize()
    print(
        f"\nrosenbrock D={D}: fval {r['fval']:.4g}; robust fits {len(robust)}, exit flags {dict(zip(*np.unique(robust, return_counts=True)))}; failed gp.fit calls {len(fails)}"
    )
    for f in fails[:4]:
        print(
            "   fail at",
            f["where"],
            "N",
            f["N"],
            "std(y)",
            round(f["ystd"], 3),
        )
        h = f["hyp0"]
        if h is not None:
            h = np.atleast_2d(h)
            print("      hyp0 rows:", np.round(h, 2).tolist())
            print(
                "      bounds lb:",
                np.round(f["lb"], 2).tolist(),
                "ub:",
                np.round(f["ub"], 2).tolist(),
            )
