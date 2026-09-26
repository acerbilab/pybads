"""Q3: the noise the posteriors actually carry: gpyreg multiplies the noise by ten per
failed Cholesky attempt and keeps the multiplier (sn2_mult) in the posterior used by
predict. How often, and how large, in default runs?"""
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bmod
from pybads import BADS

seen = []


def spy(name, orig):
    def w(*a, **k):
        out = orig(*a, **k)
        g = out[0] if isinstance(out, tuple) else out
        p = g.posteriors[0]
        seen.append((name, p.sn2_mult, bool(p.L_chol)))
        return out

    return w


bmod.local_gp_fitting = spy("rebuild", bmod.local_gp_fitting)
bmod.add_and_update_gp = spy("add", bmod.add_and_update_gp)


def rosen(z):
    z = np.atleast_1d(z)
    return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2))


def ackley(x):
    x = np.atleast_1d(x)
    d = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
        + 20
        + np.e
    )


for name, f, D in [
    ("sphere", lambda x: float(np.sum(np.atleast_1d(x) ** 2)), 2),
    ("rosenbrock", rosen, 2),
    ("rosenbrock", rosen, 4),
    ("ackley", ackley, 4),
]:
    seen.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            f,
            np.full(D, 1.5),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options=dict(display="off", random_seed=1, max_fun_evals=200),
        )
        r = b.optimize()
    m = np.array([s[1] for s in seen], dtype=float)
    vals, cnt = np.unique(m, return_counts=True)
    print(
        f"{name} D={D}: fval {r['fval']:.3g}; GP states {len(m)}; sn2_mult values {dict(zip(vals.tolist(), cnt.tolist()))}"
    )
