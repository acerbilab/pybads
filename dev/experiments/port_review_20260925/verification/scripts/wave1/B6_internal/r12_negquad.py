"""gp_mean_fun='negquad' and 'zero' (PyBADS-only options): do runs work, and what does the
negative quadratic mean predict away from the data in a minimization?"""
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.bads as bmod
from pybads import BADS

last = {}
orig = bmod.local_gp_fitting


def w(gp, *a, **k):
    out = orig(gp, *a, **k)
    last["gp"] = out[0]
    return out


bmod.local_gp_fitting = w
D = 2
for mf in ("const", "zero", "negquad"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            b = BADS(
                lambda x: float(10 + np.sum(np.atleast_1d(x) ** 2)),
                np.full(D, 1.5),
                np.full(D, -5.0),
                np.full(D, 5.0),
                np.full(D, -2.0),
                np.full(D, 2.0),
                options=dict(
                    display="off",
                    random_seed=0,
                    max_fun_evals=200,
                    gp_mean_fun=mf,
                ),
            )
            r = b.optimize()
        except Exception as e:
            print(mf, "raised", type(e).__name__, str(e)[:120])
            continue
    g = last["gp"]
    far = np.array([[4.0, 4.0], [-4.0, 4.0], [2.5, -2.5]])
    m, v = g.predict(far)
    h = g.get_hyperparameters()[0]
    print(
        f"{mf}: fval {r['fval']:.4g} evals {r['func_count']} | mean hyp {({k: np.round(v_, 3).tolist() for k, v_ in h.items() if k.startswith('mean')})} | predictions at u={far.tolist()}: {np.round(m.ravel(), 2).tolist()} (true in u units depend on transform)"
    )
