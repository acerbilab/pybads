"""init_sobol called without its unused lb and ub, as 1.1.0's defaults
allowed (the changelog's init_sobol entries)."""
import gpyreg
import numpy as np

import pybads
from pybads.init_functions import init_sobol

print(pybads.__file__, gpyreg.__file__, flush=True)
u0 = np.zeros(3)
plb, pub = -np.ones(3), np.ones(3)
for kwargs in (
    dict(u0=u0, plb=plb, pub=pub, fun_eval_start=10),
    dict(u0=u0, lb=None, ub=None, plb=plb, pub=pub, fun_eval_start=10),
):
    try:
        u, n = (
            init_sobol(**kwargs, rng=np.random.default_rng(0))
            if "rng" in init_sobol.__code__.co_varnames
            else init_sobol(**kwargs)
        )
        print(sorted(kwargs), "->", u.shape, n)
    except Exception as e:
        print(sorted(kwargs), "->", type(e).__name__, e)
