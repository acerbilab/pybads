"""Degenerate inputs to the priors: targets that are all equal on the initial design,
fit_lik=False, and a positive upper_gp_length_factor."""
import traceback
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

D = 2


def attempt(name, f, opts, x0=np.zeros(D)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            b = BADS(
                f,
                x0,
                np.full(D, -10.0),
                np.full(D, 10.0),
                np.full(D, -1.0),
                np.full(D, 1.0),
                options=dict(
                    display="off", random_seed=0, max_fun_evals=100, **opts
                ),
            )
            r = b.optimize()
            print(f"{name}: ran, fval {r['fval']:.4g} evals {r['func_count']}")
            return b
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            print(f"{name}: {type(e).__name__}: {str(e)[:150]}")
            print(
                "    raised from:", [f"{t.name}:{t.lineno}" for t in tb][-5:]
            )


# 1. flat on the plausible box: every initial target equal
plateau = lambda x: float(
    5.0 + max(0.0, np.max(np.abs(np.atleast_1d(x))) - 3.0) ** 2
)
attempt("plateau (targets all 5 on the initial design)", plateau, {})
# 2. flat on the plausible box except x0 slightly lower
plateau2 = lambda x: float(
    5.0
    - 1e-3 * (np.allclose(x, 0))
    + max(0.0, np.max(np.abs(np.atleast_1d(x))) - 3.0) ** 2
)
attempt("plateau with x0 lower by 1e-3", plateau2, {})
# 3. fit_lik = False
attempt(
    "sphere, fit_lik=False",
    lambda x: float(np.sum(np.atleast_1d(x) ** 2)),
    {"fit_lik": False},
    x0=np.full(D, 0.5),
)
# 4. upper_gp_length_factor > 0: the bounds that _gp_hyp sets
captured = {}
orig = gpt._gp_hyp


def wrap(*a, **k):
    out = orig(*a, **k)
    captured["b"] = out[0].get_bounds()["covariance_log_lengthscale"]
    return out


gpt._gp_hyp = wrap
for fac in (0, 0.05):
    attempt(
        f"sphere, upper_gp_length_factor={fac}",
        lambda x: float(np.sum(np.atleast_1d(x) ** 2)),
        {"upper_gp_length_factor": fac},
        x0=np.full(D, 0.5),
    )
    print(
        "    length-scale bounds after _gp_hyp:",
        [np.round(v, 3).tolist() for v in captured["b"]],
    )
