"""_robust_gp_fit_ after repeated fit failures: the nudge of the noise's
starting point and lower bound, and what the caller receives when every
try fails. Inputs captured from the first refit of a short default run."""
import copy

import gpyreg
import gpyreg as gpr
import numpy as np

import pybads
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)

captured = {}
_orig = gpt._robust_gp_fit_


def spy(*args, **kwargs):
    if not captured and len(args[2]) >= 40:
        captured["args"] = copy.deepcopy(args)
        captured["kwargs"] = {
            k: (v if k == "rng" else copy.deepcopy(v))
            for k, v in kwargs.items()
        }
    return _orig(*args, **kwargs)


gpt._robust_gp_fit_ = spy
fun = lambda x: float(np.sum(np.ravel(x) ** 2))
D = 2
b = BADS(
    fun,
    np.full((1, D), 1.0),
    np.full((1, D), -5.0),
    np.full((1, D), 5.0),
    np.full((1, D), -2.0),
    np.full((1, D), 2.0),
    options={"display": "off", "random_seed": 0, "max_fun_evals": 120},
)
b.optimize()
gpt._robust_gp_fit_ = _orig

gp, X, Y, s2, hyp_gp, gp_train, optim_state, options = captured["args"][:8]
cov_N = gp.covariance.hyperparameter_count(D)
print(
    "noise index",
    cov_N,
    "; bounds of noise at entry:",
    gp.lower_bounds[cov_N],
    gp.upper_bounds[cov_N],
)
print(
    "options noise_nudge =",
    options["noise_nudge"],
    "remove_points_after_tries =",
    options["remove_points_after_tries"],
)
print(
    "hyp0 rows passed to the fit:",
    hyp_gp.shape[0],
    "noise start",
    hyp_gp[:, cov_N],
)

orig_fit = gpr.GP.fit


def make_fit(n_fail):
    calls = []

    def fit(self, X=None, y=None, s2=None, hyp0=None, options=None, rng=None):
        calls.append(
            dict(
                noise_lb=float(self.lower_bounds[cov_N]),
                noise_ub=float(self.upper_bounds[cov_N]),
                noise_start=np.atleast_2d(hyp0)[:, cov_N].copy(),
                n=len(y),
            )
        )
        if len(calls) <= n_fail:
            raise np.linalg.LinAlgError("injected")
        return orig_fit(self, X, y, s2, hyp0=hyp0, options=options, rng=rng)

    return fit, calls


for n_fail in (1, 3, 4, 10):
    fit, calls = make_fit(n_fail)
    gpr.GP.fit = fit
    g = copy.deepcopy(gp)
    try:
        out = gpt._robust_gp_fit_(
            g,
            X.copy(),
            Y.copy(),
            None if s2 is None else s2.copy(),
            hyp_gp.copy(),
            gp_train,
            optim_state,
            options,
            np.random.default_rng(0),
        )
        res = f"returned exit flag {out[3]}; noise of returned hyp {np.atleast_2d(out[1])[:, cov_N]}"
    except Exception as e:
        res = f"RAISED {type(e).__name__}: {e}"
    finally:
        gpr.GP.fit = orig_fit
    print(f"\n-- fit fails {n_fail} time(s): {res}")
    for i, c in enumerate(calls):
        print(
            f"   try {i + 1}: noise lower bound {c['noise_lb']:+.3f} "
            f"(upper {c['noise_ub']:+.1f}), noise start {np.round(c['noise_start'], 3)}, "
            f"n train {c['n']}"
        )
print(
    "\nMATLAB (gpHyperOptimize.m:161-167, NoiseNudge=[1 0]): after the k-th failure "
    "the noise start is theta0 + k and the lower bound is unchanged (lb + k*0)."
)
