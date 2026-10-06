"""How often default runs reach _get_random_samples_from_priors_, and from where."""
import traceback
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

calls = []
orig = gpt._get_random_samples_from_priors_


def wrapped(gp, rng=None):
    caller = traceback.extract_stack(limit=3)[0]
    out = orig(gp, rng)
    calls.append(
        (
            caller.name,
            caller.lineno,
            out[0].copy(),
            gp.lower_bounds.copy(),
            gp.upper_bounds.copy(),
        )
    )
    return out


gpt._get_random_samples_from_priors_ = wrapped
second = []
orig_lgf = gpt.local_gp_fitting


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


probs = []
for D in (2, 4):
    probs.append((f"rosenbrock D={D}", lambda x: rosen(x), D, {}))
    probs.append((f"ackley D={D}", ackley, D, {}))
    rn = np.random.default_rng(11)
    probs.append(
        (
            f"noisy sphere D={D} (level 1)",
            lambda x, rn=rn: float(
                np.sum(np.atleast_1d(x) ** 2) + rn.normal()
            ),
            D,
            {"uncertainty_handling": True},
        )
    )
for name, f, D, opts in probs:
    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            f,
            np.full(D, 1.5),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options=dict(
                display="off", random_seed=1, max_fun_evals=200, **opts
            ),
        )
        r = b.optimize()
    sites = {}
    for c in calls:
        sites[(c[0], c[1])] = sites.get((c[0], c[1]), 0) + 1
    print(
        f"{name}: fval {r['fval']:.4g} evals {r['func_count']}; prior-sampler calls {len(calls)} {sites}"
    )
    for c in calls[:2]:
        h, lb, ub = c[2], c[3], c[4]
        print(
            "    draw:",
            np.round(h, 2),
            " outside bounds:",
            np.round(np.where(h < lb, h - lb, np.where(h > ub, h - ub, 0)), 1),
        )
