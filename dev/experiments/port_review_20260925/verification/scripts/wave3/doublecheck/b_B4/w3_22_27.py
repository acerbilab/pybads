"""W3-22: the floating-point warnings a run emits and np.geterr() after it;
W3-27: with every acquisition value NaN, the random choices come from the
run's generator (same seed, same choices; NumPy's global state untouched)."""
import collections
import logging
import warnings

import gpyreg
import numpy as np

import pybads
import pybads.bads.bads as bm
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.CRITICAL)


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


class Noisy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        return float(np.sum(np.ravel(x) ** 2) + 0.3 * self.rng.normal())


def make(fun, D, seed, **opts):
    return BADS(
        fun,
        np.full(D, -1.2),
        -5 * np.ones(D),
        5 * np.ones(D),
        -3 * np.ones(D),
        3 * np.ones(D),
        options={"display": "off", "random_seed": seed, **opts},
    )


print("--- W3-22", flush=True)
for name, fun, D, opts in (
    ("sphere D2", lambda x: float(np.sum(np.ravel(x) ** 2)), 2, {}),
    ("rosenbrock D3", rosen, 3, {"max_fun_evals": 150}),
    (
        "noisy sphere D2",
        Noisy(5),
        2,
        {"max_fun_evals": 150, "uncertainty_handling": True},
    ),
):
    np.seterr(all="warn", under="ignore")
    before = np.geterr()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        make(fun, D, 0, **opts).optimize()
    after = np.geterr()
    kinds = collections.Counter(
        (
            str(x.message)[:60],
            x.filename.split("site-packages/")[-1][-45:],
            x.lineno,
        )
        for x in w
        if issubclass(x.category, RuntimeWarning)
    )
    print(
        f"{name}: geterr unchanged {before == after} (after: divide="
        f"{after['divide']}, invalid={after['invalid']}); RuntimeWarnings:",
        dict(kinds) or "none",
        flush=True,
    )

print("--- W3-27", flush=True)
orig = bm.acq_fcn_lcb


def acq(u, func_count, gp):
    z, f_mu, fs = orig(u, func_count, gp)
    return np.full(np.shape(z), np.nan), f_mu, fs


bm.acq_fcn_lcb = acq
orig_call = (
    bm.FunctionLogger.__call__ if hasattr(bm, "FunctionLogger") else None
)
from pybads.function_logger import FunctionLogger  # noqa: E402

evals = []
orig_fl = FunctionLogger.__call__


def call(self, x, *a, **k):
    evals.append(np.array(x, float).ravel().copy())
    return orig_fl(self, x, *a, **k)


FunctionLogger.__call__ = call
runs = {}
for seed in (3, 3, 4):
    evals.clear()
    np.random.seed(11)
    g0 = np.random.get_state()[1].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make(
            lambda x: float(np.sum(np.ravel(x) ** 2)),
            2,
            seed,
            max_fun_evals=60,
        ).optimize()
    g1 = np.random.get_state()[1]
    runs.setdefault(seed, []).append(np.array(evals))
    print(
        f"seed {seed}: {len(evals)} evaluations, global state untouched "
        f"{np.array_equal(g0, g1)}",
        flush=True,
    )
print(
    "same seed, same evaluations:",
    np.array_equal(runs[3][0], runs[3][1]),
    "; seeds 3 and 4 differ:",
    not (
        runs[3][0].shape == runs[4][0].shape
        and np.array_equal(runs[3][0], runs[4][0])
    ),
    flush=True,
)
