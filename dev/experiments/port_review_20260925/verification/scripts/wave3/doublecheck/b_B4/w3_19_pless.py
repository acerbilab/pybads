"""W3-19: p_less of the port (bads.py:2337-2342 at 0d866e8) against a
transcription of bads.m:862-872, on random probabilities for n below, at and
above D, and on the (f_mu, fs) that the poll's acquisition returns in real
runs (their shapes, and gamma_z's)."""
import logging

import gpyreg
import numpy as np
from scipy.special import erfc

import pybads
import pybads.bads.bads as bads_module
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
logging.getLogger("BADS").setLevel(logging.ERROR)


def matlab_pless(ftarget, SufficientImprovement, fm, fs, nvars):
    # bads.m:862-872 with numel(gpstruct.hyp) == 1; fm, fs columns
    fm = np.asarray(fm, float).reshape(-1, 1)
    fs = np.asarray(fs, float).reshape(-1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gammaz = (ftarget - SufficientImprovement - fm) / fs
    if np.all(np.isfinite(gammaz)) and np.all(np.isreal(gammaz)):
        fpi = 0.5 * erfc(-gammaz / np.sqrt(2))
        fpi = -np.sort(-fpi, axis=0, kind="stable")  # sort(fpi,'descend')
        k = min(nvars, fpi.shape[0])  # fpi(1:min(nvars,end))
        return float(np.prod(1 - fpi[:k, 0])), False
    return 0.0, True


def port_pless(f_target, sufficient_improvement, f_mu, fs, D):
    # the port's lines, verbatim
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_z = (f_target - sufficient_improvement - f_mu) / fs
    if np.all(np.isfinite(gamma_z)) and np.all(np.isreal(gamma_z)):
        f_pi = 0.5 * erfc(-gamma_z / np.sqrt(2))
        f_pi = np.sort(f_pi, axis=None)[::-1]
        return float(np.prod(1 - f_pi[:D])), False
    return 0, True


def old_pless(f_target, sufficient_improvement, f_mu, fs, D):
    gamma_z = (f_target - sufficient_improvement - f_mu) / fs
    f_pi = 0.5 * erfc(-gamma_z / np.sqrt(2))
    f_pi = np.sort(f_pi)[::-1]
    return float(np.prod(1 - f_pi[0 : np.minimum(D + 1, len(f_pi))]))


rng = np.random.default_rng(0)
worst = 0.0
n_old_diff = 0
count = 0
for D in range(1, 8):
    for n in range(1, 2 * D + 1):
        for _ in range(300):
            fm = rng.normal(size=(n, 1)) * rng.choice([0.1, 1, 5])
            fs = rng.uniform(0.01, 2, size=(n, 1))
            ft = rng.normal()
            si = rng.uniform(0, 0.1)
            a, fa = matlab_pless(ft, si, fm, fs, D)
            b, fb = port_pless(ft, si, fm, fs, D)
            assert fa == fb
            worst = max(worst, abs(a - b) / max(abs(a), 1e-300))
            count += 1
            if abs(old_pless(ft, si, fm, fs, D) - a) > 1e-12 * max(a, 1e-300):
                n_old_diff += 1
print(
    f"random sets: {count}, max rel diff port vs MATLAB {worst:.2e}; "
    f"the old lines differ from MATLAB in {n_old_diff}",
    flush=True,
)
# zero SD: both unreliable
print(
    "zero SD:",
    matlab_pless(0, 0, np.zeros((3, 1)), np.array([[1], [0], [1]]), 2),
    port_pless(0, 0, np.zeros((3, 1)), np.array([[1], [0], [1]]), 2),
    flush=True,
)

# Real runs: the shapes the poll's acq_fcn_lcb returns, and the port's
# p_less from them against MATLAB's
shapes = set()
diffs = []
orig = bads_module.acq_fcn_lcb
holder = {}


def acq(u, func_count, gp):
    z, f_mu, fs = orig(u, func_count, gp)
    b = holder["bads"]
    if b.optim_state["search_count"] == 0:  # the poll
        shapes.add((u.shape, f_mu.shape, fs.shape))
        ft = b.optim_state["f_target"]
        si = b.sufficient_improvement
        a, _ = matlab_pless(ft, si, f_mu, fs, b.D)
        p, _ = port_pless(ft, si, f_mu, fs, b.D)
        diffs.append((len(u), abs(a - p)))
    return z, f_mu, fs


bads_module.acq_fcn_lcb = acq


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


for D, seed in ((3, 0), (4, 1), (2, 2)):
    b = BADS(
        rosen,
        np.full(D, -1.0),
        -5 * np.ones(D),
        5 * np.ones(D),
        -3 * np.ones(D),
        3 * np.ones(D),
        options={"display": "off", "max_fun_evals": 150, "random_seed": seed},
    )
    holder["bads"] = b
    b.optimize()
ns = sorted({n for n, _ in diffs})
print("poll acquisition shapes (u, f_mu, fs):", shapes, flush=True)
print(
    f"poll steps {len(diffs)}, set sizes {ns}, max |MATLAB - port| "
    f"{max(d for _, d in diffs):.2e}",
    flush=True,
)
