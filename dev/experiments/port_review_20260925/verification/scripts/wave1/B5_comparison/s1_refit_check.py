"""Compare BADS._is_gp_refit_time_ with a transcription of MATLAB's
IsRefitTime + gppredcheck + swtest on the same statistics."""
import types

import gpyreg
import matlab_ref as M
import numpy as np
from scipy.stats import shapiro
from scipy.stats import t as student_t

import pybads
from pybads.bads.bads import BADS
from pybads.utils.iteration_history import IterationHistory

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)

ALPHA = 1e-6


def fake_bads(D, func_count, lastfitgp, min_refit_time):
    s = types.SimpleNamespace()
    s.D = D
    s.function_logger = types.SimpleNamespace(func_count=func_count)
    s.optim_state = {"lastfitgp": lastfitgp}
    s.options = {"min_refit_time": min_refit_time}
    s.gp_stats = IterationHistory(["iter_gp", "fval", "ymu", "ys", "gp"])
    s._record_gp_refit_ = lambda: BADS._record_gp_refit_(s)
    s._save_gp_stats_ = lambda f, m, sd: BADS._save_gp_stats_(s, f, m, sd)
    return s


def python_decision(D, fc, last, mrt, stats):
    s = fake_bads(D, fc, last, mrt)
    for f, m, sd in stats:
        s._save_gp_stats_(f, m, sd)
    refit, unrel = BADS._is_gp_refit_time_(s, ALPHA)
    return refit, unrel


def matlab_decision(D, fc, last, mrt, stats):
    f = [a for a, _, _ in stats]
    m = [b for _, b, _ in stats]
    sd = [c for _, _, c in stats]
    try:
        unrel = M.gppredcheck(f, m, sd, ALPHA)
    except Exception:
        unrel = True
    refit = M.is_refit_time(last, fc, len(stats), unrel, D, mrt)
    if refit:
        unrel = False
    return refit, unrel


D = 2
print("\n== Case A: n = 1 stat, z = 1 (min refit time not passed)")
st = [(1.0, 0.0, 1.0)]
print(" Python (refit, unreliable):", python_decision(D, 20, 18, 2 * D, st))
print(" MATLAB (refit, unreliable):", matlab_decision(D, 20, 18, 2 * D, st))
print("== Case A': n = 1 stat, z = 1 (min refit time passed)")
print(" Python:", python_decision(D, 30, 10, 2 * D, st))
print(" MATLAB:", matlab_decision(D, 30, 10, 2 * D, st))

print("\n== Case B: n = 2 stats, z = (3, 3), sum z^2 = 18")
st = [(3.0, 0.0, 1.0), (3.0, 0.0, 1.0)]
print(" Python:", python_decision(D, 20, 18, 2 * D, st))
print(" MATLAB:", matlab_decision(D, 20, 18, 2 * D, st))
from scipy.special import gammaincinv
from scipy.stats import chi2

for v in (1, 2):
    print(
        f" v={v}: Python bounds gammaincinv(v/2,.)=",
        gammaincinv(v / 2, ALPHA / 2),
        gammaincinv(v / 2, 1 - ALPHA / 2),
        " chi2.ppf=",
        chi2.ppf(ALPHA / 2, v),
        chi2.ppf(1 - ALPHA / 2, v),
    )

print("\n== Case C: periodic refit, D=2 (refit period 10), well-calibrated z")
rng = np.random.default_rng(0)
for n in (9, 10, 11):
    z = rng.standard_normal(n)
    st = [(zi, 0.0, 1.0) for zi in z]
    print(
        f" n={n}: Python",
        python_decision(D, 100, 50, 2 * D, st),
        " MATLAB",
        matlab_decision(D, 100, 50, 2 * D, st),
    )

print("\n== Case D: swtest.m vs scipy.stats.shapiro at alpha=1e-6")
rng = np.random.default_rng(1)
# check transcription on platykurtic samples
dev = []
for _ in range(200):
    x = rng.uniform(size=20)
    h, p, W, br = M.swtest(x, ALPHA)
    if br == "SW":
        dev.append(abs(np.log(p) - np.log(shapiro(x).pvalue)))
print(" SW branch |log p_swtest - log p_scipy| max:", max(dev))
for df, n_list in ((1.5, (10, 20, 40)), (3, (10, 20, 40))):
    for n in n_list:
        dis = 0
        sf = 0
        N = 2000
        for _ in range(N):
            x = student_t.rvs(df, size=n, random_state=rng)
            h, p, W, br = M.swtest(x, ALPHA)
            hp = shapiro(x).pvalue < ALPHA
            sf += br == "SF"
            dis += h != hp
        print(
            f" t(df={df}), n={n}: SF branch {sf}/{N}, decisions differ {dis}/{N}"
        )
# one outlier among normals, typical of a mis-predicted poll point
for n in (10, 20, 40):
    dis = 0
    N = 2000
    mh = mp = 0
    for _ in range(N):
        x = rng.standard_normal(n)
        x[0] = 8.0
        h, p, W, br = M.swtest(x, ALPHA)
        hp = shapiro(x).pvalue < ALPHA
        mh += h
        mp += hp
        dis += h != hp
    print(
        f" N(0,1) + one z=8 outlier, n={n}: MATLAB rejects {mh}, scipy rejects {mp}, differ {dis}/{N}"
    )
