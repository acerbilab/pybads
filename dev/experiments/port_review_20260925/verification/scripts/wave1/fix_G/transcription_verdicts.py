"""Verdicts (refit, unreliable) of the verifier's transcription of MATLAB's
IsRefitTime and gppredcheck.m on the cases of the tests, at D = 2."""
import numpy as np
from matlab_transcriptions import is_refit_time
from scipy.stats import chi2, norm


def q(n):
    return list(norm.ppf((np.arange(1, n + 1) - 0.5) / n))


D, mrt, alpha = 2, 4, 1e-6
cases = {
    "n=1 z=0.1 (30,29)": ([0.1], 30, 29),
    "n=1 z=0.1 (30,20)": ([0.1], 30, 20),
    "n=9 quantiles (60,30)": (q(9), 60, 30),
    "n=10 quantiles (60,30)": (q(10), 60, 30),
    "n=1 z=4 (30,29)": ([4.0], 30, 29),
    "n=1 z=5.5 (30,29)": ([5.5], 30, 29),
    "n=2 z=(3,3.2) (30,29)": ([3.0, 3.2], 30, 29),
    "n=2 z=(3,3.2) (40,20)": ([3.0, 3.2], 40, 20),
    "n=2 z=(4,4) (30,29)": ([4.0, 4.0], 30, 29),
    "n=2 z=(1e-4,1e-4) (30,29)": ([1e-4, 1e-4], 30, 29),
    "n=1 z=1e-7 (30,29)": ([1e-7], 30, 29),
}
for name, (z, fc, lf) in cases.items():
    print(
        f"{name:32s}",
        is_refit_time(fc, D, lf, mrt, z, [0] * len(z), [1] * len(z), alpha),
    )
for v in (1, 2):
    print(v, chi2.ppf(alpha / 2, v), chi2.ppf(1 - alpha / 2, v))
