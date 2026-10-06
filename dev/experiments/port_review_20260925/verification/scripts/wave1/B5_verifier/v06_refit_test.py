"""F5, F6 (internal) / F7, F8, F9 (comparison): _is_gp_refit_time_ against a transcription of IsRefitTime/gppredcheck/swtest."""
import common  # noqa
import numpy as np
from matlab_transcriptions import gppredcheck, is_refit_time, swtest
from scipy.special import gammaincinv
from scipy.stats import chi2, shapiro
from scipy.stats import t as tdist

from pybads import BADS
from pybads.utils import IterationHistory

alpha = 1e-6
print(
    "chi2 bounds, v=1: python",
    [gammaincinv(0.5, alpha / 2), gammaincinv(0.5, 1 - alpha / 2)],
    " true",
    [chi2.ppf(alpha / 2, 1), chi2.ppf(1 - alpha / 2, 1)],
)
print(
    "chi2 bounds, v=2: python",
    [gammaincinv(1, alpha / 2), gammaincinv(1, 1 - alpha / 2)],
    " true",
    [chi2.ppf(alpha / 2, 2), chi2.ppf(1 - alpha / 2, 2)],
)

D = 2
b = BADS(
    lambda x: float(np.sum(np.ravel(x) ** 2)),
    np.array([[1.0, 1.0]]),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    options={"random_seed": 0, "display": "off"},
)
mrt = b.options["min_refit_time"]


def py_decide(fv, mu, s, funccount, lastfit):
    b.gp_stats = IterationHistory(["iter_gp", "fval", "ymu", "ys", "gp"])
    for a, m, ss in zip(fv, mu, s):
        b._save_gp_stats_(a, m, ss)
    b.function_logger.func_count = funccount
    b.optim_state["lastfitgp"] = lastfit
    r, u = b._is_gp_refit_time_(alpha)
    return bool(r), bool(u)


rng = np.random.default_rng(0)
cases = {
    "n=1, z=0.1 (min_refit_time not passed)": ([0.1], [0.0], [1.0], 30, 29),
    "n=1, z=0.1 (min_refit_time passed)": ([0.1], [0.0], [1.0], 30, 20),
    "n=2, z=(3,3.2)": ([3.0, 3.2], [0, 0], [1, 1], 30, 29),
    "n=2, z=(3,3.2), refit allowed": ([3.0, 3.2], [0, 0], [1, 1], 40, 20),
    "n=10 calibrated, D=2 (period 10)": (
        list(rng.standard_normal(10)),
        [0] * 10,
        [1] * 10,
        60,
        30,
    ),
    "n=11 calibrated, D=2": (
        list(rng.standard_normal(11)),
        [0] * 11,
        [1] * 11,
        60,
        30,
    ),
    "n=5 with one exact-zero SD": (
        [0.3, -0.2, 0.5, 0.01, 0.1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0.0, 1],
        60,
        30,
    ),
}
for name, (fv, mu, s, fc, lf) in cases.items():
    try:
        m = is_refit_time(fc, D, lf, mrt, fv, mu, s, alpha)
    except Exception as e:
        m = f"raised {type(e).__name__}"
    print(
        f"{name:40s} python (refit, unrel)={py_decide(fv, mu, s, fc, lf)}  MATLAB={m}"
    )

# F9: SW vs SF on heavy-tailed samples
r2 = np.random.default_rng(1)
for n in (10, 20, 40):
    dis = rej_m = rej_p = 0
    for _ in range(2000):
        z = tdist.rvs(1.5, size=n, random_state=r2)
        hm, pm = swtest(z, alpha)
        hp = int(shapiro(z).pvalue < alpha)
        rej_m += hm
        rej_p += hp
        dis += hm != hp
    print(
        f"t(1.5), n={n}: MATLAB swtest rejects {rej_m}/2000, scipy rejects {rej_p}/2000, disagree {dis}"
    )
for n in (20, 40):
    dis = rej_m = rej_p = 0
    for _ in range(2000):
        z = r2.standard_normal(n)
        z[0] = 8.0
        hm, pm = swtest(z, alpha)
        hp = int(shapiro(z).pvalue < alpha)
        rej_m += hm
        rej_p += hp
        dis += hm != hp
    print(
        f"N(0,1)+outlier 8, n={n}: MATLAB rejects {rej_m}/2000, scipy rejects {rej_p}/2000, disagree {dis}"
    )
# platykurtic sanity: transcription SW branch vs scipy p-values
d = []
for _ in range(500):
    z = r2.uniform(-1, 1, size=int(r2.integers(4, 60)))
    if swtest(z, alpha)[1] > 0 and shapiro(z).pvalue > 0:
        from matlab_transcriptions import kurtosis_biased

        if kurtosis_biased(z) <= 3:
            d.append(
                abs(np.log(swtest(z, alpha)[1]) - np.log(shapiro(z).pvalue))
            )
print(
    "platykurtic: max |log p_transcription - log p_scipy| =",
    max(d),
    "over",
    len(d),
    "samples",
)
