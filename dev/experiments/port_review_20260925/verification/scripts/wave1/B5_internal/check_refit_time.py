import types

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
from scipy.special import gammaincinv
from scipy.stats import chi2, shapiro

from pybads.bads.bads import BADS
from pybads.utils import IterationHistory

alpha = 1e-6
for n in [1, 2]:
    code_lo, code_hi = gammaincinv(n / 2, alpha / 2), gammaincinv(
        n / 2, 1 - alpha / 2
    )
    print(
        f"n={n}: code bounds [{code_lo:.3g}, {code_hi:.3g}]  chi2 bounds [{chi2.ppf(alpha/2, n):.3g}, {chi2.ppf(1-alpha/2, n):.3g}]"
        f"  P(reject | calibrated) code={chi2.cdf(code_lo, n) + chi2.sf(code_hi, n):.3g} nominal={alpha:.0e}"
    )


# Drive _is_gp_refit_time_ on a stub with 0, 1, 2, 3 recorded stats, all perfectly calibrated (z = 0.1)
def make(nstats, zs=None):
    s = types.SimpleNamespace()
    s.D = 3
    s.function_logger = types.SimpleNamespace(func_count=20)
    s.options = {
        "min_refit_time": 100
    }  # so that no refit happens: only do_gp_calibration is read
    s.optim_state = {"lastfitgp": 15}
    s.gp_stats = IterationHistory(["iter_gp", "fval", "ymu", "ys", "gp"])
    s._save_gp_stats_ = types.MethodType(BADS._save_gp_stats_, s)
    s._record_gp_refit_ = types.MethodType(BADS._record_gp_refit_, s)
    for i in range(nstats):
        z = 0.1 if zs is None else zs[i]
        s._save_gp_stats_(1.0 + z, 1.0, 1.0)
    return s


for n in [0, 1, 2, 3, 5]:
    s = make(n)
    print(
        f"{n} stats with |z| = 0.1: do_gp_calibration =",
        BADS._is_gp_refit_time_(s, alpha)[1],
    )
# Shapiro is scale-free: z-scores 1000 times too large, and all biased by +50 SDs
rng = np.random.default_rng(0)
z = rng.standard_normal(12)
for label, zz in [
    ("calibrated", z),
    ("SD x1000", 1000 * z),
    ("bias +50", z + 50),
]:
    s = make(len(zz), zz)
    print(
        f"12 stats, {label}: do_gp_calibration =",
        BADS._is_gp_refit_time_(s, alpha)[1],
        " shapiro p =",
        f"{shapiro(zz).pvalue:.3f}",
    )
# n = 2 with z = (4, 4): sum 32; chi2(2) sf = 1.1e-7 -> nominal reject; z=(3,3.2): sum 19.2, chi2 sf 6.8e-5 -> nominal accept
for zz in [(3.0, 3.2), (4.0, 4.0)]:
    s = make(2, zz)
    print(
        f"2 stats z={zz}: sum z^2={sum(v*v for v in zz):.1f}, chi2 p={chi2.sf(sum(v*v for v in zz), 2):.2g}, do_gp_calibration =",
        BADS._is_gp_refit_time_(s, alpha)[1],
    )
