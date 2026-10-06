"""F9 (comparison): scipy.shapiro vs MATLAB swtest on the z-scores PyBADS actually tests (n >= 3)."""
import logging
import warnings

import common  # noqa
import numpy as np
from matlab_transcriptions import kurtosis_biased, swtest
from scipy.stats import shapiro

from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)
warnings.simplefilter("ignore")


class B(BADS):
    def _is_gp_refit_time_(self, alpha):
        st = self.gp_stats
        if st.get("iter_gp") is not None and len(st.get("iter_gp")) >= 3:
            fv = np.array(st.get("fval"), float)
            mu = np.array(st.get("ymu"), float)
            s = np.array(st.get("ys"), float)
            s[np.isclose(0.0, s)] = 1e-6
            z = (fv - mu) / s
            if not np.any(np.isnan(z)):
                hp = shapiro(z).pvalue < alpha
                hm = bool(swtest(z, alpha)[0])
                self._c.append((hp, hm, kurtosis_biased(z) > 3))
        return super()._is_gp_refit_time_(alpha)


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ell(x):
    x = np.ravel(x)
    D = x.size
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


for name, fun, D, seed, noisy in [
    ("rosen3", rosen, 3, 50, False),
    ("ell3", ell, 3, 50, False),
    ("noisy sphere3", None, 3, 50, True),
]:
    opts = {"random_seed": seed, "display": "off", "max_fun_evals": 200}
    if noisy:
        r_ = np.random.default_rng(seed + 7)
        fun = lambda x, r_=r_: float(
            np.sum(np.ravel(x) ** 2) + r_.standard_normal()
        )
        opts["uncertainty_handling"] = True
    b = B(
        fun,
        0.5 * np.ones((1, D)) + 0.1 * np.arange(D),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options=opts,
    )
    b._c = []
    b.optimize()
    c = b._c
    print(
        f"{name}: checks with n>=3: {len(c)}; leptokurtic (SF branch) {sum(x[2] for x in c)}; scipy rejects {sum(x[0] for x in c)}, swtest rejects {sum(x[1] for x in c)}; disagree {sum(x[0] != x[1] for x in c)}"
    )
