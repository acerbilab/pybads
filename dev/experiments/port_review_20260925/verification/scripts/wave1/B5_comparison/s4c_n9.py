"""The refits at n=9 that MATLAB's rule does not make: shapiro vs swtest."""
import matlab_ref as M
import numpy as np
import s4_runs as R
from scipy.stats import shapiro

from pybads.bads.bads import BADS

orig = BADS._is_gp_refit_time_
out = []


def is_refit(self, alpha):
    st = self.gp_stats
    if st.get("iter_gp") is not None:
        f = np.array([float(v) for v in st.get("fval")])
        m = np.array([float(v) for v in st.get("ymu")])
        s = np.array([float(v) for v in st.get("ys")])
        if len(f) >= 3:
            s2 = s.copy()
            s2[np.isclose(0.0, s2)] = 1e-6
            z_py = (f - m) / s2
            z_m = (f - m) / s
            h, p, W, br = (
                M.swtest(z_m, alpha)
                if np.all(np.isfinite(z_m))
                else (None, np.nan, np.nan, "nonfinite")
            )
            p_py = shapiro(z_py).pvalue
            if (p_py < alpha) != bool(h):
                out.append((len(f), p_py, p, br, s.min(), np.round(z_m, 2)))
    return orig(self, alpha)


BADS._is_gp_refit_time_ = is_refit
D = 3
opts = {"display": "off", "random_seed": 0, "max_fun_evals": 200}
b = BADS(
    R.rosen,
    np.full((1, D), -1.5),
    np.full((1, D), -5.0),
    np.full((1, D), 5.0),
    np.full((1, D), -3.0),
    np.full((1, D), 3.0),
    options=opts,
)
b.optimize()
for o in out:
    print("n=%d scipy p=%.3g swtest p=%.3g branch=%s min(fs)=%.3g z=%s" % o)
