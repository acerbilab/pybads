"""Which refit decisions differ between the port and MATLAB's rule, on the
same stats (one deterministic run)."""
import matlab_ref as M
import numpy as np
import s4_runs as R

from pybads.bads.bads import BADS

recs = []
orig = BADS._is_gp_refit_time_


def is_refit(self, alpha):
    st = self.gp_stats
    if st.get("iter_gp") is None:
        f = m = s = []
    else:
        f = [float(v) for v in st.get("fval")]
        m = [float(v) for v in st.get("ymu")]
        s = [float(v) for v in st.get("ys")]
    n = len(f)
    fc = self.function_logger.func_count
    last = self.optim_state["lastfitgp"]
    mrt = self.options["min_refit_time"]
    try:
        u = M.gppredcheck(f, m, s, alpha)
    except Exception:
        u = True
    r = M.is_refit_time(last, fc, n, u, self.D, mrt)
    rp, up = orig(self, alpha)
    recs.append((n, fc, fc - last, bool(rp), r, bool(up), (False if r else u)))
    return rp, up


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
BADS._is_gp_refit_time_ = orig
print("D=3: min_refit_time=6, refit period=10")
print("n  fc  fc-lastfit  py_refit mat_refit  py_unrel mat_unrel")
from collections import Counter

c = Counter()
for rec in recs:
    if rec[3] != rec[4] or rec[5] != rec[6]:
        print(rec)
    if rec[3] != rec[4]:
        c[(rec[0], rec[3], rec[4])] += 1
print("refit disagreements by (n, py, mat):", dict(c))
print(
    "refits py:",
    sum(r[3] for r in recs),
    " refits MATLAB rule:",
    sum(r[4] for r in recs),
    "calls",
    len(recs),
)
