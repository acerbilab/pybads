"""Run BADS on a 10-D sphere, then inspect the stored GPs (mean prior/bounds, normalization)."""
import sys
import warnings

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

D = int(sys.argv[1]) if len(sys.argv) > 1 else 10
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
fun = lambda x: float(np.sum(np.asarray(x) ** 2))
x0 = np.full(D, 0.5)
lb = np.full(D, -5.0)
ub = np.full(D, 5.0)
plb = np.full(D, -1.0)
pub = np.full(D, 1.0)
warnings.simplefilter("always")
with warnings.catch_warnings(record=True) as W:
    bads = BADS(
        fun,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={"random_seed": seed, "max_fun_evals": 200, "display": "off"},
    )
    res = bads.optimize()
msgs = {}
for w in W:
    k = str(w.message)[:80]
    msgs[k] = msgs.get(k, 0) + 1
print("fval", res["fval"], "func_count", res["func_count"])
print("warnings:", msgs)
gps = bads.iteration_history["gp"]
print("n stored gps", len(gps))
for it, gp in enumerate(gps):
    if gp is None:
        continue
    pri = gp.get_priors()["mean_const"]
    b = gp.get_bounds()["mean_const"]
    hyp = gp.get_hyperparameters()[0]
    nc = gp.normalization_constants
    print(
        f"it {it:2d} ymin {np.min(gp.y):.3g} p90 {np.percentile(gp.y,90,method='hazen'):.3g} "
        f"prior mu {pri[1][0][0]:.3g} sd {pri[1][1][0]:.3g} bounds [{b[0][0]:.3g},{b[1][0]:.3g}] "
        f"mean_hyp {hyp['mean_const'][0]:.4g} norm_const(mean) {nc[-1]:.3g} "
        f"noise {hyp['noise_log_scale'][0]:.3g} logsf {hyp['covariance_log_outputscale'][0]:.3g}"
    )
