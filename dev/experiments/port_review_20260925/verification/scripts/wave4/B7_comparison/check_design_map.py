import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
from pybads import BADS

seen = []


def f(x):
    seen.append(np.array(x, copy=True))
    return float(np.sum(np.log(x) ** 2))


D = 2
lb = np.array([1e-3, 1e-3])
ub = np.array([1e3, 1e3])
plb = np.array([0.1, 1.0])
pub = np.array([10.0, 2.0])
bads = BADS(
    f,
    np.array([1.0, 1.5]),
    lb,
    ub,
    plb,
    pub,
    options={"random_seed": 3, "max_fun_evals": 8, "display": "off"},
)
print(
    "transformed plb/pub:",
    bads.plausible_lower_bounds,
    bads.plausible_upper_bounds,
    "log flags:",
    bads.var_transf.apply_log_t,
    flush=True,
)
bads.optimize()
X = np.array(seen)
print("shapes the target received:", {x.shape for x in seen}, flush=True)
print("points (original space):\n", np.round(X, 4), flush=True)
fl = bads.function_logger
print("rows", fl.Xn + 1, "func_count", fl.func_count, flush=True)
print("u rows:\n", np.round(fl.X[: fl.Xn + 1], 4), flush=True)
