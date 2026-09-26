import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
from pybads import BADS

f = lambda x: float(
    np.sum((np.atleast_1d(x) - 0.3) ** 2) + 0.1 * np.sin(20 * np.sum(x))
)
b = BADS(
    f,
    np.array([[1.5]]),
    np.array([[-10.0]]),
    np.array([[10.0]]),
    np.array([[-3.0]]),
    np.array([[3.0]]),
    options={"random_seed": 0, "display": "off", "max_fun_evals": 100},
)
r = b.optimize()
gp = b.iteration_history["gp"][b.optim_state["iter"]]
print(
    "D=1: fitted length scale",
    np.exp(gp.get_hyperparameters()[0]["covariance_log_lengthscale"]),
    "temporary_data len_scale",
    gp.temporary_data["len_scale"],
    "ntrain",
    b.optim_state["ntrain"],
    "func_count",
    r["func_count"],
)
