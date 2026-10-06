"""The start's count of evaluations in the log after the noise test (the
changelog's "Noise test in the schedule of the GP's fits"), and the
fraction of the budget that the first fit schedule reads."""
import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__, gpyreg.__file__, flush=True)
b = BADS(
    lambda x: float(np.sum(np.ravel(x) ** 2)),
    np.array([1.0, 1.5]),
    -5 * np.ones(2),
    5 * np.ones(2),
    -3 * np.ones(2),
    3 * np.ones(2),
    options={"display": "off", "random_seed": 0, "max_fun_evals": 20},
)
r = b.optimize()
fl = b.function_logger
print(
    "func_count",
    r["func_count"],
    "n_evals of the start's row",
    fl.n_evals[0].item(),
    "sum n_evals",
    float(np.sum(fl.n_evals[fl.X_flag])),
    "logged points",
    int(np.sum(fl.X_flag)),
    "n_noise_test",
    b.optim_state.get("n_noise_test"),
)
