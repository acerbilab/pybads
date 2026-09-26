import time

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
rng = np.random.default_rng(0)
calls = {"n": 0, "t": 0.0}


def fun(x):
    t0 = time.perf_counter()
    time.sleep(0.05)
    v = float(np.sum(np.ravel(x) ** 2) + 0.3 * rng.standard_normal())
    calls["n"] += 1
    calls["t"] += time.perf_counter() - t0
    return v


D = 2
b = BADS(
    fun,
    np.full((1, D), 0.7),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    options=dict(
        uncertainty_handling=True,
        random_seed=0,
        display="off",
        max_fun_evals=60,
        noise_final_samples=10,
    ),
)
r = b.optimize()
print("func_count", r["func_count"], "target calls", calls["n"])
print(
    "time in target (measured)",
    round(calls["t"], 2),
    "s; logger total_fun_eval_time",
    round(b.function_logger.total_fun_eval_time, 2),
    "s",
)
print(
    "total_time",
    round(r["total_time"], 2),
    "overhead reported",
    round(r["overhead"], 3),
    "overhead from measured target time",
    round(r["total_time"] / calls["t"] - 1, 3),
)
