import logging
import time

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)


def sph(x):
    return float(np.sum(np.ravel(x) ** 2))


b = BADS(
    sph,
    x0=np.array([1.0, 1.0]),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-2 * np.ones(2),
    plausible_upper_bounds=2 * np.ones(2),
    options={"random_seed": 0, "display": "off", "max_fun_evals": 60},
)
r = b.optimize()
print("keys:", sorted(r.keys()))
print("missing from whitelist:", sorted(set(r._keys) - set(r.keys())))
try:
    print(r["status"])
except Exception as e:
    print("r['status'] ->", type(e).__name__, e)
try:
    print(r.status)
except Exception as e:
    print("r.status ->", type(e).__name__, e)
print("success:", r["success"], "message:", r["message"])
print(
    "iterations:",
    r["iterations"],
    "optim_state iter:",
    b.optim_state["iter"],
    "func_count:",
    r["func_count"],
    "logger Xn+1:",
    b.function_logger.Xn + 1,
)
print(
    "x:",
    r["x"],
    "fval:",
    r["fval"],
    "fsd:",
    r["fsd"],
    "mesh_size:",
    r["mesh_size"],
    "problem_type:",
    r["problem_type"],
    "target_type:",
    r["target_type"],
)
print(
    "x0:",
    r["x0"],
    " first evaluated point:",
    b.function_logger.X_orig[0],
    " yval_vec:",
    r["yval_vec"],
    "ysd_vec:",
    r["ysd_vec"],
)
print("f(x)==fval:", sph(r["x"]) == r["fval"])
# noisy run: overhead accounting of the final samples
nrng = np.random.default_rng(1000)


def noisy(x):
    time.sleep(0.002)
    return float(np.sum(np.ravel(x) ** 2) + nrng.normal())


b = BADS(
    noisy,
    x0=np.array([1.0, 1.0]),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-2 * np.ones(2),
    plausible_upper_bounds=2 * np.ones(2),
    options={
        "random_seed": 0,
        "display": "off",
        "max_fun_evals": 80,
        "uncertainty_handling": True,
    },
)
r = b.optimize()
fl = b.function_logger
print(
    "noisy: func_count",
    r["func_count"],
    "rows",
    fl.Xn + 1,
    "n_evals sum",
    fl.n_evals[: fl.Xn + 1].sum(),
    "total_fun_eval_time",
    round(fl.total_fun_eval_time, 4),
    "sum row times",
    round(
        np.nansum(fl.fun_eval_time[: fl.Xn + 1] * fl.n_evals[: fl.Xn + 1]), 4
    ),
)
print(
    "noisy: yval_vec size",
    np.size(r["yval_vec"]),
    "fsd",
    r["fsd"],
    "iterations",
    r["iterations"],
    "status/exit:",
    r.get("status"),
    r["message"],
)
