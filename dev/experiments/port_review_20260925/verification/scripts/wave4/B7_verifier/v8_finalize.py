"""B7 verifier, F9: finalize after the arrays grew past the filled rows."""
import gpyreg
import numpy as np

import pybads
from pybads.function_logger import FunctionLogger

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
lg = FunctionLogger(lambda x: float(np.sum(x**2)), 2, False, 0, cache_size=3)
for i in range(4):
    lg(np.array([0.1 * i, 0.0]))
print(f"4 rows filled: X {lg.X.shape}, n_evals {lg.n_evals.shape}", flush=True)
lg.finalize()
print(
    f"after finalize: X {lg.X.shape}, X_flag {lg.X_flag.shape}, n_evals "
    f"{lg.n_evals.shape}, fun_eval_time {lg.fun_eval_time.shape}",
    flush=True,
)
try:
    lg.n_evals[lg.X_flag]
    print("n_evals[X_flag] works", flush=True)
except IndexError as e:
    print(f"n_evals[X_flag] raises IndexError: {e}", flush=True)
lg(np.array([0.9, 0.9]))
print(
    f"one more call: X {lg.X.shape}, n_evals {lg.n_evals.shape}, X_flag "
    f"{lg.X_flag.shape}",
    flush=True,
)
lg2 = FunctionLogger(
    lambda x: float(np.sum(x**2)), 2, False, 0, cache_size=3
)
for i in range(6):
    lg2(np.array([0.1 * i, 0.0]))
lg2.reset_fun_eval_time()
print(
    f"reset_fun_eval_time after growth: fun_eval_time {lg2.fun_eval_time.shape}"
    f" vs X {lg2.X.shape}",
    flush=True,
)
