import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)
from pybads.function_logger import FunctionLogger

fl = FunctionLogger(lambda x: np.complex128(2.0), 2, False, 0, 10, None)
try:
    fl(np.array([0.1, 0.2]))
except Exception as e:
    print(type(e).__name__, e)
print(
    "Xn",
    fl.Xn,
    "X_max_idx",
    fl.X_max_idx,
    "X row",
    fl.X[0],
    "Y row",
    fl.Y[0],
    "X_flag",
    fl.X_flag[0],
    "func_count",
    fl.func_count,
)
# n_eff offset at the first GP fit, D = 2 defaults
for D in [2, 6]:
    a = -(128 - 8)
    b = -3 * a
    c = 3 * a
    d = 128
    f = lambda x: a * x**3 + b * x**2 + c * x + d
    n_design = 2 ** int(np.ceil(np.log2(D))) * (
        2 if 2 ** int(np.ceil(np.log2(D))) == D else 1
    )
    eff = 1 + n_design
    nb = min(500 * D, 50 + 10 * D) - eff
    for extra in [0, 10, 30]:
        x_true = extra / nb
        x_code = (extra + 1) / nb
        print(
            f"D={D}: evaluations after the design {extra}: init_N {max(round(f(x_true)), 8)} without the noise test in n_eff, {max(round(f(x_code)), 8)} with it"
        )
