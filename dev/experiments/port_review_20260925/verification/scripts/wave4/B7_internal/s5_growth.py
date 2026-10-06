import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)
from pybads import BADS
from pybads.function_logger import FunctionLogger
from pybads.variable_transformer import VariableTransformer


def make_target(level, seed):
    r = np.random.default_rng(seed)

    def f(x):
        x = np.asarray(x)
        v = float(np.sum((x - 0.3) ** 2 * np.array([1.0, 4.0])))
        if level == 0:
            return v
        sd = 0.3 + 0.1 * abs(x[0])
        y = v + sd * r.standard_normal()
        return (y, sd) if level == 2 else y

    return f


D = 2
lb = -5 * np.ones(D)
ub = 5 * np.ones(D)
plb = -2 * np.ones(D)
pub = 2 * np.ones(D)
for level, extra in [(0, {}), (2, {"specify_target_noise": True})]:
    res = {}
    for cs in [500, 3]:
        opts = {
            "random_seed": 3,
            "display": "off",
            "max_fun_evals": 80,
            "cache_size": cs,
        }
        opts.update(extra)
        b = BADS(
            make_target(level, 7),
            np.array([1.2, -0.7]),
            lb,
            ub,
            plb,
            pub,
            options=opts,
        )
        r = b.optimize()
        fl = b.function_logger
        lens = {
            k: getattr(fl, k).shape[0]
            for k in [
                "X",
                "Y",
                "X_orig",
                "Y_orig",
                "X_flag",
                "n_evals",
                "fun_eval_time",
            ]
            + (["S"] if fl.noise_flag else [])
        }
        res[cs] = (
            r["x"],
            r["fval"],
            r["func_count"],
            fl.X[: fl.Xn + 1].copy(),
        )
        print(
            f"level {level} cache_size {cs}: x {r['x']}, fval {r['fval']:.6g}, func_count {r['func_count']}, rows {fl.Xn+1}, array lengths {lens}",
            flush=True,
        )
    print(
        f"level {level}: same result for cache_size 500 and 3: {np.array_equal(res[500][0], res[3][0]) and res[500][1] == res[3][1] and np.array_equal(res[500][3], res[3][3])}",
        flush=True,
    )

# finalize, then the reads of n_evals by X_flag
vt = VariableTransformer(
    D,
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    np.zeros((1, D)),
)
fl = FunctionLogger(lambda x: float(np.sum(x**2)), D, False, 0, 10, vt)
for i in range(3):
    fl(np.array([0.1 * i, 0.2]))
fl.finalize()
print(
    "after finalize: X",
    fl.X.shape,
    "X_flag",
    fl.X_flag.shape,
    "n_evals",
    fl.n_evals.shape,
)
try:
    print(np.sum(fl.n_evals[fl.X_flag]))
except Exception as e:
    print("n_evals[X_flag] after finalize ->", type(e).__name__, str(e)[:100])
try:
    fl(np.array([0.9, 0.9]))
    print(
        "call after finalize: rows",
        fl.Xn + 1,
        "X",
        fl.X.shape,
        "n_evals",
        fl.n_evals.shape,
        "X_flag",
        fl.X_flag.shape,
    )
except Exception as e:
    print("call after finalize ->", type(e).__name__, str(e)[:100])

# add: x is taken in the transformed space
fl = FunctionLogger(lambda x: 0.0, D, True, 2, 10, vt)
out = fl.add(np.array([0.5, -1.0]), 3.0)
print(
    "add returns",
    out,
    "X_orig row",
    fl.X_orig[0],
    "X row",
    fl.X[0],
    "S row",
    fl.S[0],
    "func_count",
    fl.func_count,
    "cache_count",
    fl.cache_count,
)
try:
    fl.add(np.array([0.5, -1.0]), np.array([3.0]))
except Exception as e:
    print(
        "add with a one-element array value ->", type(e).__name__, str(e)[:80]
    )
