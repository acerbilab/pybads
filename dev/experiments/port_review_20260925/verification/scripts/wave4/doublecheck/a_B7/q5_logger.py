"""The logger's checks at the package on PYTHONPATH: well-formed outputs
accepted (as 1.1.0 did), malformed ones refused with ValueError before any
state changes; finalize and reset_fun_eval_time keep the arrays equal."""
import copy

import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
from pybads.function_logger import FunctionLogger

STATE = [
    "Xn",
    "X_max_idx",
    "func_count",
    "cache_count",
    "X",
    "Y",
    "X_orig",
    "Y_orig",
    "n_evals",
    "fun_eval_time",
    "total_fun_eval_time",
    "X_flag",
    "Y_max",
]


def snap(fl):
    s = {k: copy.deepcopy(getattr(fl, k)) for k in STATE}
    if hasattr(fl, "S"):
        s["S"] = fl.S.copy()
    return s


def same(a, b):
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray):
            if not np.array_equal(x, y, equal_nan=True):
                return k
        elif not (
            x == y or (isinstance(x, float) and np.isnan(x) and np.isnan(y))
        ):
            return k
    return None


good0 = [
    1.0,
    np.float64(1.0),
    np.float32(1.0),
    np.int64(3),
    3,
    True,
    np.bool_(True),
    np.array([1.0]),
    np.array(1.0),
    np.array([[1.0]]),
    [1.0],
    (1.0,),
    np.array([3]),
]
print("level 0, well-formed values:")
for v in good0:
    fl = FunctionLogger(lambda x, v=v: v, 2, False, 0)
    try:
        f, s, i = fl(np.zeros(2))
        print(
            f"   {v!r:28} -> accepted fval={f!r} type={type(f).__name__} Y={fl.Y[0,0]}"
        )
    except Exception as e:
        print(f"   {v!r:28} -> {type(e).__name__}: {str(e)[:80]}")

good2 = [
    0.5,
    np.float64(0.5),
    np.float32(0.5),
    1,
    np.array(0.5),
    np.array([0.5]),
    [0.5],
    np.array([[0.5]]),
]
print("level 2, well-formed SDs (value 1.0):")
for sd in good2:
    fl = FunctionLogger(lambda x, sd=sd: (1.0, sd), 2, True, 2)
    try:
        f, s, i = fl(np.zeros(2))
        print(f"   {sd!r:28} -> accepted fsd={s!r} S={fl.S[0,0]}")
    except Exception as e:
        print(f"   {sd!r:28} -> {type(e).__name__}: {str(e)[:80]}")

bad0 = [
    np.array([1.0, 2.0]),
    "1.0",
    None,
    1 + 0j,
    np.complex128(1),
    np.nan,
    np.inf,
    -np.inf,
    (1.0, 0.5),
    [],
    np.array([]),
    {},
    object(),
    [[1.0, 2.0]],
    [1, [2, 3]],
    10**400 if False else 10**30,
]
print("level 0, malformed values: refused before any state change?")
for v in bad0:
    outs = iter([1.0, v])
    fl = FunctionLogger(lambda x: next(outs), 2, False, 0)
    fl(np.zeros(2))
    before = snap(fl)
    try:
        fl(np.ones(2))
        print(f"   {v!r:28} -> ACCEPTED")
    except Exception as e:
        ch = same(before, snap(fl))
        note = "FuncError" in str(e)
        print(
            f"   {str(v)[:26]!s:28} -> {type(e).__name__}{' (note)' if note else ''}; state changed: {ch}"
        )
bad2 = [
    None,
    "0.5",
    [0.5, 0.6],
    np.array([0.5, 0.6]),
    1 + 0j,
    np.complex128(0.5),
    0.0,
    -1.0,
    np.nan,
    np.inf,
    [],
    True,
]
print("level 2, malformed SDs:")
for sd in bad2:
    outs = iter([(1.0, 0.5), (1.0, sd)])
    fl = FunctionLogger(lambda x: next(outs), 2, True, 2)
    fl(np.zeros(2))
    before = snap(fl)
    try:
        fl(np.ones(2))
        print(f"   {sd!r:28} -> ACCEPTED fsd, S={fl.S[1,0]}")
    except Exception as e:
        ch = same(before, snap(fl))
        print(
            f"   {str(sd)[:26]!s:28} -> {type(e).__name__}; state changed: {ch}"
        )

# finalize / reset_fun_eval_time lengths
fl = FunctionLogger(lambda x: float(np.sum(x)), 3, False, 0, cache_size=3)
for i in range(7):
    fl(np.ones(3) * i)
lens = lambda fl: {
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
}
print("grown:", lens(fl))
fl.finalize()
print("finalized:", lens(fl))
fl(np.ones(3) * 9)
print(
    "call after finalize:",
    lens(fl),
    "n_evals over flag",
    fl.n_evals[fl.X_flag].ravel(),
)
fl.reset_fun_eval_time()
print("reset:", lens(fl))
fl2 = FunctionLogger(
    lambda x: (float(np.sum(x)), 0.1), 3, True, 2, cache_size=2
)
for i in range(5):
    fl2(np.ones(3) * i)
fl2.finalize()
print("level 2 finalized S", fl2.S.shape, lens(fl2))
fl3 = FunctionLogger(lambda x: 1.0, 2, False, 0, cache_size=4)
fl3.finalize()
print("empty finalized:", lens(fl3))
fl3(np.zeros(2))
print("then call:", lens(fl3))
