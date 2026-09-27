import warnings

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, "gpyreg", gpyreg.__file__, flush=True)
from pybads.function_logger import FunctionLogger
from pybads.variable_transformer import VariableTransformer

D = 2
vt = VariableTransformer(
    D,
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    np.zeros((1, D)),
)


def trial(level, ret):
    fl = FunctionLogger(lambda x: ret, D, level > 1, level, 10, vt)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = fl(np.array([0.1, 0.2]))
        return f"accepted -> returns {out[0]!r}, {out[1]!r}; S row {fl.S[0] if fl.noise_flag else '-'}; warnings {[str(x.message)[:40] for x in w]}"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:110]!r}"


cases0 = [
    1.0,
    np.float32(1.0),
    np.array(1.0),
    np.array([1.0]),
    [1.0],
    np.array([[1.0]]),
    np.array([1.0, 2.0]),
    (1.0, 0.5),
    np.nan,
    np.inf,
    None,
    "1.0",
    1 + 0j,
    1 + 1j,
    True,
    np.array([]),
]
for r in cases0:
    print("level 0, returns", repr(r), "->", trial(0, r), flush=True)
cases2 = [
    (1.0, 0.5),
    (1.0, np.array(0.5)),
    (1.0, [0.5]),
    (1.0, np.array([0.5])),
    ([1.0], 0.5),
    (1.0, 0.0),
    (1.0, -1.0),
    (1.0, np.nan),
    (1.0, None),
    (1.0, [0.5, 0.6]),
    (1.0, "0.5"),
    [1.0, 0.5],
    np.array([1.0, 0.5]),
    1.0,
    (1.0, 0.5, 2),
    (1.0, True),
    (1.0, 1 + 0j),
]
for r in cases2:
    print("level 2, returns", repr(r), "->", trial(2, r), flush=True)


# a target that raises
def bad(x):
    raise RuntimeError("boom")


fl = FunctionLogger(bad, D, False, 0, 10, vt)
try:
    fl(np.array([0.1, 0.2]))
except Exception as e:
    print(
        "raising target ->",
        type(e).__name__,
        repr(str(e))[:200],
        "func_count",
        fl.func_count,
        "Xn",
        fl.Xn,
    )

# the point the target receives
seen = []
fl = FunctionLogger(
    lambda x: (seen.append((type(x).__name__, x.shape, x.copy())), 0.0)[1],
    D,
    False,
    0,
    10,
    vt,
)
fl(np.array([[0.5, -1.0]]))
print(
    "target receives",
    seen[0],
    "; inverse_transf of u:",
    vt.inverse_transf(np.array([[0.5, -1.0]])),
)
