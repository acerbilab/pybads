"""What FunctionLogger does with malformed target outputs (the changelog's
"Malformed target outputs" entry and its "Upgrading from 1.1.0" line), at
whichever pybads PYTHONPATH selects."""

import warnings

import gpyreg
import numpy as np

import pybads
from pybads.function_logger import FunctionLogger

print(pybads.__file__, gpyreg.__file__, flush=True)

D = 2


def run(value, sd=None, level=0):
    if level == 2:
        fun = lambda x: (value, sd)
    else:
        fun = lambda x: value
    fl = FunctionLogger(fun, D, level == 2, level)
    x = np.array([0.1, 0.2])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fl(x)
            res = f"OK -> fval={out[0]!r} sd={out[1]!r}"
        except Exception as e:
            msg = " ".join(str(a) for a in e.args).replace("\n", " ")
            msg = " ".join(msg.split())[:110]
            res = f"{type(e).__name__}: {msg}"
    wn = [str(x.category.__name__) for x in w]
    return (
        f"{res} | Xn={fl.Xn} func_count={fl.func_count} X0={fl.X[0]} warn={wn}"
    )


values = [
    ("np.complex128(1+0j)", np.complex128(1 + 0j)),
    ("np.complex128(1+1j)", np.complex128(1 + 1j)),
    ("complex(1,0)", complex(1, 0)),
    ("complex(1,1)", complex(1, 1)),
    ("[1.0]", [1.0]),
    ("np.array([1.0])", np.array([1.0])),
    ("np.array([1.0, 2.0])", np.array([1.0, 2.0])),
    ("[1.0, 2.0]", [1.0, 2.0]),
    ("'1.0'", "1.0"),
    ("None", None),
    ("True", True),
    ("np.float32(1)", np.float32(1)),
    ("np.int64(1)", np.int64(1)),
    ("nan", np.nan),
]
print("-- value, level 0")
for name, v in values:
    print(f"{name:24s} {run(v)}")

sds = [
    ("None", None),
    ("[0.5]", [0.5]),
    ("np.array([0.5])", np.array([0.5])),
    ("np.array([0.5, 0.6])", np.array([0.5, 0.6])),
    ("complex(0.5,0)", complex(0.5, 0)),
    ("np.complex128(0.5)", np.complex128(0.5)),
    ("0", 0),
    ("-1", -1),
    ("inf", np.inf),
    ("'0.5'", "0.5"),
    ("0.5", 0.5),
]
print("-- SD, level 2, value 1.0")
for name, s in sds:
    print(f"{name:24s} {run(1.0, s, level=2)}")
