import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)
from pybads.function_logger import FunctionLogger

x = np.array([0.1, 0.2])
cases = [
    ("level 0, returns (f, sd) tuple", 0, lambda x: (1.0, 0.5)),
    ("level 0, returns [f]", 0, lambda x: [1.0]),
    ("level 0, returns 2-element array", 0, lambda x: np.array([1.0, 2.0])),
    ("level 0, returns None", 0, lambda x: None),
    ("level 0, returns nan", 0, lambda x: np.nan),
    ("level 0, returns 1+0j", 0, lambda x: 1 + 0j),
    ("level 0, target raises KeyError", 0, lambda x: {}["a"]),
    ("level 2, returns f only", 2, lambda x: 1.0),
    ("level 2, returns [f, sd] list", 2, lambda x: [1.0, 0.5]),
    ("level 2, returns (f, None)", 2, lambda x: (1.0, None)),
    ("level 2, returns (f, [0.5])", 2, lambda x: (1.0, [0.5])),
    ("level 2, returns (f, [0.5, 0.6])", 2, lambda x: (1.0, [0.5, 0.6])),
    (
        "level 2, returns (f, array([0.5, 0.6]))",
        2,
        lambda x: (1.0, np.array([0.5, 0.6])),
    ),
    ("level 2, returns (f, 0.0)", 2, lambda x: (1.0, 0.0)),
    ("level 2, returns (f, True)", 2, lambda x: (1.0, True)),
]
for name, level, fun in cases:
    fl = FunctionLogger(fun, 2, level == 2, level)
    try:
        out = fl(x)
        print(
            f"{name}: accepted -> {out[:2]}, func_count {fl.func_count}",
            flush=True,
        )
    except Exception as e:
        print(
            f"{name}: {type(e).__name__}: {str(e)[:110]!r}, func_count {fl.func_count}",
            flush=True,
        )
