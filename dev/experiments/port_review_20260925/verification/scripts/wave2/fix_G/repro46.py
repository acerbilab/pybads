import warnings

import numpy as np

import pybads
from pybads.variable_transformer import VariableTransformer

print(pybads.__file__)

cases = {
    "mixed, linear ub inf": (
        [[1.0, -5.0]],
        [[1000.0, np.inf]],
        [[2.0, -1.0]],
        [[500.0, 1.0]],
    ),
    "mixed, linear lb -inf": (
        [[1.0, -np.inf]],
        [[1000.0, 5.0]],
        [[2.0, -1.0]],
        [[500.0, 1.0]],
    ),
    "mixed, linear ub 1e3": (
        [[1.0, -1e3]],
        [[1000.0, 1e3]],
        [[2.0, -1.0]],
        [[500.0, 1.0]],
    ),
    "all log, ub inf": (
        [[1.0, 1.0]],
        [[np.inf, np.inf]],
        [[2.0, 2.0]],
        [[500.0, 500.0]],
    ),
    "all log, ub 1e300": (
        [[1.0, 1.0]],
        [[1e300, 1e300]],
        [[2.0, 2.0]],
        [[500.0, 500.0]],
    ),
    "all log, lb tiny": (
        [[1e-300, 1.0]],
        [[1e3, 1e3]],
        [[2.0, 2.0]],
        [[500.0, 500.0]],
    ),
}
for name, (lb, ub, plb, pub) in cases.items():
    lb, ub, plb, pub = map(np.array, (lb, ub, plb, pub))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            vt = VariableTransformer(2, lb, ub, plb, pub, None)
            print(
                name,
                vt.apply_log_t,
                vt.lb,
                vt.ub,
                vt.plb,
                vt.pub,
                [str(x.message) for x in w],
            )
        except Exception as e:
            print(name, "raised", e, [str(x.message) for x in w])
