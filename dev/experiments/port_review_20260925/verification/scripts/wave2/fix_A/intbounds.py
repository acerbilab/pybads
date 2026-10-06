import numpy as np

import pybads
from pybads import BADS
from pybads.variable_transformer import VariableTransformer

print(pybads.__file__)
for dt in (int, float):
    a = lambda v: np.array([[v]], dtype=dt)
    try:
        t = VariableTransformer(
            1, a(1), a(1000), a(2), a(500), np.full((1, 1), np.nan)
        )
        print(
            dt.__name__,
            "log",
            t.apply_log_t,
            "plb/pub u",
            t.plb,
            t.pub,
            "lb/ub u",
            t.lb,
            t.ub,
        )
    except ValueError as e:
        print(dt.__name__, "ValueError", str(e)[:60])
for dt in (int, float):
    try:
        b = BADS(
            lambda x: float(np.sum(np.log(np.asarray(x)) ** 2)),
            np.array([10, 10], dtype=dt),
            np.array([1, 1], dtype=dt),
            np.array([1000, 1000], dtype=dt),
            np.array([2, 2], dtype=dt),
            np.array([500, 500], dtype=dt),
            options={"display": "off", "random_seed": 0},
        )
        print(
            dt.__name__,
            "BADS plb_orig",
            b.optim_state["plb_orig"],
            "plb u",
            b.optim_state["plb"],
            "pub u",
            b.optim_state["pub"],
        )
    except Exception as e:
        print(dt.__name__, type(e).__name__, str(e)[:80])
