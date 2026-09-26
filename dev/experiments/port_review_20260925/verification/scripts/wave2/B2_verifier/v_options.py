"""I-F4/C-F6 sloppy_improvement=False; I-F5/C-F8 f_vals; I-F7/C-F7 display levels."""
import logging
import traceback

import common
import numpy as np

from pybads import BADS

f = lambda x: float(np.sum(np.ravel(x) ** 2))


def mk(**o):
    oo = dict(random_seed=0, max_fun_evals=30)
    oo.update(o)
    return BADS(
        f,
        np.array([1.0, 1.0]),
        np.full(2, -5.0),
        np.full(2, 5.0),
        np.full(2, -2.0),
        np.full(2, 2.0),
        options=oo,
    )


for label, o in [
    (
        "sloppy_improvement=False",
        dict(sloppy_improvement=False, display="off"),
    ),
    ("f_vals=[2.0], display off", dict(f_vals=np.array([2.0]), display="off")),
]:
    try:
        r = mk(**o).optimize()
        print(label, "-> ran, fval", r["fval"])
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(
            label,
            "->",
            type(e).__name__,
            e,
            "| at",
            tb.filename.split("pybads-review/")[-1],
            tb.lineno,
        )


class H(logging.Handler):
    def __init__(self):
        super().__init__()
        self.n = 0

    def emit(self, rec):
        self.n += 1


for disp in ["off", "notify", "final", "iter", "full"]:
    b = mk(display=disp)
    h = H()
    b.logger.addHandler(h)
    b.optimize()
    print(
        f"display={disp!r}: logger level {logging.getLevelName(b.logger.level)}, records emitted {h.n}"
    )
