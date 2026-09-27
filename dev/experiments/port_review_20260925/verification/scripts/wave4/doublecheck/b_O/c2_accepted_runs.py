"""Short runs (D = 2, at most 80 evaluations) with values that the checks
of W4-18, W4-19, W4-25 and W4-29 accept, to see whether the run uses them."""
import traceback
import warnings
from decimal import Decimal
from fractions import Fraction

import hdr  # noqa
import numpy as np

from pybads import BADS


def f(x):
    x = np.atleast_2d(x)
    return float(np.sum(x**2) + 0.3 * np.sum(np.cos(3 * x)))


x0 = np.array([[1.5, -1.2]])
lb, ub = -5 * np.ones((1, 2)), 5 * np.ones((1, 2))
plb, pub = -3 * np.ones((1, 2)), 3 * np.ones((1, 2))
cases = [
    ("baseline", {}),
    (
        "hedge_gamma np.complex128(0.1-5j)",
        {"hedge_gamma": np.complex128(0.1 - 5j)},
    ),
    ("hedge_gamma np.complex128(0.1)", {"hedge_gamma": np.complex128(0.1)}),
    ("hedge_gamma np.array([[0.1]])", {"hedge_gamma": np.array([[0.1]])}),
    ("hedge_gamma np.array([0.1])", {"hedge_gamma": np.array([0.1])}),
    ("hedge_beta np.array([[1.0]])", {"hedge_beta": np.array([[1.0]])}),
    ("hedge_decay np.array([[0.5]])", {"hedge_decay": np.array([[0.5]])}),
    ("hedge_decay np.array([0.5])", {"hedge_decay": np.array([0.5])}),
    (
        "hedge_decay np.complex128(0.5+3j)",
        {"hedge_decay": np.complex128(0.5 + 3j)},
    ),
    ("hedge_gamma Decimal('0.25')", {"hedge_gamma": Decimal("0.25")}),
    ("hedge_beta Fraction(1,4)", {"hedge_beta": Fraction(1, 4)}),
    ("n_search_iter 4096", {"n_search_iter": 4096}),
    ("n_search_iter 4097", {"n_search_iter": 4097}),
    ("sqrt_beta [0.5] list", {"search_acq_fcn": ("acq_LCB", [0.5])}),
    (
        "sqrt_beta np.array([[2.0]])",
        {"search_acq_fcn": ("acq_LCB", np.array([[2.0]]))},
    ),
]
for label, opts in cases:
    o = {"display": "off", "max_fun_evals": 80, "random_seed": 3}
    o.update(opts)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            b = BADS(f, x0, lb, ub, plb, pub, options=o)
            r = b.optimize()
            print(
                f"{label:36s} ran: fval={r['fval']:.6g} func_count={r['func_count']} "
                f"warnings={sorted({type(x.message).__name__ for x in w})}"
            )
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            print(
                f"{label:36s} FAILED {type(e).__name__}: {str(e)[:90]} "
                f"at {tb.filename.split('/')[-1]}:{tb.lineno}"
            )
