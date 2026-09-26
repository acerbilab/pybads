import logging

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)
f = lambda x: float(np.sum(np.atleast_2d(x) ** 2))


def attempt(label, **kw):
    opts = kw.pop("options", {})
    opts.setdefault("random_seed", 0)
    opts.setdefault("display", "off")
    try:
        b = BADS(f, options=opts, **kw)
        vt = b.var_transf
        print(
            f"[{label}] OK D={b.D} x0={b.x0} plb_orig={vt.orig_plb} pub_orig={vt.orig_pub} "
            f"log={vt.apply_log_t.astype(int)} lb_u={vt.lb} ub_u={vt.ub} u0={b.u}"
        )
        return b
    except Exception as e:
        print(f"[{label}] {type(e).__name__}: {str(e).strip()[:160]}")


# a. scalar bounds, D=3
attempt(
    "scalar lb/ub D=3",
    x0=np.zeros(3),
    lower_bounds=-5,
    upper_bounds=5,
    plausible_lower_bounds=-2,
    plausible_upper_bounds=2,
)
attempt(
    "scalar plb/pub only D=3",
    x0=np.zeros(3),
    lower_bounds=-5 * np.ones(3),
    upper_bounds=5 * np.ones(3),
    plausible_lower_bounds=-2,
    plausible_upper_bounds=2,
)
# b. mixed bounded/unbounded
attempt(
    "mixed bounded+unbounded",
    x0=np.array([0.5, 0.0]),
    lower_bounds=np.array([0.0, -np.inf]),
    upper_bounds=np.array([1.0, np.inf]),
    plausible_lower_bounds=np.array([0.1, -1.0]),
    plausible_upper_bounds=np.array([0.9, 1.0]),
)
# c. half-bounded
attempt(
    "half-bounded",
    x0=np.array([2.0]),
    lower_bounds=np.array([0.0]),
    upper_bounds=np.array([np.inf]),
    plausible_lower_bounds=np.array([1.0]),
    plausible_upper_bounds=np.array([10.0]),
)
# d. plb = lb (not given), finite bounds
attempt(
    "plb,pub not given [-5,5]",
    x0=np.zeros(2),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
)
attempt(
    "plb=lb given [-5,5]",
    x0=np.zeros(2),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-5 * np.ones(2),
    plausible_upper_bounds=5 * np.ones(2),
)
attempt(
    "plb=lb [1,10] (MATLAB: log)",
    x0=np.array([2.0]),
    lower_bounds=np.array([1.0]),
    upper_bounds=np.array([10.0]),
)
attempt(
    "plb=lb [0.1,100] log",
    x0=np.array([2.0]),
    lower_bounds=np.array([0.1]),
    upper_bounds=np.array([100.0]),
)
# e. x0 on bound
attempt(
    "x0 on lb",
    x0=np.array([-5.0, 0.0]),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-2 * np.ones(2),
    plausible_upper_bounds=2 * np.ones(2),
)
# f. x0 with 2 rows
b = attempt(
    "x0 2 rows, lb None",
    x0=np.array([[0.0, 0.0], [1.0, 1.0]]),
    plausible_lower_bounds=None,
)
if b is not None:
    try:
        b.optimize()
    except Exception as e:
        print("   optimize():", type(e).__name__, str(e)[:120])
attempt(
    "x0 2 rows, bounds given",
    x0=np.array([[0.0, 0.0], [1.0, 1.0]]),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
)
# g. x0 None with lb/ub only
attempt(
    "x0 None, lb/ub only",
    x0=None,
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
)
# h. fun_values
attempt(
    "fun_values",
    x0=np.zeros(2),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-2 * np.ones(2),
    plausible_upper_bounds=2 * np.ones(2),
    options={
        "fun_values": {"X": np.array([[1.0, 1.0]]), "Y": np.array([[2.0]])}
    },
)
# i. non_box_cons returning scalar
attempt(
    "nonbcon scalar",
    x0=np.zeros(2),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-2 * np.ones(2),
    plausible_upper_bounds=2 * np.ones(2),
    non_box_cons=lambda x: np.sum(x**2) > 100,
)
attempt(
    "nonbcon (2,2) out",
    x0=np.zeros(2),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-2 * np.ones(2),
    plausible_upper_bounds=2 * np.ones(2),
    non_box_cons=lambda x: np.atleast_2d(x) ** 2 > 100,
)
# fixed variable with x0 different
attempt(
    "fixed var",
    x0=np.array([0.0, 1.0]),
    lower_bounds=np.array([-5.0, 1.0]),
    upper_bounds=np.array([5.0, 1.0]),
    plausible_lower_bounds=np.array([-2.0, 1.0]),
    plausible_upper_bounds=np.array([2.0, 1.0]),
)
# x0 outside plausible box
attempt(
    "x0 outside plausible box",
    x0=np.array([4.0, 0.0]),
    lower_bounds=-5 * np.ones(2),
    upper_bounds=5 * np.ones(2),
    plausible_lower_bounds=-2 * np.ones(2),
    plausible_upper_bounds=2 * np.ones(2),
)
# x0 = 1-D list
attempt(
    "x0 list, plb list",
    x0=[0.0, 0.0],
    lower_bounds=[-5.0, -5.0],
    upper_bounds=[5.0, 5.0],
    plausible_lower_bounds=[-2.0, -2.0],
    plausible_upper_bounds=[2.0, 2.0],
)
attempt(
    "x0 None, plb list",
    x0=None,
    lower_bounds=[-5.0, -5.0],
    upper_bounds=[5.0, 5.0],
    plausible_lower_bounds=[-2.0, -2.0],
    plausible_upper_bounds=[2.0, 2.0],
)
