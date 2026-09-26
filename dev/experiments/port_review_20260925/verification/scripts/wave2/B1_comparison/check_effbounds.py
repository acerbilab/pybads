import logging

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)
import importlib.util

spec = importlib.util.spec_from_file_location("ct", "check_transform.py")
# reuse MATLAB transcription without running its tests
src = open("check_transform.py").read().split("def compare(")[0]
ns = {}
exec(src, ns)
MTrinfo = ns["MTrinfo"]


def matlab_setup(x0, lb, ub, plb, pub):
    D = x0.size
    lb, ub = [
        np.broadcast_to(np.asarray(a, float), (D,)).copy() for a in (lb, ub)
    ]
    plb = (
        lb.copy()
        if plb is None
        else np.broadcast_to(np.asarray(plb, float), (D,)).copy()
    )
    pub = (
        ub.copy()
        if pub is None
        else np.broadcast_to(np.asarray(pub, float), (D,)).copy()
    )
    T = MTrinfo(D, lb, ub, plb, pub, np.full(D, np.nan))
    sms = 2.0**-10
    u0 = sms * np.round(T.dir(x0.reshape(1, -1)) / sms)
    return T, u0


def case(label, x0, lb, ub, plb=None, pub=None):
    T, u0m = matlab_setup(np.asarray(x0, float), lb, ub, plb, pub)
    kw = dict(
        x0=np.asarray(x0, float),
        lower_bounds=np.asarray(lb, float),
        upper_bounds=np.asarray(ub, float),
    )
    if plb is not None:
        kw["plausible_lower_bounds"] = np.asarray(plb, float)
    if pub is not None:
        kw["plausible_upper_bounds"] = np.asarray(pub, float)
    b = BADS(lambda x: 0.0, options={"random_seed": 0, "display": "off"}, **kw)
    vt = b.var_transf
    print(
        f"[{label}]\n  MATLAB: log={T.logct.astype(int)} plb={T.old['plb']} pub={T.old['pub']} lb_u={T.lb} ub_u={T.ub} u0={u0m.ravel()} x0_eval={T.inv(u0m).ravel()}"
    )
    print(
        f"  PyBADS: log={vt.apply_log_t.ravel().astype(int)} plb={vt.orig_plb.ravel()} pub={vt.orig_pub.ravel()} lb_u={vt.lb.ravel()} ub_u={vt.ub.ravel()} u0={b.u} x0_eval={vt.inverse_transf(b.u.reshape(1,-1)).ravel()}"
    )
    # plausible box of PyBADS in MATLAB u units
    print(
        f"  PyBADS plausible box in MATLAB's u: [{T.dir(vt.orig_plb).ravel()}, {T.dir(vt.orig_pub).ravel()}]"
    )
    return b


case(
    "test config (-100,100; -8,12)",
    [4, 4, 4],
    -100 * np.ones(3),
    100 * np.ones(3),
    -8 * np.ones(3),
    12 * np.ones(3),
)
case(
    "plb=lb, pub=ub [-5,5]",
    [0.0, 0.0],
    -5 * np.ones(2),
    5 * np.ones(2),
    -5 * np.ones(2),
    5 * np.ones(2),
)
case("plb/pub omitted [-5,5]", [0.0, 0.0], -5 * np.ones(2), 5 * np.ones(2))
case(
    "log var, plb=lb [0.001,1000]", [1.0], [0.001], [1000.0], [0.001], [1000.0]
)
case("log var, plb=lb [0.01,100]", [1.0], [0.01], [100.0], [0.01], [100.0])
case("ratio 10, plb=lb [1,10]", [2.0], [1.0], [10.0], [1.0], [10.0])
case(
    "x0 on lb, plb inside",
    [-5.0, 0.0],
    -5 * np.ones(2),
    5 * np.ones(2),
    -2 * np.ones(2),
    2 * np.ones(2),
)
case(
    "x0 on lb=0, log-like [0,10] plb [0.1,5]",
    [0.0],
    [0.0],
    [10.0],
    [0.1],
    [5.0],
)
# Initial design of PyBADS for the log case, first evaluations
X = []


def f(x):
    X.append(float(np.ravel(x)[0]))
    return float((np.log10(np.ravel(x)[0]) + 2.0) ** 2)  # min at x=0.01


b = BADS(
    f,
    x0=np.array([1.0]),
    lower_bounds=np.array([0.001]),
    upper_bounds=np.array([1000.0]),
    plausible_lower_bounds=np.array([0.001]),
    plausible_upper_bounds=np.array([1000.0]),
    options={
        "random_seed": 0,
        "display": "off",
        "max_fun_evals": 60,
        "uncertainty_handling": False,
    },
)
r = b.optimize()
print(
    "log case: initial design x (first",
    b.optim_state["eff_starting_points"],
    "evals):",
    np.round(X[: b.optim_state["eff_starting_points"]], 4),
)
print(
    "log case: min x evaluated",
    min(X),
    " result x",
    r["x"],
    "fval",
    r["fval"],
    "func_count",
    r["func_count"],
)
