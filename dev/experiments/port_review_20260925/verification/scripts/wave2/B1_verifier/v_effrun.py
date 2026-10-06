"""B1 verifier, C-F1 / I-F4 / I-F3: consequence of the effective bounds
in runs. 'pybads' is the code as it is; 'no-move' wraps
BADS._bounds_check_ in memory so that, after its checks pass, x0, plb and
pub are the user's (plb/pub = lb/ub when omitted), which is what MATLAB's
boundscheck.m/setupvars.m do. 200 evaluations at most, seed 1."""
import common  # noqa: F401
import numpy as np

from pybads import BADS

orig = BADS._bounds_check_


def nomove(self, x0, lb, ub, plb=None, pub=None, nbc=None):
    x0u = np.atleast_2d(x0).copy()
    plbu = np.atleast_2d(lb if plb is None else plb).astype(float).copy()
    pubu = np.atleast_2d(ub if pub is None else pub).astype(float).copy()
    out = orig(self, x0, lb, ub, plb, pub, nbc)
    return (x0u, out[1], out[2], plbu, pubu)


def f1(x):
    return float(np.sum((np.log10(np.asarray(x).ravel()) + 2) ** 2))


cases = [
    (
        "1-D log, x0 = optimum 0.01, plb=1e-2, pub=1e2",
        f1,
        [0.01],
        [1e-3],
        [1e3],
        [1e-2],
        [1e2],
    ),
    (
        "3-D log, x0 = 1, plb/pub omitted",
        f1,
        [1.0] * 3,
        [1e-3] * 3,
        [1e3] * 3,
        None,
        None,
    ),
    (
        "2-D linear, optimum at lb=-5, plb/pub omitted",
        lambda x: float(np.sum((np.asarray(x).ravel() + 5) ** 2)),
        [0.0, 0.0],
        [-5.0, -5.0],
        [5.0, 5.0],
        None,
        None,
    ),
]
for name, fun, x0, lb, ub, plb, pub in cases:
    print(f"\n== {name}")
    for variant in ("pybads", "no-move"):
        BADS._bounds_check_ = orig if variant == "pybads" else nomove
        a = lambda v: None if v is None else np.array(v, float)
        b = BADS(
            fun,
            a(x0),
            a(lb),
            a(ub),
            a(plb),
            a(pub),
            options={"display": "off", "random_seed": 1, "max_fun_evals": 200},
        )
        vt = b.var_transf
        first = b.var_transf.inverse_transf(
            np.atleast_2d(b.optim_state["u"])
        ).ravel()
        r = b.optimize()
        X = b.function_logger.X_orig[: b.function_logger.Xn + 1]
        Y = b.function_logger.Y_orig[: b.function_logger.Xn + 1].ravel()
        hit = np.flatnonzero(Y < 1e-4)
        print(
            f"  {variant:8s} plb {np.round(vt.orig_plb.ravel(), 4)} "
            f"pub {np.round(vt.orig_pub.ravel(), 2)} first x "
            f"{np.round(first, 4)} f(first) {fun(first):.3g}; final fval "
            f"{r['fval']:.3g} in {r['func_count']} evals; first eval with "
            f"f<1e-4: {hit[0] + 1 if hit.size else 'none'}"
        )
BADS._bounds_check_ = orig
