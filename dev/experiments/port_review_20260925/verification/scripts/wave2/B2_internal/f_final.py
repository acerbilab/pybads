import io
import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
a = np.array([1.0, np.nan, 0.5, 2.0], dtype=object)
print("nanargmin(object) ->", np.nanargmin(a), "nanargmax ->", np.nanargmax(a))


def mk(seed, sd=0.5, level2=False):
    rng = np.random.default_rng(1000 + seed)

    def fun(x):
        v = float(np.sum(np.ravel(x) ** 2) + sd * rng.standard_normal())
        return (v, sd) if level2 else v

    return fun


D = 2
bounds = (
    np.array([[-5, -5]]),
    np.array([[5, 5]]),
    np.array([[-2, -2]]),
    np.array([[2, 2]]),
)
for nfs, lvl2 in [(1, False), (1, True), (0, False)]:
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    lg = logging.getLogger("BADS")
    lg.addHandler(h)
    opts = dict(
        uncertainty_handling=True,
        max_fun_evals=120,
        random_seed=3,
        noise_final_samples=nfs,
    )
    if lvl2:
        opts["specify_target_noise"] = True
    b = BADS(
        mk(3, level2=lvl2), np.array([[1.5, -1.0]]), *bounds, options=opts
    )
    yval_loop_end = None
    r = b.optimize()
    lg.removeHandler(h)
    last = [
        l
        for l in stream.getvalue().splitlines()
        if "function value at minimum" in l or "Estimated" in l
    ]
    hist_y = b.iteration_history.get("yval")
    print(
        f"nfs={nfs} level2={lvl2}: yval_vec={r['yval_vec']}, ysd_vec={r['ysd_vec']}, fval={r['fval']:.4g}, fsd={r['fsd']:.4g}"
    )
    print("   final message:", last)
    print(
        "   b.yval (chosen iterate's observation) =",
        b.yval,
        "; hist yval last =",
        hist_y[-1],
    )
