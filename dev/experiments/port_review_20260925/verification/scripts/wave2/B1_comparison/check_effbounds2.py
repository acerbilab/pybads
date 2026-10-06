import logging

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.basicConfig(format="%(message)s")
for x0 in (0.01, 0.5, 2.0):
    X = []

    def f(x):
        X.append(float(np.ravel(x)[0]))
        return float((np.log10(np.ravel(x)[0]) + 2.0) ** 2)

    b = BADS(
        f,
        x0=np.array([x0]),
        lower_bounds=np.array([0.001]),
        upper_bounds=np.array([1000.0]),
        plausible_lower_bounds=np.array([0.01]),
        plausible_upper_bounds=np.array([100.0]),
        options={
            "random_seed": 0,
            "display": "off",
            "max_fun_evals": 60,
            "uncertainty_handling": False,
        },
    )
    vt = b.var_transf
    print(
        f"x0 given={x0}: x0 used={b.x0.ravel()} plb/pub used=[{vt.orig_plb.ravel()}, {vt.orig_pub.ravel()}] log={vt.apply_log_t.ravel()} lb_u={vt.lb.ravel()} ub_u={vt.ub.ravel()}"
    )
    r = b.optimize()
    ne = b.optim_state["eff_starting_points"]
    print(
        f"   first evals: {np.round(X[:ne], 4)}; result x={r['x'].ravel()} fval={r['fval']:.3g} func_count={r['func_count']} result x0={r['x0'].ravel()}"
    )
