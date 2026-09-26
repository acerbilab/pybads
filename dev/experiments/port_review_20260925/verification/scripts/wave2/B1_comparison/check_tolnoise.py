import logging

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)
print(
    "PyBADS tol_noise =",
    np.spacing(1.0) * 1e-3,
    " MATLAB TolNoise = sqrt(eps)*TolFun =",
    np.sqrt(np.spacing(1.0)) * 1e-3,
)
for jitter in (0.0, 1e-15, 1e-13, 1e-12, 1e-10):
    nrng = np.random.default_rng(1000)

    def f(x):
        return float(np.sum(np.ravel(x) ** 2) + 1.0 + jitter * nrng.normal())

    b = BADS(
        f,
        x0=np.array([1.0, 1.0]),
        lower_bounds=-5 * np.ones(2),
        upper_bounds=5 * np.ones(2),
        plausible_lower_bounds=-2 * np.ones(2),
        plausible_upper_bounds=2 * np.ones(2),
        options={"random_seed": 0, "display": "off", "max_fun_evals": 100},
    )
    r = b.optimize()
    d = None
    print(
        f"jitter SD {jitter:g}: level={b.optim_state['uncertainty_handling_level']} target_type={r['target_type']} func_count={r['func_count']} fval={r['fval']:.6g} x={np.round(r['x'].ravel(),4)}"
    )
