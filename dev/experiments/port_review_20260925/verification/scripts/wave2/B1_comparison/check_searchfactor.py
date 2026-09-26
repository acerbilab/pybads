import logging

import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS

logging.getLogger("BADS").setLevel(logging.ERROR)


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ellip(x):
    x = np.ravel(x)
    return float(np.sum((x / np.arange(1, x.size + 1) ** 2) ** 2))


for name, fun, D in (
    ("rosenbrock", rosen, 2),
    ("ellipsoid", ellip, 6),
    ("rosenbrock", rosen, 6),
):
    b = BADS(
        fun,
        x0=np.zeros(D),
        lower_bounds=-20 * np.ones(D),
        upper_bounds=20 * np.ones(D),
        plausible_lower_bounds=-5 * np.ones(D),
        plausible_upper_bounds=5 * np.ones(D),
        options={
            "random_seed": 0,
            "display": "off",
            "max_fun_evals": 200,
            "uncertainty_handling": False,
        },
    )
    r = b.optimize()
    lsf = np.array(b.optim_state["search_stats"]["log_search_factor"])
    sf = np.exp(lsf)
    print(
        f"{name} D={D}: search_n_try={b.options['search_n_try']}, searches={sf.size}, factor<0.5 used in {np.sum(sf < 0.5 - 1e-12)} "
        f"({np.mean(sf < 0.5-1e-12):.0%}), min factor={sf.min():.4f}; fval={r['fval']:.3g} evals={r['func_count']}"
    )
