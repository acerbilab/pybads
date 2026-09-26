import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
for seed in (0, 1, 2):
    b = BADS(
        lambda x: float(
            (np.ravel(x)[0] - 0.4) ** 2 * 3 + np.sin(5 * np.ravel(x)[0])
        ),
        np.array([[1.5]]),
        np.array([[-5.0]]),
        np.array([[5.0]]),
        np.array([[-2.0]]),
        np.array([[2.0]]),
        options={"random_seed": seed, "display": "off", "max_fun_evals": 60},
    )
    r = b.optimize()
    gp = b.iteration_history["gp"][b.optim_state["iter"]]
    print(
        seed,
        r["x"],
        r["fval"],
        r["func_count"],
        "len_scale",
        gp.temporary_data["len_scale"],
        "fitted",
        np.exp(gp.get_hyperparameters()[0]["covariance_log_lengthscale"]),
        "ntrain",
        b.optim_state["ntrain"],
    )
