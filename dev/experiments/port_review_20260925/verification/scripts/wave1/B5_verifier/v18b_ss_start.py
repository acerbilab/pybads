import common  # noqa
import numpy as np

from pybads import BADS


def f(x):
    x = np.ravel(x)
    return float(np.sum((x - 0.3) ** 2 * np.array([1.0, 4.0, 9.0])))


D = 3
b = BADS(
    f,
    np.array([[1.0, -1.0, 0.5]]),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    options={"random_seed": 5, "display": "off", "max_fun_evals": 70},
)
b.optimize()
gp = b.iteration_history["gp"][b.optim_state["iter"]]
h = gp.get_hyperparameters(as_array=True)[0]
print("hyp - lower bound:", np.round(h - gp.lower_bounds, 4))
print("upper bound - hyp:", np.round(gp.upper_bounds - h, 4))
