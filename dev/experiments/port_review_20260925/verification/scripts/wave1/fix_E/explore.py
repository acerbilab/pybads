import time

import gpyreg as gpr
import numpy as np

import pybads
from pybads import BADS
from pybads.bads import gaussian_process_train as gpt

print(pybads.__file__, gpr.__file__)


def f(x):
    x = np.ravel(x)
    return float(np.sum((x - 0.3) ** 2 * np.array([1.0, 4.0, 9.0])))


t = time.time()
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
print("time", time.time() - t)
gp = b.iteration_history["gp"][b.optim_state["iter"]]
print(gp.X.shape, gp.get_bounds())
print(gp.get_priors())
print(gp.get_hyperparameters(as_array=True))
print(gp.s2)
h = gp.get_hyperparameters(as_array=True)[0]
print("lp fitted", gp.log_posterior(h))
w = h.copy()
w[3] -= 3
print("lp lower outputscale", gp.log_posterior(w))
n = h.copy()
n[0] = np.nan
try:
    print("lp nan", gp.log_posterior(n))
except Exception as e:
    print("nan raised", type(e).__name__, e)
o = h.copy()
o[0] += 1
print("lp out of bounds", gp.log_posterior(o))
