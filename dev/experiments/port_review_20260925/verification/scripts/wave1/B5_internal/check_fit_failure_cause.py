import copy
import traceback

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS


def f(x):
    x = np.atleast_2d(x)
    return float(np.sum((np.array([1.0, 10.0, 0.3]) * x) ** 2))


b = BADS(
    f,
    np.array([[2.0, 2.0, 2.0]]),
    np.array([[-20.0] * 3]),
    np.array([[20.0] * 3]),
    np.array([[-5.0] * 3]),
    np.array([[5.0] * 3]),
    options={"random_seed": 1, "display": "off", "max_fun_evals": 60},
)
b.optimize()
gp_train = {
    "init_N": 8,
    "opts_N": 1,
    "n_samples": 0,
    "init_method": "rand",
    "tol_opt": 1e-5,
}
gp2 = copy.deepcopy(b.iteration_history["gp"][b.optim_state["iter"]])
hyp = gp2.get_hyperparameters(as_array=True)
print("hyp", gp2.get_hyperparameters()[0])
print("y range", gp2.y.min(), gp2.y.max(), "X spread", np.ptp(gp2.X, axis=0))
tmp = copy.deepcopy(gp2)
try:
    tmp.fit(
        gp2.X,
        gp2.y,
        gp2.s2,
        hyp0=hyp,
        options=gp_train,
        rng=np.random.default_rng(0),
    )
    print("fit ok")
except Exception as e:
    print(type(e).__name__, e)
    traceback.print_exc(limit=-6)
# The posterior at the held hyperparameters
try:
    g3 = copy.deepcopy(gp2)
    g3.update(hyp=hyp)
    print("update at held hyp ok")
except Exception as e:
    print("update at held hyp:", type(e).__name__, e)
# Which design points fail? Evaluate the objective at prior draws from gpyreg's own design
fails = 0
for s in range(20):
    t = copy.deepcopy(gp2)
    try:
        t.fit(
            gp2.X,
            gp2.y,
            gp2.s2,
            hyp0=hyp,
            options=gp_train,
            rng=np.random.default_rng(s),
        )
    except np.linalg.LinAlgError:
        fails += 1
print("fits with LinAlgError over 20 design seeds:", fails)
