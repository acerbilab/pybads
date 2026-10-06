import copy

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
gp = copy.deepcopy(b.iteration_history["gp"][b.optim_state["iter"]])
hyp = gp.get_hyperparameters(as_array=True)
gp_train = {
    "init_N": 8,
    "opts_N": 1,
    "n_samples": 0,
    "init_method": "rand",
    "tol_opt": 1e-5,
}

# Record what each try receives: the noise lower bound and the noise start
calls = []
orig_fit = gpyreg.GP.fit


def failing_fit(
    self, X=None, y=None, s2=None, hyp0=None, options=None, rng=None
):
    names = self.get_hyperparameters()[0].keys()
    b_ = self.get_bounds()["noise_log_scale"]
    h = self.hyperparameters_to_dict(np.atleast_2d(hyp0))[0]["noise_log_scale"]
    calls.append((len(y), float(np.ravel(b_[0])[0]), float(np.ravel(h)[0])))
    raise np.linalg.LinAlgError("injected")


gpyreg.GP.fit = failing_fit
try:
    out = gpt._robust_gp_fit_(
        gp,
        gp.X,
        gp.y,
        gp.s2,
        hyp,
        gp_train,
        b.optim_state,
        b.options,
        np.random.default_rng(0),
    )
    print("returned exit flag", out[3])
except Exception as e:
    print("all tries failed ->", type(e).__name__, e)
print(
    "orig noise lower bound",
    float(np.ravel(gp.get_bounds()["noise_log_scale"][0])[0]),
)
print("(n_train, noise lower bound, noise start) per try:")
for c in calls:
    print("  ", c)

# Now: first try fails, second succeeds
calls.clear()
state = {"n": 0}


def fail_once(self, *a, **k):
    state["n"] += 1
    if state["n"] == 1:
        raise np.linalg.LinAlgError("injected")
    return orig_fit(self, *a, **k)


gpyreg.GP.fit = fail_once
gp2 = copy.deepcopy(b.iteration_history["gp"][b.optim_state["iter"]])
out = gpt._robust_gp_fit_(
    gp2,
    gp2.X,
    gp2.y,
    gp2.s2,
    hyp,
    gp_train,
    b.optim_state,
    b.options,
    np.random.default_rng(0),
)
print(
    "fail-once exit flag",
    out[3],
    "gp noise lower bound after",
    float(np.ravel(gp2.get_bounds()["noise_log_scale"][0])[0]),
    "fitted noise",
    out[0].get_hyperparameters()[0]["noise_log_scale"],
)
