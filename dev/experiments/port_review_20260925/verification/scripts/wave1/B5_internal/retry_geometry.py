import copy

import gpyreg
import numpy as np

import pybads

print("pybads", pybads.__file__)
print("gpyreg", gpyreg.__file__)
import pybads.bads.gaussian_process_train as gpt
from pybads import BADS


def ell(x):
    x = np.atleast_2d(x)
    D = x.shape[1]
    return float(np.sum((10 ** (np.arange(D) / (D - 1) * 2) * x) ** 2))


b = BADS(
    ell,
    np.full((1, 3), 1.5),
    np.full((1, 3), -10.0),
    np.full((1, 3), 10.0),
    np.full((1, 3), -3.0),
    np.full((1, 3), 3.0),
    options={"random_seed": 0, "display": "off", "max_fun_evals": 80},
)
b.optimize()
gp = copy.deepcopy(b.iteration_history["gp"][b.optim_state["iter"]])
old = gp.get_hyperparameters()[0]
print(
    "entry: log ell",
    np.round(old["covariance_log_lengthscale"], 3),
    "poll_scale",
    np.round(gp.temporary_data["poll_scale"], 3),
)

flag = {"armed": False}
orig_rob = gpt._robust_gp_fit_


def rob(*a, **k):
    out = orig_rob(*a, **k)
    flag["armed"] = True
    return out


gpt._robust_gp_fit_ = rob
orig_update = gpyreg.GP.update


def upd(self, *a, **k):
    if flag["armed"]:
        flag["armed"] = False
        raise np.linalg.LinAlgError("injected")
    return orig_update(self, *a, **k)


gpyreg.GP.update = upd
# Move the centre a little so that the training set changes, and refit
u = b.u.copy() + 0.5
gp2, ef = gpt.local_gp_fitting(
    gp,
    u,
    b.function_logger,
    b.options,
    b.optim_state,
    b.iteration_history,
    True,
    rng=np.random.default_rng(1),
)
h = gp2.get_hyperparameters()[0]
ll = h["covariance_log_lengthscale"]
expected_ps = np.exp(ll - ll.mean())
print(
    "exit flag",
    ef,
    "markers",
    {k: gp2.temporary_data.get(k) for k in ("needs_rebuild", "needs_refit")},
)
print(
    "GP held log ell",
    np.round(ll, 3),
    "(the entry values:",
    np.allclose(ll, old["covariance_log_lengthscale"]),
    ")",
)
print(
    "temporary_data len_scale",
    np.round(np.log(gp2.temporary_data["len_scale"]), 3),
    " poll_scale",
    np.round(gp2.temporary_data["poll_scale"], 3),
)
print(
    "poll_scale implied by the GP's hyperparameters (before clipping)",
    np.round(expected_ps, 3),
)
