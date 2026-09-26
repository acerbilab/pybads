"""_get_target_from_gp_ when the GP's prediction is not finite: the
fallback replaces the mean and SD by the incumbent's, but the target is
computed from the non-finite variance."""
import gpyreg
import numpy as np

import pybads

print("pybads:", pybads.__file__)
print("gpyreg:", gpyreg.__file__)
from pybads import BADS


class FakeGP:
    def __init__(self, mu, s2):
        self.mu, self.s2 = mu, s2

    def set_hyperparameters(self, hyp, compute_posterior=True):
        pass

    def predict(self, x, **kw):
        return np.array([[self.mu]]), np.array([[self.s2]])


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


D = 2
b = BADS(
    sphere,
    np.full((1, D), 0.5),
    -5 * np.ones((1, D)),
    5 * np.ones((1, D)),
    -2 * np.ones((1, D)),
    2 * np.ones((1, D)),
    options={"random_seed": 0, "display": "off"},
)
b.optim_state["fval"] = 3.0
b.optim_state["fsd"] = 0.2
for mu, s2 in [
    (1.0, 0.04),
    (np.nan, 0.04),
    (1.0, np.nan),
    (1.0, np.inf),
    (np.inf, 0.04),
]:
    out = b._get_target_from_gp_(np.zeros(D), FakeGP(mu, s2), None)
    print(
        "mu",
        mu,
        "s2",
        s2,
        "->",
        "f_target_mu",
        np.ravel(out[0]),
        "f_target_s",
        np.ravel(out[1]),
        "f_target",
        np.ravel(out[2]),
    )
