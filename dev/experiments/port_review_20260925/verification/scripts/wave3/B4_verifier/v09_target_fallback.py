"""I-F7 / K3: _get_target_from_gp_ with a non-finite prediction; I-F8: argmin
on a NaN LCB; K11: the Sto-BADS rule on a NaN estimate; C-F7: the quantile."""

import numpy as np
from scipy.special import erfcinv
from vhdr import box, sphere

from pybads import BADS

D = 2
lb, ub, plb, pub = box(D)
b = BADS(
    sphere,
    np.ones(D),
    lb,
    ub,
    plb,
    pub,
    options=dict(random_seed=0, display="off", max_fun_evals=40),
)
b.optimize()
b.optim_state["fval"] = 0.5
b.optim_state["fsd"] = 0.1
gp = b.iteration_history.get("gp")[0]
hyp = gp.get_hyperparameters(as_array=True)


class FakeGP:
    """A GP stand-in whose predict returns the given values."""

    def __init__(self, mu, s2):
        self.mu, self.s2 = mu, s2

    def set_hyperparameters(self, h):
        pass

    def predict(self, x):
        return np.array([[self.mu]]), np.array([[self.s2]])


for mu, s2 in ((np.nan, 0.01), (0.3, np.nan), (0.3, np.inf), (np.nan, np.nan)):
    ft_mu, ft_s, ft = b._get_target_from_gp_(np.zeros(D), FakeGP(mu, s2), hyp)
    # what the fallback intends: fval and fsd for mu and sigma
    intended = (
        0.5
        - b.optim_state["sd_level"]
        * np.sqrt(0.1**2 + b.options["tol_fun"] ** 2)
        if not (np.isfinite(mu) and np.isfinite(s2))
        else None
    )
    print(
        f"pred mu={mu}, s2={s2}: f_target_mu={np.ravel(ft_mu)}, f_target_s={np.ravel(ft_s)}, f_target={np.ravel(ft)}, "
        f"with fsd in the formula: {intended}"
    )

# I-F8: np.argmin on a LCB with a NaN, and the fallback test after it
z = np.array([[0.3], [np.nan], [-1.0]])
i = np.argmin(z)
print(
    "argmin with NaN ->",
    i,
    "| nanargmin ->",
    np.nanargmin(z),
    "| fallback condition fires:",
    i is None or i.size < 1 or np.any(~np.isfinite(i)),
)

# K11: the Sto-BADS rule on a NaN estimate
print(
    "sto rule, NaN f_new ->",
    b._sto_success_improvement_(1.0, np.nan, 0.1, np.nan, 0.5),
    "| finite, uncertain ->",
    b._sto_success_improvement_(1.0, 0.99, 0.1, 0.1, 0.5),
)

# C-F7: _eval_improvement_ at q in {0, 1} and outside (0, 1)
for q in (0.0, 1.0, -0.1, 1.2, 0.5):
    print(
        f"q={q}: level 0 (sds 0) ->",
        b._eval_improvement_(1.0, 0.5, 0.0, 0.0, q),
        "| sds 0.1 ->",
        b._eval_improvement_(1.0, 0.5, 0.1, 0.1, q),
    )
