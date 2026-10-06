"""Noisy run: the incumbent's fval in the first two iterations is the raw
minimum observation of the initial design, not a GP estimate."""
import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


class P(BADS):
    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        s.rows = []

    def _poll_step_(s, gp):
        x = s.var_transf.inverse_transf(np.atleast_2d(s.u))
        mu, s2 = gp.predict(np.atleast_2d(s.u))
        s.rows.append(
            (
                s.optim_state["iter"],
                float(s.fval),
                float(s.fsd),
                float(mu.item()),
                float(np.sum(x**2)),
                s.mesh_size,
            )
        )
        out = super()._poll_step_(gp)
        s.rows[-1] = s.rows[-1] + (s.mesh_size,)
        return out


for seed in range(3):
    rng = np.random.default_rng(1000 + seed)
    fun = lambda x: float(
        np.sum(np.ravel(x) ** 2) + 0.5 * rng.standard_normal()
    )
    b = P(
        fun,
        np.array([[1.5, -1.0]]),
        np.array([[-5, -5]]),
        np.array([[5, 5]]),
        np.array([[-2, -2]]),
        np.array([[2, 2]]),
        options=dict(
            uncertainty_handling=True,
            max_fun_evals=120,
            random_seed=seed,
            display="off",
        ),
    )
    b.optimize()
    for r in b.rows[:3]:
        print(
            f"seed {seed} iter {r[0]+1}: incumbent fval {r[1]:+.3f} (fsd {r[2]:.2f}), GP mean there {r[3]:+.3f}, true f {r[4]:.3f}; mesh {r[5]} -> {r[6]}"
        )
