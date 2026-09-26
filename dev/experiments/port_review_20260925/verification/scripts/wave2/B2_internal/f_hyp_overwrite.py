"""After a move by the re-evaluation, does the moved iterate's hyperparameter
vector (best_gp_hyp) reach the poll's target? Line 1425 overwrites it at the
end of every pass."""
import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


class P(BADS):
    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        s.pending = None
        s.stats = dict(moves=0, poll_uses_moved=0, poll_after=0, hyp_differs=0)

    def _search_step_(s, gp):
        if not np.array_equal(np.ravel(s.u), np.ravel(s.u_best)):
            s.stats["moves"] += 1
            s.pending = np.array(s.best_gp_hyp, dtype=float).copy()
            s.stats["hyp_differs"] += int(
                not np.allclose(
                    s.pending, gp.get_hyperparameters(as_array=True)
                )
            )
        return super()._search_step_(gp)

    def _poll_step_(s, gp):
        if s.pending is not None:
            s.stats["poll_after"] += 1
            s.stats["poll_uses_moved"] += int(
                np.allclose(np.array(s.best_gp_hyp, dtype=float), s.pending)
            )
            s.pending = None
        return super()._poll_step_(gp)


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
            max_fun_evals=200,
            random_seed=seed,
            display="off",
        ),
    )
    b.optimize()
    print("seed", seed, b.stats)
