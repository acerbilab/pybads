import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
exec(
    open("f1_move_revert.py")
    .read()
    .split("def run(seed):")[0]
    .split("print(pybads.__file__)")[1]
    .split("\n", 2)[2]
)


class FixedProbe(Probe):
    @property
    def best_u(self):
        return self.u_best

    @best_u.setter
    def best_u(self, v):
        self.u_best = np.array(v, dtype=float).copy()
        self.moves_set = getattr(self, "moves_set", 0) + 1


class AsIs(Probe):
    pass


for cls in (AsIs, FixedProbe):
    for seed in (0, 1):
        noise_rng = np.random.default_rng(1000 + seed)

        def fun(x):
            x = np.ravel(x)
            return float(np.sum(x**2) + 0.5 * noise_rng.standard_normal())

        b = cls(
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
        r = b.optimize()
        # iteration history: u per iteration
        us = [np.round(np.ravel(u), 4) for u in b.iteration_history.get("u")]
        print(
            cls.__name__,
            seed,
            "moves via setter:",
            getattr(b, "moves_set", 0),
            "fcount",
            r["func_count"],
        )
        print("   u history:", [tuple(u) for u in us])
