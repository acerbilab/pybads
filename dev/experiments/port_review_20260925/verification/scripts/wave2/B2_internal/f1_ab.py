"""F1: the same runs with self.best_u aliased to self.u_best (the move kept)."""
import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


class Fixed(BADS):
    @property
    def best_u(self):
        return self.u_best

    @best_u.setter
    def best_u(self, v):
        self.u_best = np.array(v, dtype=float).copy()


def run(cls, seed):
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
    return (
        float(np.sum(np.ravel(r["x"]) ** 2)),
        r["func_count"],
        r["iterations"],
        r["fval"],
        r["fsd"],
    )


for seed in range(6):
    a = run(BADS, seed)
    f = run(Fixed, seed)
    print(
        f"seed {seed}: as is: ftrue={a[0]:.4g} fcount={a[1]} iters={a[2]} fval={a[3]:.4g}+-{a[4]:.3g} | "
        f"move kept: ftrue={f[0]:.4g} fcount={f[1]} iters={f[2]} fval={f[3]:.4g}+-{f[4]:.3g}"
    )
