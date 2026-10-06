import logging

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)
logging.getLogger("BADS").setLevel(logging.ERROR)
orig_search = BADS._search_step_


def search_full_move(self, gp):
    # Variant: the re-estimate's move also moves the incumbent (u_best), as a
    # full move would; at search entry u differs from u_best only after it
    if not np.array_equal(np.ravel(self.u), np.ravel(self.u_best)):
        self.u_best = np.array(self.u, dtype=float).copy()
        self.optim_state["u"] = self.u_best.copy()
        (
            self.optim_state["yval"],
            self.optim_state["fval"],
            self.optim_state["fsd"],
        ) = (self.yval, self.fval, self.fsd)
    return orig_search(self, gp)


D = 2
xopt = np.full(D, 0.3)
res = {"as_is": [], "full_move": []}
for variant in res:
    BADS._search_step_ = (
        orig_search if variant == "as_is" else search_full_move
    )
    for seed in range(10):
        nrng = np.random.default_rng(100 + seed)
        f = lambda x: float(np.sum((x - 0.3) ** 2) + nrng.normal())
        b = BADS(
            f,
            np.full(D, 1.5),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options=dict(
                display="off",
                random_seed=seed,
                max_fun_evals=150,
                uncertainty_handling=True,
            ),
        )
        r = b.optimize()
        res[variant].append(float(np.sum((np.ravel(r["x"]) - xopt) ** 2)))
a, m = np.array(res["as_is"]), np.array(res["full_move"])
print("true f(x) - f* per seed, as is   :", np.round(a, 4))
print("true f(x) - f* per seed, full move:", np.round(m, 4))
print(
    "runs changed:",
    int(np.sum(~np.isclose(a, m))),
    "of",
    a.size,
    "; median as is %.4f, full move %.4f" % (np.median(a), np.median(m)),
)
