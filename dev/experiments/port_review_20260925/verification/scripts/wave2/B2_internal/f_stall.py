import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


class P(BADS):
    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        s.log = []

    def _search_step_(s, gp):
        s.log.append(("S", s.optim_state["iter"]))
        return super()._search_step_(gp)

    def _poll_step_(s, gp):
        s.log.append(("P", s.optim_state["iter"]))
        return super()._poll_step_(gp)


def f(x):
    x = np.ravel(x)
    return float(np.sum(np.abs(x)) + 0.3 * np.sum(np.cos(7 * x)))


D = 2
for seed in range(4):
    b = P(
        f,
        np.full((1, D), 0.9),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options=dict(random_seed=seed, display="off", max_fun_evals=200),
    )
    r = b.optimize()
    if "tol_fun" not in r["message"]:
        print("seed", seed, "ended by", r["message"])
        continue
    k = b.optim_state["iter"]
    T = b.options["tol_stall_iters"]
    last_pass = b.log[-1]
    fv = b.iteration_history.get("fval").astype(float)
    print(
        f"seed {seed}: T={T}, ended at iteration index {k} on a {'search' if last_pass[0]=='S' else 'poll'} pass;"
        f" polls in iteration {k}: {sum(1 for p in b.log if p==('P', k))}; base index {k-T};"
        f" fval at base {fv[k-T]:.6g}, at end {r['fval']:.6g}; complete iterations after base: {k-1-(k-T)}"
    )
