"""B7 verifier, K4 and F3/F2: the size of the initial design at D = 1..20,
default options, levels 0 and 1, measured by running _init_mesh_ only (no
GP), against the rounding alone and MATLAB's Ninit."""
import warnings

import gpyreg
import numpy as np

import pybads
from pybads import BADS

print("pybads", pybads.__file__, flush=True)
print("gpyreg", gpyreg.__file__, flush=True)


def sphere(x):
    x = np.atleast_2d(x)
    return float(np.sum((x - 0.1) ** 2))


class Noisy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        return sphere(x) + self.rng.normal()


def init_only(fun, D):
    lb, ub = -5 * np.ones((1, D)), 5 * np.ones((1, D))
    plb, pub = -2 * np.ones((1, D)), 2 * np.ones((1, D))
    x0 = 0.3 * np.ones((1, D))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BADS(
            fun,
            x0,
            lb,
            ub,
            plb,
            pub,
            options={"display": "off", "random_seed": 0},
        )
        b.logging_action = []
        b._init_mesh_()
    lg = b.function_logger
    return (
        lg.func_count,
        lg.Xn + 1 - 1,
        b.optim_state["eff_starting_points"],
        b.optim_state["uncertainty_handling_level"],
        b.options["max_fun_evals"],
    )


def rounded(n):
    return 2 ** int(np.ceil(np.log2(n)))


print(
    "D | level 0: evals in init (x0, noise test, design), design rows | "
    "rounding alone | MATLAB Ninit || level 1: evals, design rows | "
    "rounding alone | MATLAB",
    flush=True,
)
for D in range(1, 21):
    f0, d0, e0, l0, m0 = init_only(sphere, D)
    f1, d1, e1, l1, m1 = init_only(Noisy(D), D)
    print(
        f"{D:2d} | L{l0}: {f0:3d} evals, design {d0:3d} (share of "
        f"{m0} budget {d0 / m0:.3%}) | {rounded(D):3d} | {D:3d} || "
        f"L{l1}: {f1:3d} evals, design {d1:3d} | {rounded(20):3d} | "
        f"{min(max(20, D), m1 - 1):3d}",
        flush=True,
    )

print(
    "\nfun_eval_start set by the user, at D = 8 and 16 (init_sobol sizes)",
    flush=True,
)
from pybads.init_functions.init_sobol import init_sobol  # noqa: E402

for D in (4, 8, 16):
    sizes = []
    for fes in range(1, 2 * D + 2):
        u, _ = init_sobol(
            np.zeros(D), None, None, -np.ones((1, D)), np.ones((1, D)), fes
        )
        sizes.append(f"{fes}:{u.shape[0]}")
    print(f"D={D}: " + " ".join(sizes), flush=True)
