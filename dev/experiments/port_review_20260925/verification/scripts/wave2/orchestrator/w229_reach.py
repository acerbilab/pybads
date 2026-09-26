"""W2-29 where it binds: the B1/B2 verifier's cases that reach the iteration
where only MATLAB tested the accelerated mesh reduction (v_accel.py: a
sphere started at its minimum, Rosenbrock from [-1, 1]), seeds 0-29, at the
PyBADS on PYTHONPATH. Prints one line per run."""

import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
cases = [
    (
        "sphere_D2_x0=0",
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        np.zeros(2),
        0.0,
    ),
    (
        "rosen_D2_x0=[-1,1]",
        lambda x: float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2),
        np.array([-1.0, 1.0]),
        0.0,
    ),
]
for name, f, x0, fmin in cases:
    for seed in range(30):
        D = x0.size
        b = BADS(
            f,
            x0.copy(),
            np.full(D, -5.0),
            np.full(D, 5.0),
            np.full(D, -2.0),
            np.full(D, 2.0),
            options=dict(display="off", random_seed=seed, max_fun_evals=200),
        )
        r = b.optimize()
        print(
            f"{name} {seed} err {r['fval'] - fmin:.3e} evals {r['func_count']} "
            f"iters {r['iterations']} mesh {r['mesh_size']:.3g}"
        )
