"""Sphere D = 1 with an interior optimum (x0 = 3, bounds +-20, plausible +-5): repeats by stage."""
import numpy as np
import vhdr  # noqa

exec(
    open("v_repeats.py")
    .read()
    .split("for D in (1, 2):")[0]
    .split("import vhdr  # noqa")[1]
)
for seed in range(4):
    run(
        f"sphere D=1 interior seed={seed}",
        sphere,
        np.full(1, 3.0),
        np.full(1, -20.0),
        np.full(1, 20.0),
        np.full(1, -5.0),
        np.full(1, 5.0),
        {"random_seed": seed, "max_fun_evals": 200},
    )
