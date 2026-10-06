"""F2: the initial design is capped at max_fun_evals - 1 before it is rounded
up to a power of two, and the noise test is not counted."""
import gpyreg
import numpy as np

import pybads
from pybads import BADS

print(pybads.__file__)
print(gpyreg.__file__)


def sphere(x):
    return float(np.sum(np.ravel(x) ** 2))


for D, mfe, extra in [
    (2, 5, {}),
    (3, 5, {}),
    (5, 7, {}),
    (2, 5, dict(uncertainty_handling=False)),
    (4, 9, dict(uncertainty_handling=False)),
]:
    x0 = np.full((1, D), 0.7)
    b = BADS(
        sphere,
        x0,
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options=dict(max_fun_evals=mfe, random_seed=0, display="off", **extra),
    )
    r = b.optimize()
    print(
        f"D={D} max_fun_evals={mfe} {extra}: func_count={r['func_count']}, iterations={r['iterations']}, msg={r['message']!r}"
    )

# noisy: at least 20 design points, rounded to 32
noise_rng = np.random.default_rng(1)


def noisy(x):
    return float(np.sum(np.ravel(x) ** 2) + 0.3 * noise_rng.standard_normal())


for mfe in (25, 38):
    D = 2
    b = BADS(
        noisy,
        np.full((1, D), 0.7),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options=dict(
            max_fun_evals=mfe,
            random_seed=0,
            display="off",
            uncertainty_handling=True,
        ),
    )
    r = b.optimize()
    print(
        f"noisy D=2 max_fun_evals={mfe}: func_count={r['func_count']}, eff_starting_points={b.optim_state['eff_starting_points']}, "
        f"options max_fun_evals after={b.options['max_fun_evals']}, noise_final_samples after={b.options['noise_final_samples']}, "
        f"yval_vec={r['yval_vec']}, fsd={r['fsd']}, iterations={r['iterations']}"
    )
