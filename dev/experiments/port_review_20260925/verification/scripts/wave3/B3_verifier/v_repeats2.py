"""Repeats in noisy runs (levels 1 and 2), PyBADS contraints_check only."""
import numpy as np
import vhdr  # noqa

exec(
    open("v_repeats.py")
    .read()
    .split("for D in (1, 2):")[0]
    .split("import vhdr  # noqa")[1]
)


def hetero(seed, w):
    rng = np.random.default_rng(seed)

    def f(x):
        x = np.atleast_2d(x)
        fx = float(np.sum(w * x**2))
        sd = 1.0 + fx**0.5
        return fx + sd * rng.standard_normal(), sd

    return f


def noisy_sphere(seed):
    rng = np.random.default_rng(seed)

    def f(x):
        return float(np.sum(np.atleast_2d(x) ** 2)) + rng.standard_normal()

    return f


for seed in (1, 2, 3):
    D = 3
    run(
        f"hetero ellipsoid D=3 L2 seed={seed}",
        hetero(seed, np.array([1.0, 10.0, 100.0])),
        np.full(D, 2.0),
        np.full(D, -10.0),
        np.full(D, 10.0),
        np.full(D, -3.0),
        np.full(D, 3.0),
        {
            "random_seed": seed,
            "max_fun_evals": 200,
            "specify_target_noise": True,
        },
    )
for seed in (0, 1):
    D = 3
    run(
        f"noisy sphere D=3 L1 seed={seed}",
        noisy_sphere(seed),
        np.full(D, 2.0),
        np.full(D, -10.0),
        np.full(D, 10.0),
        np.full(D, -3.0),
        np.full(D, 3.0),
        {
            "random_seed": seed,
            "max_fun_evals": 200,
            "uncertainty_handling": True,
        },
    )
