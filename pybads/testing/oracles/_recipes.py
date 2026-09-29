"""The runs whose states the oracle fixtures hold.

Each recipe is a short seeded PyBADS run on a small synthetic target, with
its bounds, its options and the point at which its state is taken: the
start of the first search step (or poll step) that begins with at least
``capture_evals`` evaluations. The first poll step of a run comes before its
first search, and receives the GP of the initialization.
``dev/scripts/make_oracle_fixtures.py`` runs the recipes; the tests never
do, since they rebuild each state from its fixture. The targets use
elementwise arithmetic only, so that a recipe's run does not depend on BLAS
through its target.

The non-box constraint of ``nonbox_D3`` is registered by name in
``NON_BOX_CONS``: the oracles that take a run's constraint look it up by the
name that the fixture records.
"""

import numpy as np


def disk_violation(X):
    """The non-box constraint of ``nonbox_D3``: a point violates it (True)
    when its first two coordinates lie outside the disk of radius
    ``sqrt(2)``."""
    X = np.atleast_2d(X)
    return X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1] > 2.0


NON_BOX_CONS = {"disk_violation": disk_violation}


def _sq_dist(x, center):
    d = np.asarray(x, dtype=float).ravel() - center
    return float(np.sum(d * d))


class Recipe:
    """One run: ``target`` names a target of :func:`make_target`, with the
    keyword arguments ``target_args``; ``bounds`` are the lower, upper,
    plausible lower and plausible upper bounds; ``options`` are the user
    options of the run, which always set ``random_seed`` and
    ``uncertainty_handling``; ``capture_step`` is ``"search"`` or
    ``"poll"``."""

    def __init__(
        self,
        name,
        target,
        target_args,
        x0,
        bounds,
        options,
        capture_evals,
        note,
        non_box_cons=None,
        capture_step="search",
    ):
        self.name = name
        self.target = target
        self.target_args = dict(target_args)
        self.x0 = np.asarray(x0, dtype=float)
        self.bounds = tuple(np.asarray(b, dtype=float) for b in bounds)
        self.options = dict(options)
        self.capture_evals = int(capture_evals)
        self.note = note
        self.non_box_cons = non_box_cons
        if capture_step not in ("search", "poll"):
            raise ValueError(f"capture_step {capture_step!r}")
        self.capture_step = capture_step
        self.D = self.x0.size


def make_target(recipe):
    """A fresh target function of ``recipe``, its noise generator (if any)
    seeded anew, so that each run of a recipe repeats."""
    args = recipe.target_args
    center = np.asarray(args["center"], dtype=float)
    if recipe.target == "sphere":
        return lambda x: _sq_dist(x, center)
    if recipe.target == "ellipsoid":
        # Axis weights 1, 10, 100 and a cross term that tilts the axes
        def ellipsoid(x):
            d = np.asarray(x, dtype=float).ravel() - center
            return float(
                d[0] * d[0]
                + 10.0 * d[1] * d[1]
                + 100.0 * d[2] * d[2]
                + 3.0 * d[0] * d[1]
            )

        return ellipsoid
    if recipe.target == "logsphere":
        # A sphere in the logs of the variables named by `log_vars`
        log_vars = np.asarray(args["log_vars"], dtype=bool)

        def logsphere(x):
            x = np.asarray(x, dtype=float).ravel().copy()
            c = center.copy()
            x[log_vars] = np.log(x[log_vars])
            c[log_vars] = np.log(c[log_vars])
            return _sq_dist(x, c)

        return logsphere
    if recipe.target == "noisy_sphere":
        noise = np.random.default_rng(args["noise_seed"])
        sd = float(args["noise_sd"])
        return lambda x: _sq_dist(x, center) + sd * noise.standard_normal()
    if recipe.target == "sphere_with_sd":
        # Heteroskedastic noise whose SD the target returns
        noise = np.random.default_rng(args["noise_seed"])

        def sphere_with_sd(x):
            x = np.asarray(x, dtype=float).ravel()
            sd = 0.5 + 0.2 * abs(x[0])
            return _sq_dist(x, center) + sd * noise.standard_normal(), sd

        return sphere_with_sd
    raise ValueError(f"unknown target {recipe.target!r}")


def _box(D, lb, ub, plb, pub):
    return tuple(np.full(D, v, dtype=float) for v in (lb, ub, plb, pub))


RECIPES = {
    r.name: r
    for r in [
        Recipe(
            "sphere_D2_init",
            "sphere",
            {"center": [0.7, -1.3]},
            [2.5, 2.0],
            _box(2, -5.0, 5.0, -3.0, 3.0),
            {"uncertainty_handling": False, "random_seed": 1},
            1,
            "deterministic, D = 2, the first poll step: the GP of the "
            "initialization, with _gp_hyp's unit scales, and no hedge yet",
            capture_step="poll",
        ),
        Recipe(
            "ellipsoid_D3",
            "ellipsoid",
            {"center": [0.7, -1.3, 0.4]},
            [2.5, 2.0, -1.0],
            _box(3, -5.0, 5.0, -3.0, 3.0),
            {"uncertainty_handling": False, "random_seed": 3},
            60,
            "deterministic, D = 3, a tilted ellipsoid after several refits, "
            "whose GP's training set leaves out some logged points",
        ),
        Recipe(
            "noisy_sphere_D3",
            "noisy_sphere",
            {"center": [0.7, -1.3, 0.4], "noise_sd": 1.0, "noise_seed": 7},
            [2.5, 2.0, -1.0],
            _box(3, -5.0, 5.0, -3.0, 3.0),
            {"uncertainty_handling": True, "random_seed": 4},
            70,
            "noise inferred (uncertainty handling level 1), D = 3",
        ),
        Recipe(
            "target_noise_D2",
            "sphere_with_sd",
            {"center": [0.7, -1.3], "noise_seed": 9},
            [2.5, 2.0],
            _box(2, -5.0, 5.0, -3.0, 3.0),
            {
                "uncertainty_handling": True,
                "specify_target_noise": True,
                "random_seed": 5,
            },
            40,
            "the target returns its noise SD (level 2), D = 2: the GP "
            "carries the noise variances",
        ),
        Recipe(
            "log_D3",
            "logsphere",
            {"center": [0.5, 2.0, 0.3], "log_vars": [True, True, False]},
            [5.0, 0.5, 1.0],
            (
                np.array([1e-3, 1e-3, -5.0]),
                np.array([1e3, 1e3, 5.0]),
                np.array([1e-2, 1e-1, -2.0]),
                np.array([1e2, 10.0, 2.0]),
            ),
            {"uncertainty_handling": False, "random_seed": 6},
            40,
            "two log-transformed variables and a linear one, D = 3",
        ),
        Recipe(
            "nonbox_D3",
            "sphere",
            {"center": [1.2, 1.1, -0.3]},
            [-1.0, 0.5, 1.0],
            _box(3, -3.0, 3.0, -2.0, 2.0),
            {"uncertainty_handling": False, "random_seed": 8},
            40,
            "a non-box constraint (a disk in the first two variables) "
            "that excludes the target's minimum, D = 3",
            non_box_cons="disk_violation",
        ),
    ]
}
