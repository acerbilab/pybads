"""Whole optimizations: the problems of MATLAB BADS's `runtest.m`, and a few
more.

Each run is seeded: the `random_seed` of BADS is `SEED`, and a noisy target
draws its noise from a generator seeded with `NOISE_SEED`. On another
platform, or with other versions of NumPy, SciPy or gpyreg, a seeded run
can follow another trajectory, so a tolerance has to hold over seeds, not
only at `SEED`. Each is set from the errors of its test over seeds 0-99
(`SEED = s`, `NOISE_SEED = s + 1000`): ten times the largest error, rounded
up to 1, 2 or 5 times a power of ten, or the tolerance of `runtest.m` if
that is lower. The section "The seed sweep behind the tolerances" of
`dev/results/2026-09-23-codebase-survey.md` records them.

`dev/scripts/tolerance_sweep.py` measures the errors again, as a change
that moves results calls for. It relies on two properties of this module:
the tests read `SEED` and `NOISE_SEED` when called, and each passes its
tolerance to `run_bads` as `tol_err`, which the script disables."""

import numpy as np

from pybads.bads import BADS

SEED = 0
NOISE_SEED = 1000


def get_test_opt_conf(D=3):
    x0 = np.ones((1, D)) * 4
    LB = -100 * np.ones(D)  # Lower bound
    UB = 100 * np.ones(D)  # Upper bound
    PLB = -8 * np.ones(D)  # Plausible lower bound
    PUB = 12 * np.ones(D)  # Plausible upper bound
    return D, x0, LB, UB, PLB, PUB


def run_bads(
    fun,
    x0,
    LB,
    UB,
    PLB,
    PUB,
    tol_err,
    f_min,
    oracle_fun=None,
    non_box_cons=None,
    uncertainty_handling=False,
    max_fun_evals=None,
):
    options = {}
    options["display"] = "full"  # debug_flag = True
    options["random_seed"] = SEED

    if uncertainty_handling > 0:
        options["uncertainty_handling"] = True
        options["max_fun_evals"] = (
            200 if max_fun_evals is None else max_fun_evals
        )
        if uncertainty_handling > 1:
            options["specify_target_noise"] = True
    else:
        options["max_fun_evals"] = (
            100 if max_fun_evals is None else max_fun_evals
        )

    optimize_result = BADS(
        fun=fun,
        x0=x0,
        lower_bounds=LB,
        upper_bounds=UB,
        plausible_lower_bounds=PLB,
        plausible_upper_bounds=PUB,
        non_box_cons=non_box_cons,
        options=options,
    ).optimize()
    x = optimize_result["x"]
    fval = optimize_result["fval"]

    if oracle_fun is None:
        err = np.abs(fval - f_min).item()
    else:
        fval_true = oracle_fun(x)
        err = np.abs(fval_true - f_min).item()
    assert (
        err < tol_err
    ), f"Error {err} is not smaller than tolerance {tol_err} when optimizing {fun.__name__}."

    return optimize_result, err


def ellipsoid(x):
    return np.sum((np.atleast_2d(x) / np.arange(1, len(x) + 1) ** 2) ** 2)


def sphere(x):
    return np.sum(np.atleast_2d(x) ** 2, axis=1)


def test_ellipsoid_opt():
    D, x0, LB, UB, PLB, PUB = get_test_opt_conf()
    run_bads(ellipsoid, x0, LB, UB, PLB, PUB, tol_err=1e-3, f_min=0.0)


def test_univariate_input_and_opt():
    """Scalar start point and plausible bounds, no hard bounds, and noise that
    BADS detects by itself, at the default budget of 500 evaluations."""
    rng = np.random.default_rng(NOISE_SEED)
    rfn = lambda x: x**2 + 3.2 + rng.normal(scale=0.1)
    optimize_result, _ = run_bads(
        rfn,
        3,
        None,
        None,
        -5,
        5,
        tol_err=2e-2,
        f_min=3.2,
        oracle_fun=lambda x: x**2 + 3.2,
        max_fun_evals=500,
    )
    assert optimize_result["x"].size == 1


def test_1D_opt_ndarray():
    """Test optimization with 1D inputs and nd arrays"""
    fun = lambda x: np.atleast_2d(x) ** 2
    x0 = np.array([[2.0]])
    LB = np.array([[-10.0]])
    UB = np.array([[10.0]])
    PLB = np.array([[-5.0]])
    PUB = np.array([[5.0]])
    run_bads(fun, x0, LB, UB, PLB, PUB, tol_err=5e-6, f_min=0.0)


def test_1D_opt_1darray():
    """Test optimization with 1D inputs and 1D arrays"""
    fun = lambda x: np.atleast_2d(x) ** 2
    x0 = np.array([2.0])
    LB = np.array([-10.0])
    UB = np.array([10.0])
    PLB = np.array([-5.0])
    PUB = np.array([5.0])
    run_bads(fun, x0, LB, UB, PLB, PUB, tol_err=5e-6, f_min=0.0)


def test_1D_opt_scalar():
    """Test optimization with scalar inputs"""
    fun = lambda x: np.atleast_2d(x) ** 2
    x0 = 2.0
    LB = -10.0
    UB = 10.0
    PLB = -5.0
    PUB = 5.0
    run_bads(fun, x0, LB, UB, PLB, PUB, tol_err=5e-6, f_min=0.0)


def test_high_dim_opt():
    D, x0, LB, UB, PLB, PUB = get_test_opt_conf(D=60)
    run_bads(
        ellipsoid,
        x0,
        LB,
        UB,
        PLB,
        PUB,
        tol_err=1.0,
        f_min=0.0,
        max_fun_evals=200,
    )


def test_sphere_opt():
    """Sphere under the constraint `x1 + x2 >= sqrt(2)`, whose minimum, 1, lies
    on the boundary of the feasible region at `(sqrt(2)/2, sqrt(2)/2, 0)`."""
    D, x0, LB, UB, PLB, PUB = get_test_opt_conf()

    def non_box_cons(x):
        """True where a point violates the constraint."""
        x = np.atleast_2d(x)
        return x[:, 0] + x[:, 1] < np.sqrt(2)

    optimize_result, _ = run_bads(
        sphere,
        x0,
        LB,
        UB,
        PLB,
        PUB,
        tol_err=2e-3,
        f_min=1.0,
        non_box_cons=non_box_cons,
    )
    assert not np.any(non_box_cons(optimize_result["x"]))


def test_noisy_sphere_opt():
    D, x0, LB, UB, PLB, PUB = get_test_opt_conf()
    rng = np.random.default_rng(NOISE_SEED)
    fun = lambda x: sphere(x) + rng.standard_normal()  # Noisy objective
    run_bads(
        fun,
        x0,
        LB,
        UB,
        PLB,
        PUB,
        tol_err=1.0,
        f_min=0.0,
        oracle_fun=sphere,
    )


def test_small_noisy_func():
    rng = np.random.default_rng(NOISE_SEED)

    def noisy_sphere(x, sigma=1.0):
        """Simple quadratic function with added noise."""
        x_2d = np.atleast_2d(x)
        f = np.sum(x_2d**2, axis=1)
        noise = 1e-4 * sigma * rng.normal(size=x_2d.shape[0])
        return f + noise

    x0 = np.array([-3, -3, -3])
    LB = np.array([-5, -5, -5])
    UB = np.array([5, 5, 5])
    PLB = np.array([-2, -2, -2])
    PUB = np.array([2, 2, 2])

    run_bads(
        noisy_sphere,
        x0,
        LB,
        UB,
        PLB,
        PUB,
        tol_err=1e-2,
        f_min=0.0,
        oracle_fun=sphere,
        uncertainty_handling=1,
        max_fun_evals=300,
    )


def he_noisy_sphere(rng):
    """Sphere with heteroskedastic noise, returning the noise's standard
    deviation with the value."""

    def fun(x):
        y = sphere(x)
        s = 2 + 1 * np.sqrt(y)
        y = y + s * rng.standard_normal()
        return y, s

    return fun


def test_he_noisy_sphere_opt():
    D, x0, LB, UB, PLB, PUB = get_test_opt_conf()
    fun = he_noisy_sphere(np.random.default_rng(NOISE_SEED))
    run_bads(
        fun,
        x0,
        LB,
        UB,
        PLB,
        PUB,
        tol_err=1.0,
        f_min=0.0,
        oracle_fun=sphere,
        uncertainty_handling=2,
    )
