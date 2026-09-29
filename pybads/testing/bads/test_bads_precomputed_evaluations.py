"""Evaluations made before the run (`precomputed_evaluations`): their checks
when `BADS` is created, their import into the function log and the GP's
training set, which leaves them out of the count of evaluations and of the
choice of the first incumbent, and their counts in the result."""

import numpy as np
import pytest

import pybads.bads.gaussian_process_train as gpt
from pybads import BADS

D = 3
LB, UB = -10 * np.ones(D), 10 * np.ones(D)
PLB, PUB = -5 * np.ones(D), 5 * np.ones(D)


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def _sphere_with_sd(x):
    return _sphere(x), 0.5


def _evaluations(n=12, seed=1):
    X = np.random.default_rng(seed).uniform(-3, 3, (n, D))
    y = np.array([_sphere(x) for x in X])
    return X, y


def _make_bads(evaluations, fun=_sphere, non_box_cons=None, **options):
    return BADS(
        fun,
        np.ones(D) * 4,
        LB,
        UB,
        PLB,
        PUB,
        non_box_cons=non_box_cons,
        options={
            "display": "off",
            "random_seed": 0,
            "max_fun_evals": 30,
            **options,
        },
        precomputed_evaluations=evaluations,
    )


def test_evaluations_enter_the_log_but_not_the_count():
    X, y = _evaluations()
    bads = _make_bads((X, y))
    logger = bads.function_logger
    assert logger.func_count == 0
    assert logger.Xn + 1 == len(X)
    assert np.allclose(logger.X_orig[: len(X)], X, rtol=1e-12, atol=1e-12)
    assert np.all(logger.X[: len(X)] == bads.var_transf(X))
    assert np.all(logger.Y[: len(X), 0] == y)
    assert np.all(logger.n_evals[: len(X)] == 1)

    result = bads.optimize()
    # The budget is the run's own: every evaluation of it is fresh, and each
    # but the noise test's adds a row
    assert result["func_count"] == 30
    assert bads.function_logger.Xn + 1 == len(X) + 30 - 1
    assert result["precomputed_observations"] == len(X)
    assert result["precomputed_locations"] == len(X)


@pytest.mark.parametrize("evaluations", [None, (np.empty((0, D)), [])])
def test_without_evaluations_the_result_has_no_counts(evaluations):
    bads = _make_bads(evaluations)
    assert bads.optim_state["precomputed_observations"] == 0
    assert bads.function_logger.Xn == -1
    result = bads.optimize()
    assert "precomputed_observations" not in result
    assert "precomputed_locations" not in result


def test_the_first_incumbent_is_never_an_evaluation_made_before():
    """The run starts from `x0` and its initial design, as MATLAB BADS does
    (`evalinitmesh.m:120-123`), even when a point given has a lower value
    than all of them."""
    X, y = _evaluations()
    X[0] = 0.01
    y[0] = _sphere(X[0])
    bads = _make_bads((X, y))
    bads._init_optimization_()
    assert bads._init_incumbent_row >= len(X)
    assert bads.yval > y[0]
    assert not np.any(np.all(bads.var_transf(X) == bads.u, axis=1))


def test_the_gp_holds_the_evaluations_made_before():
    """The first GP is fitted on the start and the initial design, however
    many evaluations were given; they enter the GP at its first local
    rebuild, among the neighbours of the incumbent (here all of them, fewer
    than `n_train_min`)."""
    X, y = _evaluations()
    bads = _make_bads((X, y))
    gp, _, _, _ = bads._init_optimization_()
    assert gp.X.shape[0] == bads.optim_state["eff_starting_points"] == 5
    assert np.all(gp.X == bads.function_logger.X[len(X) : len(X) + 5])

    bads = _make_bads((X, y))
    bads.optimize()
    gp = bads.iteration_history["gp"][0]
    for u in bads.var_transf(X):
        assert np.any(np.all(gp.X == u, axis=1))


def test_the_first_fit_leaves_out_many_evaluations():
    bads = _make_bads(_evaluations(n=1000), max_fun_evals=100)
    gp, _, _, _ = bads._init_optimization_()
    assert gp.X.shape[0] == 5


def test_the_training_schedule_leaves_them_out(monkeypatch):
    """The number of starts of the GP's first fit follows the run's own
    budget: many evaluations given do not make it the final one."""
    seen = []
    original = gpt._get_gp_training_options

    def spy(*args, **kwargs):
        gp_train = original(*args, **kwargs)
        seen.append(gp_train["init_N"])
        return gp_train

    monkeypatch.setattr(gpt, "_get_gp_training_options", spy)
    bads = _make_bads(_evaluations(n=300), max_fun_evals=100)
    bads._init_optimization_()
    assert bads.optim_state["eff_starting_points"] == 5
    assert seen[0] == bads.options["gp_train_n_init"]


def test_the_arrays_are_copied():
    X, y = _evaluations()
    X_given, y_given = X.copy(), y.copy()
    bads = _make_bads((X, y))
    assert np.all(X == X_given) and np.all(y == y_given)
    X[:] = 0.0
    y[:] = -1.0
    assert np.all(bads.function_logger.Y[: len(y), 0] == y_given)


def test_lists_are_taken():
    X, y = _evaluations(n=3)
    bads = _make_bads([X.tolist(), y.tolist()])
    assert bads.function_logger.Xn + 1 == 3


def test_log_transformed_variables():
    """A variable on a log scale (positive bounds, `pub/plb >= 10`) is
    transformed as the run's points are."""
    X = np.array([[1.0, 2.0, 30.0], [5.0, 0.5, 70.0]])
    y = np.array([_sphere(x) for x in X])
    bads = BADS(
        _sphere,
        np.array([2.0, 2.0, 2.0]),
        np.array([0.01, 0.01, 0.01]),
        np.array([100.0, 100.0, 100.0]),
        np.array([0.1, 0.1, 0.1]),
        np.array([10.0, 10.0, 10.0]),
        options={"display": "off", "random_seed": 0, "max_fun_evals": 20},
        precomputed_evaluations=(X, y),
    )
    assert np.all(bads.var_transf.apply_log_t)
    assert np.all(bads.function_logger.X[:2] == bads.var_transf(X))
    assert np.allclose(bads.function_logger.X_orig[:2], X, rtol=1e-12)
    bads.optimize()


def test_a_point_given_twice_is_kept_once_without_noise():
    X, y = _evaluations(n=4)
    X[2, 0] = 0.0
    X_minus_zero = X[2].copy()
    X_minus_zero[0] = -0.0
    X = np.vstack((X, X[1], X_minus_zero))
    y = np.append(y, [y[1], np.nextafter(np.nextafter(y[2], 99), 99)])
    bads = _make_bads((X, y))
    assert bads.function_logger.Xn + 1 == 4
    assert bads.optim_state["precomputed_observations"] == 6
    assert bads.optim_state["precomputed_locations"] == 4
    assert bads.optim_state["precomputed_n_evals"] == 4


@pytest.mark.parametrize("big", [np.finfo(float).max, -np.finfo(float).max])
def test_two_values_at_one_point_are_compared_up_to_the_largest_float(big):
    """At the largest float64, whose spacing is infinite, the values are
    compared within four spacings below it."""
    X = np.zeros((2, D))
    with pytest.raises(ValueError, match="Rows 0 and 1"):
        _make_bads((X, np.array([big, 1.0])))
    with pytest.raises(ValueError, match="Rows 0 and 1"):
        _make_bads((X, np.array([big, -big])))
    agreeing = np.array([big, np.nextafter(big, 0.0)])
    assert _make_bads((X, agreeing)).function_logger.Xn == 0


@pytest.mark.parametrize("uncertainty_handling", [None, False])
def test_two_values_at_one_point_are_refused_without_noise(
    uncertainty_handling,
):
    X, y = _evaluations(n=4)
    X = np.vstack((X, X[1]))
    y = np.append(y, y[1] + 1e-6)
    with pytest.raises(ValueError, match=r"Rows 1 and 4 .*uncertainty_hand"):
        _make_bads((X, y), uncertainty_handling=uncertainty_handling)


def test_with_inferred_noise_each_repeat_is_a_row():
    X, y = _evaluations(n=4)
    X = np.vstack((X, X[1], X[1]))
    y = np.append(y, [y[1] + 0.3, y[1] - 0.2])
    bads = _make_bads((X, y), uncertainty_handling=True, max_fun_evals=50)
    assert bads.function_logger.Xn + 1 == 6
    assert np.all(bads.function_logger.Y[4:6, 0] == y[4:6])
    assert bads.optim_state["precomputed_locations"] == 4
    assert bads.optim_state["precomputed_n_evals"] == 6
    result = bads.optimize()
    assert result["func_count"] <= 50
    assert result["precomputed_observations"] == 6


def test_with_target_noise_repeats_are_merged():
    """With `specify_target_noise`, the SDs are required, and a repeated
    point is merged into its row by precision weighting, as the logger
    merges an evaluation of its own."""
    X, y = _evaluations(n=3)
    X = np.vstack((X, X[0]))
    y = np.append(y, y[0] + 1.0)
    y_sd = np.array([1.0, 0.5, 0.5, 2.0])
    bads = _make_bads(
        (X, y, y_sd),
        fun=_sphere_with_sd,
        specify_target_noise=True,
        max_fun_evals=50,
    )
    logger = bads.function_logger
    assert logger.Xn + 1 == 3
    tau = 1 / y_sd[[0, 3]] ** 2
    assert np.isclose(logger.Y[0, 0], np.sum(tau * y[[0, 3]]) / np.sum(tau))
    assert np.isclose(logger.S[0, 0], 1 / np.sqrt(np.sum(tau)))
    assert logger.n_evals[0] == 2
    assert np.all(logger.S[1:3, 0] == y_sd[1:3])
    assert bads.optim_state["precomputed_n_evals"] == 4
    result = bads.optimize()
    assert result["func_count"] <= 50
    assert result["precomputed_locations"] == 3


def test_points_that_satisfy_non_box_cons_are_taken():
    X, y = _evaluations()
    bads = _make_bads(
        (X, y), non_box_cons=lambda x: np.sum(x**2, axis=1) > 100
    )
    assert bads.function_logger.Xn + 1 == len(X)


def test_points_on_the_hard_bounds_are_taken():
    X, y = _evaluations(n=2)
    X[0] = LB
    X[1] = UB
    y = np.array([_sphere(x) for x in X])
    assert _make_bads((X, y)).function_logger.Xn + 1 == 2


_X, _Y = _evaluations(n=4)


def _with(i, value):
    X, y = _X.copy(), _Y.copy()
    evaluations = [X, y]
    if i == "X":
        return (value, y)
    evaluations[i] = value
    return tuple(evaluations)


@pytest.mark.parametrize(
    "evaluations, message",
    [
        (_X, "must be a tuple"),
        ((_X,), "must be a tuple"),
        ((_X, _Y, np.ones(4), np.ones(4)), "must be a tuple"),
        ({"X": _X, "Y": _Y}, "must be a tuple"),
        (_with("X", _X[:, :2]), r"points X .* shape \(N, 3\), not \(4, 2\)"),
        (_with("X", _X[0]), r"points X .* shape \(N, 3\), not \(3,\)"),
        (_with("X", 1.0), r"points X .* shape \(N, 3\), not \(\)"),
        (_with(1, _Y[:3]), r"values y .* shape \(4,\), not \(3,\)"),
        (_with(1, _Y[:, None]), r"values y .* shape \(4,\), not \(4, 1\)"),
        (_with(1, np.array([1.0, np.nan, 2.0, np.inf])), r"rows \[1, 3\]"),
        (
            _with("X", np.vstack((_X[:3], [[0.0, np.inf, 0.0]]))),
            r"points X .* finite; rows \[3\]",
        ),
        (_with(1, _Y.astype(complex)), "values y .* real numbers"),
        (_with(1, ["1", "2", "3", "4"]), "values y .* real numbers"),
        (_with(1, [1.0, [2.0], 3.0, 4.0]), "values y .* an array"),
        ((_X, _Y, np.ones(4)), "require.*specify_target_noise"),
        (
            _with("X", np.vstack((_X[:3], [[0.0, 11.0, 0.0]]))),
            r"within the hard bounds; rows \[3\]",
        ),
        (
            _with("X", np.vstack((_X[:3], [[-10.5, 0.0, 0.0]]))),
            r"within the hard bounds; rows \[3\]",
        ),
    ],
)
def test_refused(evaluations, message):
    with pytest.raises(ValueError, match=message):
        _make_bads(evaluations)


@pytest.mark.parametrize(
    "y_sd, message",
    [
        (None, r"must hold the noise SDs"),
        (np.array([1.0, 0.0, 1.0, -1.0]), r"positive; rows \[1, 3\]"),
        (np.array([1.0, np.nan, 1.0, 1.0]), r"finite; rows \[1\]"),
        (np.ones(3), r"shape \(4,\), not \(3,\)"),
    ],
)
def test_refused_with_target_noise(y_sd, message):
    evaluations = (_X, _Y) if y_sd is None else (_X, _Y, y_sd)
    with pytest.raises(ValueError, match=message):
        _make_bads(evaluations, fun=_sphere_with_sd, specify_target_noise=True)


def test_points_beyond_float64_in_the_transformed_space_are_refused():
    """With infinite hard bounds, a point far outside a narrow plausible box
    maps beyond float64."""
    X = np.zeros((2, D))
    X[1, 0] = 1e308
    with pytest.raises(ValueError, match=r"finite coordinates.*rows \[1\]"):
        BADS(
            _sphere,
            np.zeros(D),
            -np.inf * np.ones(D),
            np.inf * np.ones(D),
            -0.01 * np.ones(D),
            0.01 * np.ones(D),
            options={"display": "off"},
            precomputed_evaluations=(X, np.zeros(2)),
        )


def test_the_first_sd_with_target_noise_is_the_incumbent_s():
    """With `specify_target_noise`, the run's first `fsd` is the SD of its
    first incumbent, not of a lower evaluation given before the run."""
    X = np.array([[0.01, 0.01, 0.01], [1.0, 1.0, 1.0]])
    bads = _make_bads(
        (X, np.array([3e-4, 3.0]), np.array([7.0, 7.0])),
        fun=_sphere_with_sd,
        specify_target_noise=True,
    )
    bads._init_optimization_()
    assert bads._init_incumbent_row >= 2
    assert bads.fsd == 0.5


def test_points_that_violate_non_box_cons_are_refused():
    X = _X.copy()
    X[:, 0] = [-2.5, 1.0, -2.9, 0.0]
    with pytest.raises(ValueError, match=r"non_box_cons; rows \[0, 2\]"):
        _make_bads((X, _Y), non_box_cons=lambda x: x[:, 0] < -2)


def test_final_samples_leave_the_log_as_it_is():
    """The final samples of a noisy run are not recorded, and leave the
    count of evaluations of the incumbent's row as it was, as MATLAB BADS's
    `funlogger(..., 'single')` does."""
    noise = np.random.default_rng(0)
    bads = BADS(
        lambda x: _sphere(x) + noise.standard_normal(),
        np.ones(D) * 4,
        LB,
        UB,
        PLB,
        PUB,
        options={
            "display": "off",
            "random_seed": 0,
            "max_fun_evals": 80,
            "uncertainty_handling": True,
        },
    )
    result = bads.optimize()
    n_samples = bads.options["noise_final_samples"]
    assert result["yval_vec"].size == n_samples == 10
    logger = bads.function_logger
    assert np.all(logger.n_evals[logger.X_flag] == 1)
    assert logger.Xn + 1 == result["func_count"] - n_samples
