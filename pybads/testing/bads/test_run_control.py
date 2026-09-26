"""The output function, the iteration count and the shortest runs, as MATLAB
BADS has them: `output_fcn(x, optim_state, state)` is called at the start
(`"init"`), after each poll (`"iter"`) and at the end (`"done"`), and stops
the run when it returns a true value; `iterations` counts from 1."""

import numpy as np
import pytest

from pybads import BADS

D = 3


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def _make_bads(fun=_sphere, **options):
    opts = {"display": "off", "max_fun_evals": 80, "random_seed": 3}
    opts.update(options)
    return BADS(
        fun,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=opts,
    )


class _Recorder:
    """An output function that records its calls and returns `stop_at(n)`,
    `n` the number of calls so far."""

    def __init__(self, stop_at=lambda n: False):
        self.calls = []
        self.stop_at = stop_at

    def __call__(self, x, optim_state, state):
        self.calls.append((np.array(x, dtype=float), optim_state, state))
        return self.stop_at(len(self.calls))

    @property
    def states(self):
        return [state for _, _, state in self.calls]


def test_output_fcn_calls():
    """One call at the start, one after each poll and one at the end, each
    with the incumbent in the original space and a copy of `optim_state`;
    the last receives the returned point. An iteration is a round of
    searches that a poll ends, and a run can end within one, so the polls
    number `iterations` or one less."""
    record = _Recorder()
    bads = _make_bads(output_fcn=record, max_iter=6)
    result = bads.optimize()
    n_polls = record.states.count("iter")
    assert record.states == ["init"] + ["iter"] * n_polls + ["done"]
    assert result["iterations"] - 1 <= n_polls <= result["iterations"]
    assert n_polls >= 3
    for x, optim_state, _ in record.calls:
        assert x.size == D
        assert isinstance(optim_state, dict)
        assert optim_state is not bads.optim_state
    assert np.array_equal(record.calls[-1][0].ravel(), result["x"].ravel())
    # optim_state["iter"] counts the iterations from 0
    polled = [s["iter"] for _, s, state in record.calls if state == "iter"]
    assert polled == list(range(n_polls))


def test_output_fcn_that_changes_nothing_leaves_the_run_unchanged():
    """An output function that never stops the run, and alters the copy of
    `optim_state` it receives, gives the result of a run without one."""

    def meddle(x, optim_state, state):
        optim_state["iter"] = 1000
        optim_state["mesh_size"] = 0.0
        return False

    plain = _make_bads().optimize()
    with_fcn = _make_bads(output_fcn=meddle).optimize()
    assert np.array_equal(with_fcn["x"], plain["x"])
    assert with_fcn["fval"] == plain["fval"]
    assert with_fcn["func_count"] == plain["func_count"]
    assert with_fcn["iterations"] == plain["iterations"]


def test_output_fcn_stops_run_at_init():
    """A true return value at the start ends the run before its first
    iteration; the result reports the best point of the initial design."""
    record = _Recorder(stop_at=lambda n: True)
    bads = _make_bads(output_fcn=record)
    result = bads.optimize()
    assert record.states == ["init", "done"]
    assert result["iterations"] == 0
    assert result["func_count"] == bads.optim_state["eff_starting_points"] + 1
    assert result["message"] == (
        "Optimization terminated by options['output_fcn']."
    )
    assert result["fval"] == pytest.approx(_sphere(result["x"]))


def test_output_fcn_stops_run_after_a_poll():
    record = _Recorder(stop_at=lambda n: n == 3)  # the second "iter" call
    result = _make_bads(output_fcn=record).optimize()
    assert record.states == ["init", "iter", "iter", "done"]
    assert result["iterations"] == 2
    assert result["message"] == (
        "Optimization terminated by options['output_fcn']."
    )


@pytest.mark.parametrize("max_iter", [1, 2, 4])
def test_iterations_count_from_one(max_iter):
    """A run that ends on `max_iter` reports `max_iter` iterations, as
    MATLAB BADS does. The budget leaves `max_iter` the first criterion to
    end the run (this run, without it, ends on `tol_fun` at the 8th)."""
    result = _make_bads(max_iter=max_iter, max_fun_evals=200).optimize()
    assert result["iterations"] == max_iter
    assert result["message"] == (
        "Optimization terminated: reached maximum number of iterations "
        "options['max_iter']."
    )


@pytest.mark.parametrize(
    "uncertainty_handling, func_count",
    [(None, 2), (True, 1), (False, 1)],
    ids=["noise_test", "declared_noisy", "declared_deterministic"],
)
def test_one_function_evaluation(uncertainty_handling, func_count):
    """`max_fun_evals=1` evaluates the starting point (on the search grid),
    and a second time as the noise test when `uncertainty_handling` is
    empty, as MATLAB BADS does, and returns it. A target declared noisy
    takes no final samples."""
    bads = _make_bads(
        max_fun_evals=1, uncertainty_handling=uncertainty_handling
    )
    result = bads.optimize()
    assert np.allclose(result["x"].ravel(), np.ones(D) * 4, atol=0.01)
    assert result["fval"] == _sphere(result["x"])
    assert result["func_count"] == func_count
    assert result["iterations"] == 0
    assert result["message"] == (
        "Optimization terminated: reached maximum number of function "
        "evaluations after initialization."
    )


def _small_budget_bads(dim, max_fun_evals, noisy=False):
    rng = np.random.default_rng(0)

    def fun(x):
        return _sphere(x) + (rng.standard_normal() if noisy else 0.0)

    return BADS(
        fun,
        np.ones(dim) * 0.5,
        -5 * np.ones(dim),
        5 * np.ones(dim),
        -2 * np.ones(dim),
        2 * np.ones(dim),
        options={
            "display": "off",
            "max_fun_evals": max_fun_evals,
            "random_seed": 1,
        },
    )


@pytest.mark.parametrize(
    "dim, max_fun_evals", [(2, 3), (2, 4), (2, 5), (2, 6), (5, 7)]
)
def test_initial_design_within_budget(dim, max_fun_evals):
    """The initial design, rounded up to a power of two (doubled when that
    equals D), keeps its first points within the evaluations left after the
    starting point and the noise test, so a budget below it is kept."""
    result = _small_budget_bads(dim, max_fun_evals).optimize()
    assert result["func_count"] <= max_fun_evals


def test_initial_design_within_budget_of_noisy_run():
    """In a noisy run the design, of 32 points at D = 2, keeps within the
    budget, and the reserve for the final samples is never negative, so
    `max_fun_evals` never grows."""
    bads = _small_budget_bads(2, 25, noisy=True)
    result = bads.optimize()
    assert bads.optim_state["uncertainty_handling_level"] == 1
    assert result["func_count"] <= 25
    assert bads.options["noise_final_samples"] >= 0
    assert bads.options["max_fun_evals"] <= 25


def _box():
    return (
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
    )


def test_options_in_matlab_argument_order():
    """`BADS(fun, x0, lb, ub, plb, pub, non_box_cons, options)`, MATLAB
    BADS's order, passes the options."""
    options = {"display": "off", "max_fun_evals": 7, "random_seed": 3}
    bads = BADS(_sphere, *_box(), None, options)
    assert bads.options["max_fun_evals"] == 7
    assert bads.gamma_uncertain_interval is None


def test_gamma_uncertain_interval_is_keyword_only():
    options = {"display": "off", "random_seed": 3}
    with pytest.raises(TypeError):
        BADS(_sphere, *_box(), None, options, 2.0)
    bads = BADS(
        _sphere, *_box(), options=options, gamma_uncertain_interval=2.0
    )
    assert bads.gamma_uncertain_interval == 2.0


def test_successful_points_are_recorded_as_arrays():
    """`optim_state["u_success"]` holds the points of the successful searches
    and polls, as arrays. Without searches, every success is a poll's."""
    bads = _make_bads(search_n_try=0)
    bads.optimize()
    successes = bads.optim_state["u_success"]
    assert len(successes) > 0
    for u in successes:
        assert isinstance(u, np.ndarray) and u.size == D


def test_declared_deterministic_target_takes_no_noise_test():
    """With `uncertainty_handling=False`, as in MATLAB BADS, the starting
    point is not evaluated again to test for noise, and a noisy target is
    optimized as a deterministic one."""
    rng = np.random.default_rng(0)

    def noisy(x):
        return _sphere(x) + rng.standard_normal()

    bads = _make_bads(noisy, uncertainty_handling=False, max_fun_evals=40)
    bads.optimize()
    assert bads.optim_state["uncertainty_handling_level"] == 0
    evaluated = bads.function_logger.X[bads.function_logger.X_flag]
    assert bads.function_logger.func_count == len(evaluated)


def _one_ulp_apart():
    """The sphere, one unit in the last place higher on every other call,
    as a deterministic target whose value depends on the order of a sum."""
    n_calls = [0]

    def fun(x):
        n_calls[0] += 1
        y = _sphere(x)
        return float(np.nextafter(y, np.inf)) if n_calls[0] % 2 else y

    return fun


def _small_noise():
    rng = np.random.default_rng(0)

    def fun(x):
        return _sphere(x) + 1e-6 * rng.standard_normal()

    return fun


@pytest.mark.parametrize(
    "make_fun, level",
    [(_one_ulp_apart, 0), (_small_noise, 1)],
    ids=["one_ulp", "noise_sd_1e-6"],
)
def test_noise_test_threshold(make_fun, level):
    """The noise test takes a target as noisy when its repeat at the
    starting point differs by more than `tol_noise`, `sqrt(eps) * tol_fun`
    (1.5e-11) as in MATLAB BADS (`bads.m`): a difference in the last bit is
    not noise, a noise of SD 1e-6 is."""
    bads = _make_bads(make_fun(), max_fun_evals=50)
    bads.optimize()
    assert bads.optim_state["uncertainty_handling_level"] == level


def test_random_x0_is_uniform_in_the_transformed_box():
    """A missing `x0` is drawn uniformly in the transformed plausible box,
    as in MATLAB BADS (`setupvars.m`): log-uniform in the original space for
    a log-transformed variable. The draw is the run's first."""
    lb, ub = np.array([0.5, -10.0]), np.array([200.0, 10.0])
    plb, pub = np.array([1.0, -5.0]), np.array([100.0, 5.0])
    bads = BADS(
        _sphere,
        None,
        lb,
        ub,
        plb,
        pub,
        options={"display": "off", "random_seed": 5},
    )
    assert bads.var_transf.apply_log_t.tolist() == [[True, False]]
    u = np.random.default_rng(5).uniform(-1.0, 1.0, size=(1, 2))
    # [1, 100] maps log-linearly to [-1, 1], [-5, 5] linearly
    expected = [10.0 ** (1.0 + u[0, 0]), 5.0 * u[0, 1]]
    np.testing.assert_allclose(bads.x0.ravel(), expected, rtol=1e-12)


def test_run_without_sloppy_improvement():
    """`sloppy_improvement=False`, which MATLAB BADS supports, requires the
    improvement of the mesh size alone, without the floor at `tol_fun`, and
    the run completes."""
    result = _make_bads(sloppy_improvement=False, max_fun_evals=60).optimize()
    assert result["func_count"] <= 60
    assert result["iterations"] > 1
    assert result["fval"] < _sphere(np.ones(D) * 4)
