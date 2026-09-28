"""The output function, the iteration count and the shortest runs, as MATLAB
BADS has them: `output_fcn(x, optim_state, state)` is called at the start
(`"init"`), after each poll (`"iter"`) and at the end (`"done"`), and stops
the run when it returns a true value; `iterations` counts from 1."""

import logging
import threading

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
    end the run: without it, this run ends on `tol_fun` after more
    iterations than any `max_iter` here."""
    result = _make_bads(max_iter=max_iter, max_fun_evals=200).optimize()
    assert result["iterations"] == max_iter
    assert result["message"] == (
        "Optimization terminated: reached maximum number of iterations "
        "options['max_iter']."
    )


# The options that end the run on each criterion, the status and the message
_ENDS = {
    "max_fun_evals": (
        {"max_fun_evals": 30},
        0,
        "reached maximum number of function evaluations "
        "options['max_fun_evals']",
    ),
    "max_iter": (
        {"max_iter": 2, "max_fun_evals": 200},
        0,
        "reached maximum number of iterations options['max_iter']",
    ),
    "output_fcn": (
        {"output_fcn": lambda x, optim_state, state: state == "iter"},
        0,
        "terminated by options['output_fcn']",
    ),
    "initialization": ({"max_fun_evals": 1}, 0, "after initialization"),
    "tol_mesh": (
        {"tol_mesh": 1e-2, "max_fun_evals": 200},
        1,
        "mesh size less than options['tol_mesh']",
    ),
    "stall": (
        {"max_fun_evals": 200},
        2,
        "change in the function value less than options['tol_fun']",
    ),
}


@pytest.mark.parametrize("end", list(_ENDS))
def test_status_is_the_exit_flag(end):
    """`status` is MATLAB BADS's exit flag: 0 when the run ends on
    `max_fun_evals`, `max_iter`, a stop by `output_fcn` or in its
    initialization, 1 on `tol_mesh`, 2 on the stall criterion; `success` is
    True for the last two only, as in MATLAB and scipy. The `fsd` of a
    deterministic run is the float 0.0."""
    options, status, message = _ENDS[end]
    result = _make_bads(**options).optimize()
    assert message in result["message"]
    assert result["status"] == status
    assert result["success"] is (status > 0)
    assert isinstance(result["fsd"], float) and result["fsd"] == 0.0


def test_accelerated_mesh_reduction_counts_iterations_from_one(monkeypatch):
    """A failed poll shrinks the mesh once more when the last
    `accelerate_mesh_steps` iterations improved by less than `tol_fun`, from
    MATLAB BADS's iteration `accelerate_mesh_steps + 1` on, the 0-based
    `optim_state["iter"] == accelerate_mesh_steps`. The run starts at the
    minimum, so that every poll fails and every such test shrinks the
    mesh."""
    polls = []
    original_poll = BADS._poll_step_

    def poll(self, gp):
        iteration = self.optim_state["iter"]
        mesh_size_integer = self.mesh_size_integer
        out = original_poll(self, gp)
        polls.append((iteration, mesh_size_integer - self.mesh_size_integer))
        return out

    monkeypatch.setattr(BADS, "_poll_step_", poll)
    bads = BADS(
        _sphere,
        np.zeros(D),
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        8 * np.ones(D),
        options={"display": "off", "max_fun_evals": 200, "random_seed": 3},
    )
    result = bads.optimize()
    steps = bads.options["accelerate_mesh_steps"]
    assert result["fval"] == 0
    assert len(polls) > steps
    assert polls == [
        (iteration, 1 if iteration < steps else 2)
        for iteration in range(len(polls))
    ]


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


def test_noise_test_leaves_the_start_row_as_it_was(monkeypatch):
    """The noise test evaluates the starting point again without recording
    it, as MATLAB BADS calls the target directly for it: the start's row of
    the log keeps its one evaluation and its own time, and the rows' counts
    of evaluations, whose sum is the `n_eff` of the GP's training, count the
    points recorded, as `eff_starting_points` does."""
    calls = []

    class _CountingTimer:
        """A timer that times the k-th evaluation at k s."""

        def start_timer(self, name):
            calls.append(name)

        def stop_timer(self, name):
            pass

        def get_duration(self, name):
            return float(len(calls))

    monkeypatch.setattr(
        "pybads.function_logger.function_logger.Timer", _CountingTimer
    )
    bads = _make_bads(max_fun_evals=10)
    bads.optimize()
    logger = bads.function_logger
    assert logger.n_evals[0, 0] == 1
    assert logger.fun_eval_time[0, 0] == 1.0
    assert np.sum(logger.n_evals[logger.X_flag]) == logger.Xn + 1
    assert logger.Xn + 1 == logger.func_count - 1


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


def test_reserve_of_final_samples_is_floored_at_zero():
    """A noisy run whose noise test takes it past `max_fun_evals`, at a
    budget of 1, reserves no final samples, where the evaluations left are
    -1, so that `max_fun_evals` does not grow; the run ends after its two
    evaluations, as MATLAB BADS's does."""
    bads = _small_budget_bads(2, 1, noisy=True)
    result = bads.optimize()
    assert bads.optim_state["uncertainty_handling_level"] == 1
    assert result["func_count"] == 2
    assert bads.options["noise_final_samples"] == 0
    assert bads.options["max_fun_evals"] == 1


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


class _Locked:
    """A target and a constraint that hold a lock, which cannot be copied."""

    def __init__(self):
        self.lock = threading.Lock()

    def target(self, x):
        with self.lock:
            return _sphere(x)

    def __call__(self, x):
        with self.lock:
            return np.sum(np.atleast_2d(x) ** 2, axis=1) - 1e4


def test_result_keeps_the_callables_by_reference():
    """The result holds the target and the constraint that the run was
    given, not copies: a bound method or callable object whose instance
    holds a lock does not stop `optimize()`."""
    locked = _Locked()
    target = locked.target
    bads = BADS(
        target,
        *_box(),
        locked,
        options={"display": "off", "max_fun_evals": 40, "random_seed": 3},
    )
    result = bads.optimize()
    assert result["fun"] is target
    assert result["fun"].__self__ is locked
    assert result["non_box_cons"] is locked


def test_run_with_certain_incumbent():
    """With `uncertain_incumbent=False`, a deterministic target's
    optimization target is the incumbent's value less `tol_fun`, as in
    MATLAB BADS (`UpdateTarget` in `bads.m`), in the form the search and
    the poll store with `.item()`, and the run completes."""
    bads = _make_bads(uncertain_incumbent=False, max_fun_evals=60)
    result = bads.optimize()
    optim_state = bads.optim_state
    assert optim_state["uncertainty_handling_level"] == 0
    assert result["iterations"] > 1
    assert result["fval"] < _sphere(np.ones(D) * 4)
    assert (
        optim_state["f_target"]
        == optim_state["f_target_mu"] - bads.options["tol_fun"]
    )
    assert np.all(optim_state["f_target_s"] == 0)


def _acquisition_run(monkeypatch, nan_mask):
    """A run whose acquisition values at the search and the poll are NaN
    where `nan_mask(n)` is true, `n` the number of candidates. It returns
    the events in their order: each acquisition's site and candidates, and
    each evaluation's point."""
    import pybads.bads.bads as bads_module
    from pybads.function_logger import FunctionLogger

    events = []
    bads = _make_bads(max_fun_evals=60)
    original_acq = bads_module.acq_fcn_lcb
    original_call = FunctionLogger.__call__

    def acq(u, func_count, gp):
        z, f_mu, fs = original_acq(u, func_count, gp)
        z = np.array(z, dtype=float)
        z[nan_mask(len(z))] = np.nan
        # The poll runs after the round of searches, which resets the count
        site = "search" if bads.optim_state["search_count"] > 0 else "poll"
        events.append((site, u.copy()))
        return z, f_mu, fs

    def call(self, x, *args, **kwargs):
        events.append(("eval", np.array(x, dtype=float).ravel()))
        return original_call(self, x, *args, **kwargs)

    monkeypatch.setattr(bads_module, "acq_fcn_lcb", acq)
    monkeypatch.setattr(FunctionLogger, "__call__", call)
    bads.optimize()
    return [
        (site, u, x)
        for (site, u), (kind, x) in zip(events, events[1:])
        if site != "eval" and kind == "eval"
    ]


def test_acquisition_skips_nan(monkeypatch):
    """A NaN acquisition value is skipped, as by MATLAB BADS's `min`: with
    every value NaN but the last, the poll evaluates its last candidate (the
    search has one candidate, the best of its ES search)."""
    chosen = _acquisition_run(monkeypatch, lambda n: np.arange(n) < n - 1)
    for _, u, x in chosen:
        assert np.array_equal(x, u[-1].ravel())
    assert {site for site, u, _ in chosen if len(u) > 1} == {"poll"}


def test_acquisition_all_nan_chooses_at_random(monkeypatch, caplog):
    """When every acquisition value is NaN, the search and the poll choose
    a candidate at random, with a warning, and the run goes on."""
    with caplog.at_level(logging.WARNING, logger="BADS"):
        chosen = _acquisition_run(
            monkeypatch, lambda n: np.ones(n, dtype=bool)
        )
    warnings = [
        record
        for record in caplog.records
        if "Acquisition function failed" in record.getMessage()
    ]
    assert len(warnings) >= len(chosen) > 0
    assert {site for site, _, _ in chosen} == {"search", "poll"}
    indices = []
    for _, u, x in chosen:
        (index,) = [i for i in range(len(u)) if np.array_equal(u[i], x)]
        indices.append(index)
    assert max(indices) > 0


def test_poll_stop_probability_takes_the_largest_probabilities(monkeypatch):
    """The probability that no poll point improves, which decides whether
    the poll stops, is the product of the complements of the D largest
    probabilities of improvement of the points left, as in MATLAB BADS
    (`bads.m:868-869`). With the predictions of the run's first poll step
    patched so that its first point has a probability of improvement of
    0.69 and the five others of 1e-9, it is 0.31 * (1 - 1e-9)**2, not about
    1."""
    from scipy.stats import norm

    import pybads.bads.bads as bads_module

    bads = _make_bads(max_fun_evals=30)
    original_acq = bads_module.acq_fcn_lcb
    original_stop = BADS._is_poll_stop_
    patched = []
    p_less = []

    def acq(u, func_count, gp):
        z, f_mu, fs = original_acq(u, func_count, gp)
        poll = bads.optim_state["search_count"] == 0
        if poll and not patched and len(u) == 2 * D:
            # f_mu and fs such that gamma_z is norm.ppf of each probability
            poi = np.full(f_mu.shape, 1e-9)
            poi[0] = 0.69
            fs = np.ones(f_mu.shape)
            f_mu = (
                bads.optim_state["f_target"]
                - bads.sufficient_improvement
                - norm.ppf(poi)
            )
            patched.append(True)
        return z, f_mu, fs

    def stop(self, certain_good_poll, do_gp_calibration, p, poll_count):
        if patched and not p_less:
            p_less.append(p)
        return original_stop(
            self, certain_good_poll, do_gp_calibration, p, poll_count
        )

    monkeypatch.setattr(bads_module, "acq_fcn_lcb", acq)
    monkeypatch.setattr(BADS, "_is_poll_stop_", stop)
    bads.optimize()
    assert len(p_less) == 1
    assert p_less[0] == pytest.approx(0.31 * (1 - 1e-9) ** 2, rel=1e-9)


def test_rebuilds_of_the_local_gp_after_a_move(monkeypatch):
    """A move of the incumbent asks for a rebuild of the local GP, as MATLAB
    BADS empties the posterior: after a poll that moves the incumbent, each
    search of the next rounds rebuilds it, until a poll that does not move
    (`bads.m:1049`, at the end of every pass); otherwise the first search of
    a round rebuilds it, as the poll does at its first step, and a later
    search only after a search that moved the incumbent, or for a refit. On
    Rosenbrock's function some polls move (on the sphere, none)."""
    import pybads.bads.bads as bads_module

    def rosenbrock(x):
        x = np.ravel(x)
        return float(
            np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
        )

    original_fitting = bads_module.local_gp_fitting
    steps = []
    refits = []

    def fitting(*args, **kwargs):
        refits.append(args[6])  # refit_flag
        return original_fitting(*args, **kwargs)

    def step(original, kind):
        def wrapper(self, gp):
            first = self.optim_state["search_count"] == 0
            u_best, n_fits = self.u_best.copy(), len(refits)
            out = original(self, gp)
            moved = not np.array_equal(u_best, self.u_best)
            steps.append((kind, first, moved, refits[n_fits:]))
            return out

        return wrapper

    monkeypatch.setattr(bads_module, "local_gp_fitting", fitting)
    monkeypatch.setattr(
        BADS, "_search_step_", step(BADS._search_step_, "search")
    )
    monkeypatch.setattr(BADS, "_poll_step_", step(BADS._poll_step_, "poll"))
    _make_bads(rosenbrock, max_fun_evals=100).optimize()

    # The searches that only a poll's move asks to rebuild the local GP, and
    # those that nothing asks to
    after_poll_move, unasked = 0, 0
    poll_moved = None
    search_moved = False
    for kind, first, moved, step_refits in steps:
        if kind == "poll":
            poll_moved = moved
        elif poll_moved is not None:
            if first or search_moved or poll_moved:
                assert len(step_refits) == 1
            else:
                assert all(step_refits)  # a refit only
            if not first and not search_moved and not any(step_refits):
                if poll_moved:
                    after_poll_move += 1
                else:
                    unasked += 1
        search_moved = kind == "search" and moved
    assert after_poll_move > 0 and unasked > 0
