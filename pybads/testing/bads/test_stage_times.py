"""The stage times of a run. `optimize` charges each second of the run to
one stage (`pybads.utils.timer.stage_timer.StageTimer`, private) and the
target's evaluations to the pseudo-stage "target", and stores the times as
plain numbers: `optim_state["stage_times"]` at the end of the run and
`iteration_history["timer"]` at the end of each iteration. The stages and
the target make `total_time`."""

import sys
from types import SimpleNamespace

import gpyreg as gpr
import numpy as np
import pytest

import pybads.utils.timer.stage_timer as stage_timer_module
import pybads.utils.timer.timer as timer_module
from pybads.utils.timer.stage_timer import StageTimer

from .test_gp_update_failures import (
    LEVELS,
    Injector,
    _make_bads,
    _pybads_frames,
)

TOP_LEVEL = {
    "loop",
    "init",
    "gp_init",
    "search",
    "poll",
    "history",
    "reestimate",
    "final_samples",
    "output_fcn",
    "target",
}
TICK = 1e-3


@pytest.fixture(autouse=True)
def _restore_global_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def tick_clock(monkeypatch):
    """Every reading of the clock of `Timer` and `StageTimer` is one tick
    after the previous one."""
    ticks = iter(range(1, sys.maxsize))
    clock = SimpleNamespace(perf_counter=lambda: next(ticks) * TICK)
    monkeypatch.setattr(timer_module, "time", clock)
    monkeypatch.setattr(stage_timer_module, "time", clock)


def _assert_stage_times(bads, result, exact):
    """The stage times of a finished run: the stages and the target make
    `total_time`, exactly with the tick clock (up to the two readings of the
    clock between the starts of the two timers and between their stops),
    each iteration has a snapshot, and no stage is left open."""
    stage_times = bads.optim_state["stage_times"]
    seconds, calls = stage_times["seconds"], stage_times["calls"]
    assert {key.split("/")[0] for key in seconds} <= TOP_LEVEL
    assert set(calls) == set(seconds) - {"target"}
    if exact:
        assert sum(seconds.values()) + 2 * TICK + 1e-9 == pytest.approx(
            result["total_time"], rel=0, abs=1e-9
        )
    else:
        assert sum(seconds.values()) < result["total_time"]
    assert not bads._stage_timer.running

    snapshots = bads.iteration_history["timer"]
    assert len(snapshots) == len(bads.iteration_history["fval"])
    totals = [sum(snapshot["seconds"].values()) for snapshot in snapshots]
    assert all(np.diff(totals) > 0)
    assert totals[-1] <= sum(seconds.values())
    for snapshot in snapshots:
        assert set(snapshot) == {"seconds", "calls"}
        assert all(type(v) is float for v in snapshot["seconds"].values())
        assert all(type(v) is int for v in snapshot["calls"].values())


@pytest.mark.parametrize("level", [0, 1, 2])
def test_stages_and_target_make_total_time(tick_clock, level):
    make_fun, options = LEVELS[level]
    bads = _make_bads(make_fun(), max_fun_evals=80, **options)
    result = bads.optimize()
    _assert_stage_times(bads, result, exact=True)
    stage_times = bads.optim_state["stage_times"]
    calls = stage_times["calls"]
    assert calls["loop"] == calls["init"] == calls["gp_init"] == 1
    assert calls["gp_init/gp_fit"] == 1
    assert calls["poll"] in (result["iterations"] - 1, result["iterations"])
    assert calls["history"] == len(bads.iteration_history["fval"])
    assert calls["search"] == calls["search/search_es"]
    # Each evaluation is timed at one tick and 1e-9 s by the function
    # logger, except the noise test's, which counts in "init"
    n_timed = result["func_count"] - (level == 0)
    assert stage_times["seconds"]["target"] == pytest.approx(
        n_timed * (TICK + 1e-9), rel=0, abs=1e-9
    )
    noisy = {"reestimate", "final_samples", "poll/gp_update"}
    assert noisy <= set(calls) if level > 0 else not noisy & set(calls)


@pytest.mark.parametrize("level", [0, 1])
def test_stage_times_with_the_real_clock(level):
    """Every stage's own time is positive, up to the 1e-9 s that the
    function logger adds to each evaluation's time."""
    make_fun, options = LEVELS[level]
    bads = _make_bads(make_fun(), max_fun_evals=60, **options)
    result = bads.optimize()
    _assert_stage_times(bads, result, exact=False)
    seconds = bads.optim_state["stage_times"]["seconds"]
    assert all(value > -1e-6 for value in seconds.values())


@pytest.mark.parametrize(
    "level, should_fail",
    [
        (
            0,
            lambda call: call.site == "add_and_update_gp" and call.n % 5 == 0,
        ),
        (
            1,
            lambda call: call.site == "local_gp_fitting" and call.n in (5, 6),
        ),
        (
            1,
            lambda call: call.site == "_get_target_from_gp_" and call.n == 3,
        ),
    ],
    ids=["add_level0", "double_local_level1", "target_level1"],
)
def test_stage_times_after_gp_update_failures(
    tick_clock, monkeypatch, level, should_fail
):
    injector = Injector(should_fail)
    for method in ("update", "set_hyperparameters"):
        monkeypatch.setattr(gpr.GP, method, injector.wrap(method))
    make_fun, options = LEVELS[level]
    bads = _make_bads(make_fun(), max_fun_evals=150, **options)
    result = bads.optimize()
    assert injector.failed
    _assert_stage_times(bads, result, exact=True)


def test_failed_fits_are_timed_as_failures(tick_clock, monkeypatch):
    """A fit that raises `LinAlgError` is timed as "gp_fit_failed" under its
    "gp_fit", each failure of a refit is followed by a "gp_fit_retry", and a
    refit that ends without a fit, as when its every try fails, ends in a
    "gp_fit_fallback". Failures are injected (the first initial fit, every
    try of the 2nd refit and the first try of the 3rd) beside those of the
    run itself."""
    import pybads.bads.gaussian_process_train as gpt_module

    original_fit = gpr.GP.fit
    original_robust_fit = gpt_module._robust_gp_fit_
    refits = {"n": 0, "fallbacks": 0}
    failures = {"init_and_train_gp": 0, "_robust_gp_fit_": 0}

    def fit(gp, *args, **kwargs):
        frames = _pybads_frames(sys._getframe(1))
        caller = frames[0] if frames else None
        if caller not in failures:
            return original_fit(gp, *args, **kwargs)
        i_try = sys._getframe(1).f_locals.get("i_try")
        if caller == "_robust_gp_fit_" and i_try == 0:
            refits["n"] += 1
        if caller == "init_and_train_gp":
            inject = failures[caller] == 0
        else:
            inject = refits["n"] == 2 or (refits["n"] == 3 and i_try == 0)
        try:
            if inject:
                raise np.linalg.LinAlgError("injected failure")
            return original_fit(gp, *args, **kwargs)
        except np.linalg.LinAlgError:
            failures[caller] += 1
            raise

    def robust_fit(*args, **kwargs):
        out = original_robust_fit(*args, **kwargs)
        refits["fallbacks"] += out[3] == -1
        return out

    monkeypatch.setattr(gpr.GP, "fit", fit)
    monkeypatch.setattr(gpt_module, "_robust_gp_fit_", robust_fit)
    bads = _make_bads(LEVELS[0][0](), max_fun_evals=150)
    result = bads.optimize()
    assert refits["n"] >= 3 and refits["fallbacks"] >= 1
    assert failures["init_and_train_gp"] == 1
    _assert_stage_times(bads, result, exact=True)
    calls = bads.optim_state["stage_times"]["calls"]

    def total(leaf):
        return sum(n for key, n in calls.items() if key.endswith("/" + leaf))

    assert calls["gp_init/gp_fit/gp_fit_failed"] == 1
    assert total("gp_fit_failed") == sum(failures.values())
    assert total("gp_fit_retry") == failures["_robust_gp_fit_"]
    assert total("gp_fit_fallback") == refits["fallbacks"]


def test_target_that_raises_leaves_no_stage_open():
    """An exception of the target propagates as it would without the
    timer, which is stopped with every stage closed."""

    def fun(x):
        fun.calls += 1
        if fun.calls == 40:
            raise RuntimeError("the target failed")
        return float(np.sum(np.atleast_2d(x) ** 2))

    fun.calls = 0
    bads = _make_bads(fun, max_fun_evals=80)
    with pytest.raises(RuntimeError, match="the target failed"):
        bads.optimize()
    timer = bads._stage_timer
    assert not timer.running and timer.depth == 0
    snapshot = timer.snapshot()
    assert snapshot["calls"]["loop"] == 1
    assert "stage_times" not in bads.optim_state


def _find_stage_timers(value, seen=None):
    """The stage timers reachable from ``value`` through dicts, lists,
    tuples and object arrays."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, StageTimer):
        return [value]
    if isinstance(value, dict):
        items = list(value.values())
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        items = list(value.ravel())
    else:
        return []
    return [t for item in items for t in _find_stage_timers(item, seen)]


def test_timer_stays_out_of_copied_state():
    """`optim_state`, the GPs and their `temporary_data` are deep-copied
    during a run: none of them holds the timer."""
    make_fun, options = LEVELS[1]
    bads = _make_bads(make_fun(), max_fun_evals=60, **options)
    bads.optimize()
    assert not _find_stage_timers(bads.optim_state)
    assert not _find_stage_timers(dict(bads.iteration_history))
    for gp in bads.iteration_history["gp"]:
        assert gp is not None and not _find_stage_timers(vars(gp))


def test_output_fcn_is_a_stage():
    states = []

    def output_fcn(x, optim_state, state):
        states.append(state)
        return False

    bads = _make_bads(LEVELS[0][0](), max_fun_evals=60, output_fcn=output_fcn)
    bads.optimize()
    assert bads.optim_state["stage_times"]["calls"]["output_fcn"] == len(
        states
    )


def test_steps_outside_optimize_time_nothing():
    bads = _make_bads(LEVELS[0][0]())
    bads._init_optimization_()
    assert bads._stage_timer.snapshot() == {"seconds": {}, "calls": {}}
