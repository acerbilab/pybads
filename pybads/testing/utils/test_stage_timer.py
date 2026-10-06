"""`StageTimer`, which charges each second of a run to one stage."""

import numpy as np
import pytest

import pybads.utils.timer.stage_timer as stage_timer_module
from pybads.utils.timer.stage_timer import (
    NULL_STAGE_TIMER,
    StageTimer,
    get_stage_timer,
)


class _Clock:
    """A clock that moves only when told to."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _TargetTime:
    """A target's cumulative evaluation time, set by the test."""

    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(stage_timer_module.time, "perf_counter", clock)
    return clock


def test_nested_stage_pauses_its_parent(clock):
    timer = StageTimer()
    timer.start()
    clock.advance(1)
    with timer.stage("a"):
        clock.advance(2)
        with timer.stage("b"):
            clock.advance(3)
        clock.advance(4)
    clock.advance(5)
    timer.stop()
    snapshot = timer.snapshot()
    assert snapshot["seconds"] == {
        "loop": 6.0,
        "a": 6.0,
        "a/b": 3.0,
        "target": 0.0,
    }
    assert snapshot["calls"] == {"loop": 1, "a": 1, "a/b": 1}
    assert sum(snapshot["seconds"].values()) == 15.0
    assert not timer.running and timer.depth == 0


def test_keys_are_paths_and_count_entries(clock):
    timer = StageTimer(root="run")
    timer.start()
    for outer in ("search", "poll", "search"):
        with timer.stage(outer):
            with timer.stage("gp_rebuild"):
                with timer.stage("gp_fit"):
                    clock.advance(1)
    timer.stop()
    snapshot = timer.snapshot()
    assert snapshot["calls"] == {
        "run": 1,
        "search": 2,
        "search/gp_rebuild": 2,
        "search/gp_rebuild/gp_fit": 2,
        "poll": 1,
        "poll/gp_rebuild": 1,
        "poll/gp_rebuild/gp_fit": 1,
    }
    assert snapshot["seconds"]["search/gp_rebuild/gp_fit"] == 2.0
    assert snapshot["seconds"]["poll/gp_rebuild/gp_fit"] == 1.0


def test_target_time_goes_to_the_target(clock):
    """The increase of the target's time between two transitions is
    charged to "target", the rest to the stage that was open, so that the
    seconds add up to the time between start and stop."""
    target = _TargetTime()
    timer = StageTimer(target)
    timer.start()
    with timer.stage("init"):
        clock.advance(5)
        target.value += 2
    with timer.stage("search"):
        clock.advance(1)
        target.value += 0.25
        with timer.stage("search_es"):
            clock.advance(3)
    timer.stop()
    seconds = timer.snapshot()["seconds"]
    assert seconds["init"] == 3.0
    assert seconds["search"] == 0.75
    assert seconds["search/search_es"] == 3.0
    assert seconds["target"] == 2.25
    assert seconds["loop"] == 0.0
    assert sum(seconds.values()) == 9.0


def test_target_time_before_start_is_not_charged(clock):
    target = _TargetTime()
    target.value = 7.0
    timer = StageTimer(target)
    timer.start()
    clock.advance(1)
    timer.stop()
    assert timer.snapshot()["seconds"] == {"loop": 1.0, "target": 0.0}


def test_stage_left_by_an_exception_is_closed(clock):
    timer = StageTimer()
    timer.start()
    with pytest.raises(ValueError, match="raised in the stage"):
        with timer.stage("poll"):
            with timer.stage("gp_update"):
                clock.advance(2)
                raise ValueError("raised in the stage")
    assert timer.depth == 1
    clock.advance(1)
    timer.stop()
    seconds = timer.snapshot()["seconds"]
    assert seconds["poll/gp_update"] == 2.0
    assert seconds["loop"] == 1.0


def test_attempt_that_succeeds_stays_with_its_stage(clock):
    timer = StageTimer()
    timer.start()
    with timer.stage("gp_fit"):
        clock.advance(1)
        with timer.attempt("gp_fit_failed", np.linalg.LinAlgError):
            clock.advance(2)
            with timer.stage("inner"):
                clock.advance(4)
    timer.stop()
    snapshot = timer.snapshot()
    assert snapshot["seconds"]["gp_fit"] == 3.0
    assert snapshot["seconds"]["gp_fit/inner"] == 4.0
    assert "gp_fit/gp_fit_failed" not in snapshot["seconds"]
    assert "gp_fit/gp_fit_failed" not in snapshot["calls"]


def test_attempt_that_fails_moves_its_time(clock):
    """A failed attempt's own time moves to its key, which counts one entry;
    a stage nested in it keeps its time."""
    timer = StageTimer()
    timer.start()
    with timer.stage("gp_fit"):
        clock.advance(1)
        for _ in range(2):
            with pytest.raises(np.linalg.LinAlgError):
                with timer.attempt("gp_fit_failed", np.linalg.LinAlgError):
                    clock.advance(2)
                    with timer.stage("inner"):
                        clock.advance(4)
                    clock.advance(0.5)
                    raise np.linalg.LinAlgError("failed")
        with timer.attempt("gp_fit_failed", np.linalg.LinAlgError):
            clock.advance(8)
    timer.stop()
    snapshot = timer.snapshot()
    assert snapshot["seconds"]["gp_fit"] == 9.0
    assert snapshot["seconds"]["gp_fit/gp_fit_failed"] == 5.0
    assert snapshot["seconds"]["gp_fit/inner"] == 8.0
    assert snapshot["calls"]["gp_fit/gp_fit_failed"] == 2
    assert snapshot["calls"]["gp_fit/inner"] == 2
    assert sum(snapshot["seconds"].values()) == 22.0


def test_attempt_that_raises_another_error_stays_with_its_stage(clock):
    timer = StageTimer()
    timer.start()
    with pytest.raises(ValueError):
        with timer.stage("gp_fit"):
            with timer.attempt("gp_fit_failed", np.linalg.LinAlgError):
                clock.advance(2)
                raise ValueError("not a failed fit")
    timer.stop()
    snapshot = timer.snapshot()
    assert snapshot["seconds"]["gp_fit"] == 2.0
    assert "gp_fit/gp_fit_failed" not in snapshot["calls"]


def test_snapshot_is_of_plain_numbers_and_charges_the_open_stage(clock):
    target = _TargetTime()
    timer = StageTimer(target)
    timer.start()
    with timer.stage("search"):
        clock.advance(2)
        target.value = np.float64(0.5)
        snapshot = timer.snapshot()
        snapshot["seconds"]["search"] = -1.0
    assert timer.snapshot()["seconds"]["search"] == 1.5
    for value in snapshot["seconds"].values():
        assert type(value) is float
    for value in snapshot["calls"].values():
        assert type(value) is int
    timer.stop()


@pytest.mark.parametrize(
    "value",
    [None, "abc", float("nan"), float("inf"), object(), RuntimeError],
    ids=["none", "string", "nan", "inf", "object", "raises"],
)
def test_odd_target_time_is_ignored(clock, value):
    def target_time():
        if value is RuntimeError:
            raise RuntimeError("target time")
        return value

    timer = StageTimer(target_time)
    timer.start()
    with timer.stage("a"):
        clock.advance(1)
    timer.stop()
    seconds = timer.snapshot()["seconds"]
    assert seconds["target"] == 0.0
    assert seconds["a"] == 1.0
    assert timer.depth == 0


def test_target_time_that_goes_down_is_charged_as_is(clock):
    """The timer asserts nothing about the times it reads: a target timer
    replaced in a test can make a stage's own time negative."""
    target = _TargetTime()
    timer = StageTimer(target)
    timer.start()
    with timer.stage("a"):
        clock.advance(1)
        target.value = 3.0
    with timer.stage("b"):
        target.value = 1.0
    timer.stop()
    seconds = timer.snapshot()["seconds"]
    assert seconds["a"] == -2.0
    assert seconds["b"] == 2.0
    assert seconds["target"] == 1.0


def test_stages_outside_a_run_are_not_timed(clock):
    timer = StageTimer()
    with timer.stage("before"):
        clock.advance(1)
    assert timer.snapshot() == {"seconds": {}, "calls": {}}
    timer.start()
    timer.start()
    clock.advance(1)
    timer.stop()
    timer.stop()
    with timer.stage("after"):
        with timer.attempt("failed"):
            clock.advance(1)
    assert timer.snapshot()["seconds"] == {"loop": 1.0, "target": 0.0}
    assert timer.snapshot()["calls"] == {"loop": 1}


def test_stop_closes_open_stages_and_a_late_exit_is_ignored(clock):
    timer = StageTimer()
    timer.start()
    stage = timer.stage("search")
    stage.__enter__()
    clock.advance(2)
    timer.stop()
    assert timer.depth == 0
    clock.advance(1)
    assert stage.__exit__(None, None, None) is False
    assert stage.__exit__(ValueError, ValueError(), None) is False
    assert timer.snapshot()["seconds"]["search"] == 2.0


def test_exit_of_an_outer_stage_closes_the_stages_left_inside(clock):
    timer = StageTimer()
    timer.start()
    outer = timer.stage("outer")
    outer.__enter__()
    timer.stage("inner").__enter__()
    clock.advance(1)
    outer.__exit__(None, None, None)
    assert timer.depth == 1
    clock.advance(1)
    timer.stop()
    seconds = timer.snapshot()["seconds"]
    assert seconds["outer/inner"] == 1.0
    assert seconds["loop"] == 1.0


def test_null_stage_timer_times_nothing():
    assert get_stage_timer(None) is NULL_STAGE_TIMER
    timer = StageTimer()
    assert get_stage_timer(timer) is timer
    null = get_stage_timer()
    null.start()
    with null.stage("a"):
        with pytest.raises(ValueError):
            with null.attempt("failed"):
                raise ValueError
    null.stop()
    assert null.snapshot() == {"seconds": {}, "calls": {}}
    assert not null.running and null.depth == 0
