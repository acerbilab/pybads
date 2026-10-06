"""`Timer`, which times the run and the target's evaluations."""

import pytest

import pybads.utils.timer.timer as timer_module
from pybads.utils.timer import Timer


def test_timer_measures_with_perf_counter(monkeypatch):
    """The durations come from `time.perf_counter`, whose resolution is the
    highest: `time.time`'s is about 15.6 ms on Windows before Python 3.13,
    which timed most evaluations of a fast target as 0."""
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(timer_module.time, "perf_counter", lambda: next(ticks))
    timer = Timer(eps_t=0.0)
    timer.start_timer("fun")
    timer.stop_timer("fun")
    assert timer.get_duration("fun") == pytest.approx(0.25)


def test_timer_duration_of_a_timer_never_started_is_none():
    timer = Timer()
    timer.stop_timer("never")
    assert timer.get_duration("never") is None
