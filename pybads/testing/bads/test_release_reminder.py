"""The old-release reminder (`pybads/bads/_release_reminder.py`): one INFO
record of the run's logger, for a final release more than a year old, in an
interactive session without an opt-out variable, at most once per Python
session and three times per installed version, at least 90 days apart; its
state file, and the cache directory that holds it; its place in a run, after
the opening message and before the column headers, where it takes the tip's
place without advancing the tips' cadence; and a run that shows it computes
and draws what the same run without it does.

The suite's conftest keeps the reminder inert; the fixture here resets it
and points its state file into the test's temporary directory. The dates,
the installed version, the environment and the interactivity are injected,
so that no test depends on the calendar, the installation, the terminal or
the environment variables of the process (`CI` among them).
"""

import datetime
import functools
import importlib.metadata
import json
import logging
import os
import platform
import random
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pybads import BADS, _release
from pybads.bads import _release_reminder
from pybads.bads import _runtime_tips as rt
from pybads.bads._tip_catalog import TIPS
from pybads.bads.bads import _LOG_FINAL, _LOG_NOTIFY

D = 3
# Apple's Accelerate makes two runs of one seed differ in their last bits on
# macOS arm64, where the seed alone decides the start and the initial design
# (dev/results/2026-09-28-macos-arm64-repeatability.md)
_REPEATS_BIT_FOR_BIT = not (
    sys.platform == "darwin" and platform.machine() == "arm64"
)
_REPO = Path(__file__).resolve().parents[3]
# The state path of the package, taken before any fixture replaces it
DEFAULT_STATE_PATH = _release_reminder._default_state_path

RELEASED = datetime.date(2026, 3, 15)
RELEASE_DATE = RELEASED.isoformat()
INSTALLED = "1.5.0"
ELIGIBLE_DAY = RELEASED + datetime.timedelta(days=500)
NOTE = "Note: PyBADS"
LAST = "This is the last reminder for PyBADS"

_RELEASE_HEADING = re.compile(
    r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\] - ([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$",
    re.MULTILINE,
)


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture(autouse=True)
def reminder(_inert_release_reminder, monkeypatch, tmp_path):
    """Opt back into the reminder, with a state file and a logger of the
    test's own, and a session of the tips with a seeded shuffle."""
    inherited = SimpleNamespace(
        shown=_release_reminder._SHOWN_THIS_SESSION,
        path=_release_reminder._default_state_path(),
    )
    state_path = tmp_path / "cache" / _release_reminder.STATE_FILE_NAME
    monkeypatch.setattr(
        _release_reminder, "_default_state_path", lambda: state_path
    )
    for name in _release_reminder.OPT_OUT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    _release_reminder._reset_release_reminder_state()
    rt._reset_runtime_tip_state(rng=random.Random(0))
    logger = logging.getLogger("pybads.testing.release_reminder")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    records = _Records()
    logger.addHandler(records)
    yield SimpleNamespace(
        state_path=state_path,
        inherited=inherited,
        logger=logger,
        records=records.records,
    )
    logger.removeHandler(records)
    rt._reset_runtime_tip_state()


@pytest.fixture
def eligible(monkeypatch):
    """Make the defaults of the reminder describe an eligible start."""
    monkeypatch.setattr(_release, "RELEASE_DATE", RELEASE_DATE)
    monkeypatch.setattr(_release_reminder, "_today", lambda: ELIGIBLE_DAY)
    monkeypatch.setattr(
        _release_reminder, "_installed_version", lambda: INSTALLED
    )
    monkeypatch.setattr(_release_reminder, "_is_interactive", lambda: True)


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A home directory that `~` expands to, on every platform."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def _consider(reminder, **kwargs):
    """Consider the reminder at an eligible start, amended by *kwargs*."""
    arguments = {
        "logger": reminder.logger,
        "enabled": True,
        "today": ELIGIBLE_DAY,
        "installed": INSTALLED,
        "release_date": RELEASE_DATE,
        "interactive": True,
        "environ": {},
        "state_path": reminder.state_path,
    }
    arguments.update(kwargs)
    return _release_reminder.consider_release_reminder(**arguments)


def _messages(reminder):
    return [record.getMessage() for record in reminder.records]


def _day(days):
    return ELIGIBLE_DAY + datetime.timedelta(days=days)


def _new_session():
    _release_reminder._reset_release_reminder_state()


def _first_tip():
    """The first tip of the session's order, which the autouse fixture
    shuffles with a generator of seed 0."""
    order = list(TIPS)
    random.Random(0).shuffle(order)
    return order[0]


def test_suite_keeps_the_reminder_inert_and_out_of_the_user_cache(
    reminder, tmp_path_factory
):
    assert reminder.inherited.shown is True
    assert reminder.inherited.path.name == "update_reminder.json"
    assert reminder.inherited.path.is_relative_to(
        tmp_path_factory.getbasetemp()
    )


def test_state_file_lies_in_the_cache_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("PYBADS_CACHE_DIR", str(tmp_path / "cache"))
    assert DEFAULT_STATE_PATH() == tmp_path / "cache" / "update_reminder.json"
    assert not (tmp_path / "cache").exists()
    monkeypatch.setattr(_release_reminder, "_cache_root", lambda: None)
    assert DEFAULT_STATE_PATH() is None


def test_cache_directory_on_windows(tmp_path):
    local = tmp_path / "Local"
    environ = {"LOCALAPPDATA": str(local), "XDG_CACHE_HOME": str(tmp_path)}
    assert (
        _release_reminder._cache_root(environ=environ, platform="win32")
        == local / "pybads"
    )
    # Without an absolute LOCALAPPDATA, there is none
    for environ in (
        {},
        {"LOCALAPPDATA": ""},
        {"LOCALAPPDATA": "  "},
        {"LOCALAPPDATA": "Local"},
    ):
        assert (
            _release_reminder._cache_root(environ=environ, platform="win32")
            is None
        )


def test_cache_directory_on_macos(home, tmp_path):
    environ = {
        "LOCALAPPDATA": str(tmp_path / "Local"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg"),
    }
    assert (
        _release_reminder._cache_root(environ=environ, platform="darwin")
        == home / "Library" / "Caches" / "pybads"
    )


@pytest.mark.parametrize("sys_platform", ["linux", "freebsd14"])
def test_cache_directory_elsewhere(home, tmp_path, sys_platform):
    cache_root = functools.partial(
        _release_reminder._cache_root, platform=sys_platform
    )
    xdg = tmp_path / "xdg"
    assert cache_root(environ={"XDG_CACHE_HOME": str(xdg)}) == xdg / "pybads"
    # A relative XDG_CACHE_HOME is ignored, as the XDG specification asks,
    # one that starts with "~" included: "~" is not expanded there
    for environ in (
        {},
        {"XDG_CACHE_HOME": ""},
        {"XDG_CACHE_HOME": "  "},
        {"XDG_CACHE_HOME": "cache"},
        {"XDG_CACHE_HOME": "~/xdg"},
    ):
        assert cache_root(environ=environ) == home / ".cache" / "pybads"


@pytest.mark.parametrize("sys_platform", ["win32", "darwin", "linux"])
def test_no_cache_directory_without_a_home(
    monkeypatch, tmp_path, sys_platform
):
    # os.path.expanduser leaves "~" as it is when it finds no home, and
    # Path.expanduser then raises
    monkeypatch.setattr(os.path, "expanduser", lambda path: path)
    cache_root = functools.partial(
        _release_reminder._cache_root, platform=sys_platform
    )
    assert cache_root(environ={"PYBADS_CACHE_DIR": "~/mine"}) is None
    if sys_platform == "win32":
        environ = {"LOCALAPPDATA": str(tmp_path / "Local")}
        assert cache_root(environ=environ) == tmp_path / "Local" / "pybads"
    else:
        assert cache_root(environ={"XDG_CACHE_HOME": "cache"}) is None
    if sys_platform == "linux":
        environ = {"XDG_CACHE_HOME": str(tmp_path / "xdg")}
        assert cache_root(environ=environ) == tmp_path / "xdg" / "pybads"


@pytest.mark.parametrize("sys_platform", ["win32", "darwin", "linux"])
def test_cache_directory_variable_overrides_the_platform(
    home, tmp_path, sys_platform
):
    cache_root = functools.partial(
        _release_reminder._cache_root, platform=sys_platform
    )
    environ = {
        "LOCALAPPDATA": str(tmp_path / "Local"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg"),
    }

    def overridden(value):
        return cache_root(environ={**environ, "PYBADS_CACHE_DIR": value})

    assert overridden(str(tmp_path / "mine")) == tmp_path / "mine"
    assert overridden("~/mine") == home / "mine"
    # A blank value is no override
    assert overridden("  ") == cache_root(environ=environ)


@pytest.mark.parametrize(
    "days, logged",
    [(0, False), (100, False), (365, False), (366, True), (1000, True)],
)
def test_logs_only_for_a_release_older_than_the_threshold(
    reminder, days, logged
):
    today = RELEASED + datetime.timedelta(days=days)
    assert _consider(reminder, today=today) is logged
    assert (len(reminder.records) == 1) is logged
    assert reminder.state_path.exists() is logged


@pytest.mark.parametrize(
    "released, today, logged",
    [
        # A year that holds 29 February has 366 days: the anniversary
        # itself is not "more than a year" after the release.
        ("2027-03-01", datetime.date(2028, 3, 1), False),
        ("2027-03-01", datetime.date(2028, 3, 2), True),
        ("2028-02-29", datetime.date(2029, 2, 28), False),
        ("2028-02-29", datetime.date(2029, 3, 1), True),
    ],
)
def test_threshold_is_the_first_anniversary(reminder, released, today, logged):
    assert _consider(reminder, release_date=released, today=today) is logged
    assert (len(reminder.records) == 1) is logged


@pytest.mark.parametrize(
    "released, today, age",
    [
        ("2026-03-15", datetime.date(2027, 3, 16), "more than a year ago"),
        ("2026-03-15", datetime.date(2028, 3, 15), "more than a year ago"),
        ("2026-03-15", datetime.date(2028, 3, 16), "more than 2 years ago"),
        ("2026-03-15", datetime.date(2028, 9, 15), "more than 2 years ago"),
        ("2026-03-15", datetime.date(2031, 3, 16), "more than 5 years ago"),
        ("2028-02-29", datetime.date(2030, 2, 28), "more than a year ago"),
        ("2028-02-29", datetime.date(2030, 3, 1), "more than 2 years ago"),
    ],
)
def test_record_names_the_version_and_the_age(reminder, released, today, age):
    """One INFO record: the sentence, PyPI's address on a line of its own,
    and an empty line before the display that follows, as a tip ends."""
    assert _consider(reminder, release_date=released, today=today)
    [record] = reminder.records
    assert record.levelno == logging.INFO
    assert record.getMessage() == (
        f"Note: PyBADS 1.5.0 was released {age}. Run "
        "pybads.check_for_updates() to see whether a newer version is "
        "available.\n"
        "https://pypi.org/project/pybads/\n"
    )


def test_only_the_third_showing_is_the_last(reminder):
    for days in (0, 90, 180):
        _new_session()
        assert _consider(reminder, today=_day(days))

    messages = _messages(reminder)
    assert [LAST in message for message in messages] == [False, False, True]
    assert (
        messages[2]
        .splitlines()[0]
        .endswith(
            "to see whether a newer version is available. "
            "This is the last reminder for PyBADS 1.5.0."
        )
    )
    assert messages[2] == _release_reminder.format_reminder(
        INSTALLED, RELEASED, _day(180), last=True
    )


def test_logs_once_per_session_and_again_after_a_reset(reminder):
    assert _consider(reminder)
    assert not _consider(reminder, today=_day(90))
    assert not _consider(reminder, today=_day(90), installed="1.5.1")
    assert len(reminder.records) == 1

    _new_session()
    assert _consider(reminder, today=_day(90))


def test_cap_three_showings_per_version_at_least_ninety_days_apart(reminder):
    path = reminder.state_path

    def start(days, installed=INSTALLED):
        _new_session()
        return _consider(reminder, today=_day(days), installed=installed)

    assert start(0)
    assert not start(1)
    assert not start(89)
    assert start(90)
    assert not start(179)
    assert start(180)
    assert not start(270)
    assert not start(2000)
    assert start(2000, installed="1.5.1")

    assert len(reminder.records) == 4
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "1.5.0": [
            _day(0).isoformat(),
            _day(90).isoformat(),
            _day(180).isoformat(),
        ],
        "1.5.1": [_day(2000).isoformat()],
    }
    assert [item.name for item in path.parent.iterdir()] == [path.name]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"\xff\xfe", id="not-utf-8"),
        pytest.param(b"[]", id="not-a-mapping"),
        pytest.param(b'{"1.5.0": "2027-07-28"}', id="not-a-list"),
        pytest.param(b'{"1.5.0": [20270728]}', id="not-a-string"),
        pytest.param(b'{"1.5.0": ["yesterday"]}', id="not-a-date"),
        pytest.param(b'{"1.5.0": ["2027-02-30"]}', id="invalid-date"),
        pytest.param(b'{"1.5.0.dev1": []}', id="not-a-release"),
        pytest.param(
            b"{}" + b" " * _release_reminder._MAX_STATE_BYTES, id="oversized"
        ),
    ],
)
def test_malformed_state_starts_afresh(reminder, content):
    path = reminder.state_path
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    assert _consider(reminder)
    assert json.loads(path.read_text("utf-8")) == {
        INSTALLED: [ELIGIBLE_DAY.isoformat()]
    }
    _new_session()
    assert not _consider(reminder, today=_day(1))

    messages = _messages(reminder)
    assert len(messages) == 1
    assert LAST not in messages[0]


def test_unreadable_state_falls_back_to_once_per_session(reminder):
    path = reminder.state_path
    path.mkdir(parents=True)

    assert _consider(reminder)
    assert not _consider(reminder, today=_day(90))
    _new_session()
    assert _consider(reminder, today=_day(1))

    assert len(reminder.records) == 2
    assert path.is_dir()


def _no_cache_directory():
    raise RuntimeError("Could not determine home directory.")


@pytest.mark.parametrize(
    "default_state_path",
    [lambda: None, _no_cache_directory],
    ids=["none", "raises"],
)
def test_no_state_file_falls_back_to_once_per_session(
    reminder, eligible, monkeypatch, default_state_path
):
    """Without a cache directory, the reminder is logged once per session
    and nothing is written."""
    monkeypatch.setattr(
        _release_reminder, "_default_state_path", default_state_path
    )
    writes = []
    monkeypatch.setattr(
        _release_reminder,
        "_write_state",
        lambda path, state: writes.append((path, state)),
    )
    consider = functools.partial(
        _release_reminder.consider_release_reminder,
        logger=reminder.logger,
        enabled=True,
    )

    assert consider()
    assert not consider()
    _new_session()
    monkeypatch.setattr(_release_reminder, "_today", lambda: _day(1))
    assert consider()
    assert len(reminder.records) == 2
    assert writes == []


def test_unwritable_state_falls_back_to_once_per_session(
    reminder, monkeypatch
):
    def refuse(source, destination):
        raise PermissionError("read-only cache")

    monkeypatch.setattr(os, "replace", refuse)
    path = reminder.state_path

    assert _consider(reminder)
    assert not path.exists()
    assert list(path.parent.iterdir()) == []
    assert not _consider(reminder, today=_day(90))
    _new_session()
    assert _consider(reminder, today=_day(1))
    assert len(reminder.records) == 2


def test_write_goes_through_a_temporary_file_and_a_rename(
    reminder, monkeypatch
):
    real_replace = os.replace
    calls = []

    def spy(source, destination):
        source, destination = Path(source), Path(destination)
        calls.append((source, destination, source.read_text("utf-8")))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", spy)
    path = reminder.state_path

    assert _consider(reminder)
    [(source, destination, content)] = calls
    assert destination == path
    assert source.parent == path.parent and source != path
    assert not source.exists()
    assert json.loads(content) == {INSTALLED: [ELIGIBLE_DAY.isoformat()]}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"installed": "1.5.0.dev3"},
        {"installed": "1.5.1.dev2+g1234abcd"},
        {"installed": "1.5.0+local"},
        {"installed": "1.5.0rc1"},
        {"installed": "1.5"},
        {"release_date": "not a date"},
        {"release_date": "2026-02-30"},
        {"release_date": "20260315"},
        {"today": RELEASED - datetime.timedelta(days=1)},
        {"today": datetime.date(1970, 1, 1)},
        {"interactive": False},
        {"environ": {"CI": "true"}},
        {"environ": {"PYBADS_NO_UPDATE_REMINDER": "1"}},
        {"environ": {"NO_UPDATE_NOTIFIER": "1"}},
        {"enabled": False},
    ],
)
def test_ineligible_start_logs_writes_and_uses_up_nothing(reminder, kwargs):
    assert not _consider(reminder, **kwargs)
    assert reminder.records == []
    assert not reminder.state_path.parent.exists()

    assert _consider(reminder)


@pytest.mark.parametrize(
    "level",
    [logging.WARNING, _LOG_NOTIFY, _LOG_FINAL],
    ids=["warning", "notify", "final"],
)
def test_logger_above_info_logs_writes_and_uses_up_nothing(reminder, level):
    reminder.logger.setLevel(level)
    assert not _consider(reminder)
    assert reminder.records == []
    assert not reminder.state_path.parent.exists()

    reminder.logger.setLevel(logging.INFO)
    assert _consider(reminder)


@pytest.mark.parametrize("value", ["", "0", "false", "False", " FALSE "])
def test_unset_opt_out_values_do_not_silence(reminder, value):
    environ = {name: value for name in _release_reminder.OPT_OUT_VARIABLES}
    assert _consider(reminder, environ=environ)


def test_defaults_read_the_release_the_metadata_the_clock_and_the_environment(
    reminder, eligible, monkeypatch
):
    def consider():
        return _release_reminder.consider_release_reminder(
            logger=reminder.logger, enabled=True
        )

    monkeypatch.setattr(_release, "RELEASE_DATE", None)
    assert not consider()
    monkeypatch.setattr(_release, "RELEASE_DATE", RELEASE_DATE)
    monkeypatch.setattr(_release_reminder, "_installed_version", lambda: None)
    assert not consider()
    monkeypatch.setattr(
        _release_reminder, "_installed_version", lambda: INSTALLED
    )
    for name in _release_reminder.OPT_OUT_VARIABLES:
        monkeypatch.setenv(name, "1")
        assert not consider()
        monkeypatch.delenv(name)
    monkeypatch.setattr(
        _release_reminder,
        "_today",
        lambda: RELEASED + datetime.timedelta(days=365),
    )
    assert not consider()
    monkeypatch.setattr(_release_reminder, "_today", lambda: ELIGIBLE_DAY)
    monkeypatch.setattr(_release_reminder, "_is_interactive", lambda: False)
    assert not consider()
    assert reminder.records == []
    assert not reminder.state_path.parent.exists()

    monkeypatch.setattr(_release_reminder, "_is_interactive", lambda: True)
    assert consider()
    assert _messages(reminder) == [
        _release_reminder.format_reminder(
            INSTALLED, RELEASED, ELIGIBLE_DAY, last=False
        )
    ]
    assert json.loads(reminder.state_path.read_text(encoding="utf-8")) == {
        INSTALLED: [ELIGIBLE_DAY.isoformat()]
    }


def test_installed_version_reads_the_package_metadata(monkeypatch):
    assert _release_reminder._installed_version() == (
        importlib.metadata.version("pybads")
    )

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_release_reminder, "version", missing)
    assert _release_reminder._installed_version() is None


class _Stream:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        if isinstance(self.tty, Exception):
            raise self.tty
        return self.tty

    def write(self, text):
        return len(text)

    def flush(self):
        pass


def test_interactive_session_is_a_terminal_or_an_ipython_kernel(monkeypatch):
    is_interactive = _release_reminder._is_interactive
    monkeypatch.delitem(sys.modules, "IPython", raising=False)

    monkeypatch.setattr(sys, "stdout", _Stream(True))
    assert is_interactive()
    for stream in (
        _Stream(False),
        _Stream(ValueError("closed")),
        object(),
        None,
    ):
        monkeypatch.setattr(sys, "stdout", stream)
        assert not is_interactive()
    assert "IPython" not in sys.modules

    monkeypatch.setattr(sys, "stdout", _Stream(False))
    for shell, interactive in (
        (SimpleNamespace(kernel=object()), True),
        (SimpleNamespace(kernel=None), False),
        (SimpleNamespace(), False),
        (None, False),
    ):
        monkeypatch.setitem(
            sys.modules,
            "IPython",
            SimpleNamespace(get_ipython=lambda shell=shell: shell),
        )
        assert is_interactive() is interactive


def test_a_start_that_shows_the_reminder_leaves_the_tips_cadence(reminder):
    """The tip that the reminder displaces is the one the next eligible
    start shows."""
    assert (
        rt.consider_runtime_tip(
            logger=reminder.logger,
            enabled=True,
            release_reminder_shown=True,
        )
        is None
    )
    assert reminder.records == []
    assert rt._ELIGIBLE_STARTS == 0
    assert rt._ORDER is None
    assert rt._SEEN_IDS == set()

    tip = rt.consider_runtime_tip(logger=reminder.logger, enabled=True)
    assert tip == _first_tip()
    assert rt._ELIGIBLE_STARTS == 1


def test_reminder_draws_from_no_random_stream(reminder):
    np_saved, random_saved = np.random.get_state(), random.getstate()
    try:
        np.random.seed(7)
        random.seed(7)
        np_before, random_before = np.random.get_state(), random.getstate()
        tips_before = rt._RNG.getstate()

        assert _consider(reminder)

        np_after = np.random.get_state()
        assert np_before[0] == np_after[0]
        assert np.array_equal(np_before[1], np_after[1])
        assert np_before[2:] == np_after[2:]
        assert random.getstate() == random_before
        assert rt._RNG.getstate() == tips_before
    finally:
        np.random.set_state(np_saved)
        random.setstate(random_saved)


def test_a_forked_child_gets_a_lock_of_its_own_and_keeps_the_session_flag(
    reminder, monkeypatch
):
    """The hook that `os.register_at_fork` runs in a forked child gives it a
    free lock, even when a thread of the parent held its own at the fork;
    the reminder that the parent showed counts as shown in the child."""
    assert _consider(reminder)
    lock = _release_reminder._STATE_LOCK
    # Restored after the test
    monkeypatch.setattr(_release_reminder, "_STATE_LOCK", lock)
    with lock:
        _release_reminder._after_fork_in_child()
    child_lock = _release_reminder._STATE_LOCK
    assert child_lock is not lock
    assert isinstance(child_lock, type(threading.Lock()))
    assert not child_lock.locked()
    assert _release_reminder._SHOWN_THIS_SESSION is True
    assert not _consider(reminder, today=_day(90))
    assert len(reminder.records) == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_the_fork_hook_is_registered(reminder):
    """A real fork runs the hook: the child's lock is free although the
    parent held its own when it forked, and the session flag is the
    parent's."""
    assert _consider(reminder)
    read_end, write_end = os.pipe()
    with _release_reminder._STATE_LOCK:
        pid = os.fork()
        if pid == 0:  # the child reports and leaves at once
            try:
                own = _release_reminder._STATE_LOCK.acquire(blocking=False)
                kept = _release_reminder._SHOWN_THIS_SESSION is True
                os.write(write_end, b"1" if own and kept else b"0")
            finally:
                os._exit(0)
    os.close(write_end)
    report = os.read(read_end, 1)
    os.close(read_end)
    os.waitpid(pid, 0)
    assert report == b"1"


class _BrokenStream:
    """A stream whose writes fail, as those of a closed pipe do."""

    def __init__(self):
        self.writes = 0

    def write(self, text):
        self.writes += 1
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


@pytest.mark.parametrize("raise_exceptions", [True, False])
def test_a_failing_handler_stops_neither_the_reminder_nor_the_tip(
    reminder, monkeypatch, capsys, raise_exceptions
):
    """`logging` reports the failure of a handler's stream (on standard
    error, while `logging.raiseExceptions` holds) and raises nothing. The
    logger cannot tell that a handler failed: the reminder counts as
    shown."""
    monkeypatch.setattr(logging, "raiseExceptions", raise_exceptions)
    stream = _BrokenStream()
    handler = logging.StreamHandler(stream)
    reminder.logger.addHandler(handler)
    try:
        assert _consider(reminder)
        assert stream.writes == 1
        assert (
            rt.consider_runtime_tip(
                logger=reminder.logger,
                enabled=True,
                release_reminder_shown=True,
            )
            is None
        )
        tip = rt.consider_runtime_tip(logger=reminder.logger, enabled=True)
        assert tip == _first_tip()
        assert stream.writes == 2
    finally:
        reminder.logger.removeHandler(handler)
    assert json.loads(reminder.state_path.read_text(encoding="utf-8")) == {
        INSTALLED: [ELIGIBLE_DAY.isoformat()]
    }
    error = capsys.readouterr().err
    assert ("BrokenPipeError" in error) is raise_exceptions


def test_release_date_matches_the_changelog():
    changelog = _REPO / "CHANGELOG.md"
    if not changelog.is_file():
        pytest.skip("CHANGELOG.md is not beside the package")
    text = changelog.read_text(encoding="utf-8")
    if "changes to PyBADS" not in text:
        pytest.skip("the CHANGELOG.md beside the package is not PyBADS's")
    # The first section heading other than [Unreleased] is the latest
    # release, and it must be written in the form the date is read from.
    released = [
        line
        for line in text.splitlines()
        if line.startswith("## [") and not line.startswith("## [Unreleased]")
    ]

    if not released:
        assert _release.RELEASE_DATE is None
    else:
        heading = _RELEASE_HEADING.fullmatch(released[0].rstrip())
        assert heading is not None, released[0]
        assert _release.RELEASE_DATE == heading.group(2)
        assert _release_reminder._parse_date(_release.RELEASE_DATE)


# Whole runs


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def _make_bads(**options):
    opts = {"display": "iter", "max_fun_evals": 30, "random_seed": 3}
    opts.update(options)
    return BADS(
        _sphere,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=opts,
    )


def _bads_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "BADS"]


def _run(caplog, **options):
    """The messages of the `BADS` logger in a run."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="BADS"):
        _make_bads(**options).optimize()
    return _bads_messages(caplog)


def _index(messages, prefix):
    return next(i for i, m in enumerate(messages) if m.startswith(prefix))


def _tips(messages):
    return [m for m in messages if m.startswith("Tip: ")]


def _notes(messages):
    return [m for m in messages if NOTE in m]


@pytest.mark.parametrize("display", ["iter", "all", "full"])
def test_the_reminder_takes_the_tips_place_before_the_column_headers(
    reminder, eligible, caplog, display
):
    """The reminder is one INFO record of the `BADS` logger, after the
    opening message and before the column headers; no tip shows, and the
    next run shows the tip that it displaced."""
    messages = _run(caplog, display=display)
    note = _release_reminder.format_reminder(
        INSTALLED, RELEASED, ELIGIBLE_DAY, last=False
    )
    assert _notes(messages) == [note]
    [record] = [r for r in caplog.records if r.getMessage() == note]
    assert record.name == "BADS"
    assert record.levelno == logging.INFO
    opening = _index(messages, "Beginning optimization")
    headers = _index(messages, " Iteration")
    assert opening < messages.index(note) < headers
    assert _tips(messages) == []
    assert rt._ELIGIBLE_STARTS == 0
    assert rt._ORDER is None
    assert rt._SEEN_IDS == set()

    messages = _run(caplog, display=display)
    assert _notes(messages) == []
    assert _tips(messages) == [rt.format_tip(_first_tip())]
    assert rt._ELIGIBLE_STARTS == 1


@pytest.mark.parametrize(
    "cause",
    ["shown_this_session", "not_interactive", "recent_release", "opted_out"],
)
def test_a_run_without_the_reminder_shows_its_tip_as_before(
    reminder, eligible, monkeypatch, caplog, cause
):
    if cause == "shown_this_session":
        monkeypatch.setattr(_release_reminder, "_SHOWN_THIS_SESSION", True)
    elif cause == "not_interactive":
        monkeypatch.setattr(
            _release_reminder, "_is_interactive", lambda: False
        )
    elif cause == "recent_release":
        monkeypatch.setattr(
            _release_reminder,
            "_today",
            lambda: RELEASED + datetime.timedelta(days=100),
        )
    else:
        monkeypatch.setenv("PYBADS_NO_UPDATE_REMINDER", "1")
    messages = _run(caplog)
    assert _notes(messages) == []
    tip = rt.format_tip(_first_tip())
    assert _tips(messages) == [tip]
    assert (
        _index(messages, "Beginning optimization")
        < messages.index(tip)
        < _index(messages, " Iteration")
    )
    assert rt._ELIGIBLE_STARTS == 1


@pytest.mark.parametrize(
    "options",
    [
        {"display": "notify"},
        {"display": "final"},
        {"display": "off"},
        {"show_tips": False},
    ],
    ids=["notify", "final", "off", "tips_off"],
)
def test_run_options_silence_the_reminder_without_using_it_up(
    reminder, eligible, caplog, options
):
    assert _notes(_run(caplog, **options)) == []
    assert _release_reminder._SHOWN_THIS_SESSION is False
    assert not reminder.state_path.parent.exists()
    assert rt._ELIGIBLE_STARTS == 0

    assert _consider(reminder)


def test_the_logger_when_the_run_starts_decides(reminder, eligible, caplog):
    """The `BADS` logger is shared, and the last `BADS` object created sets
    its level: a run whose lines another object has silenced shows no
    reminder and does not use it up."""
    with caplog.at_level(logging.INFO, logger="BADS"):
        bads = _make_bads()
        _make_bads(display="off")
        bads.optimize()
    assert _notes(_bads_messages(caplog)) == []
    assert _release_reminder._SHOWN_THIS_SESSION is False
    assert not reminder.state_path.parent.exists()


def _result_summary(bads, result):
    log = bads.function_logger
    X, Y = log.X[log.X_flag].copy(), log.Y[log.X_flag].copy()
    if not _REPEATS_BIT_FOR_BIT:
        n = bads.optim_state["eff_starting_points"]
        return (X[:n], Y[:n])
    return (
        np.asarray(result["x"]).copy(),
        result["fval"],
        result["func_count"],
        X,
        Y,
    )


def test_the_reminder_changes_nothing_in_its_run(reminder, eligible, caplog):
    """A seeded run that shows the reminder draws and computes what the same
    run without it does, and leaves NumPy's global random state, that of the
    `random` module and that of the tips' generator as it found them."""
    np_saved, random_saved = np.random.get_state(), random.getstate()
    try:
        np.random.seed(7)
        random.seed(7)
        np_before, random_before = np.random.get_state(), random.getstate()
        tips_before = rt._RNG.getstate()
        with caplog.at_level(logging.INFO, logger="BADS"):
            with_note = _make_bads()
            summary_with = _result_summary(with_note, with_note.optimize())
        assert len(_notes(_bads_messages(caplog))) == 1
        np_after, random_after = np.random.get_state(), random.getstate()
        assert np_before[0] == np_after[0]
        assert np.array_equal(np_before[1], np_after[1])
        assert np_before[2:] == np_after[2:]
        assert random_before == random_after
        assert rt._RNG.getstate() == tips_before

        without = _make_bads(show_tips=False)
        summary_without = _result_summary(without, without.optimize())
        for a, b in zip(summary_with, summary_without):
            assert np.array_equal(a, b)
        if _REPEATS_BIT_FOR_BIT:
            assert (
                with_note.rng.bit_generator.state
                == without.rng.bit_generator.state
            )
    finally:
        np.random.set_state(np_saved)
        random.setstate(random_saved)


def test_a_failing_handler_does_not_stop_the_run(
    reminder, eligible, monkeypatch
):
    monkeypatch.setattr(logging, "raiseExceptions", False)
    stream = _BrokenStream()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("BADS")
    logger.addHandler(handler)
    try:
        result = _make_bads().optimize()
    finally:
        logger.removeHandler(handler)
    assert result["func_count"] == 30
    assert stream.writes > 0
    assert json.loads(reminder.state_path.read_text(encoding="utf-8")) == {
        INSTALLED: [ELIGIBLE_DAY.isoformat()]
    }
