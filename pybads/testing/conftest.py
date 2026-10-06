"""Fixtures shared by the whole shipped test suite."""

import pytest

from pybads.bads import _release_reminder


@pytest.fixture(scope="session")
def _release_reminder_directory(tmp_path_factory):
    """A directory for the old-release reminder's state file."""
    return tmp_path_factory.mktemp("release_reminder")


def _make_inert(patch, directory):
    patch.setattr(
        _release_reminder,
        "_default_state_path",
        lambda: directory / _release_reminder.STATE_FILE_NAME,
    )
    patch.setattr(_release_reminder, "_SHOWN_THIS_SESSION", True)


@pytest.fixture(scope="session", autouse=True)
def _inert_release_reminder_for_the_session(_release_reminder_directory):
    """The reminder kept inert for the fixtures of a module or a session,
    which are set up before a test's own fixtures."""
    with pytest.MonkeyPatch.context() as patch:
        _make_inert(patch, _release_reminder_directory)
        yield


@pytest.fixture(autouse=True)
def _inert_release_reminder(monkeypatch, _release_reminder_directory):
    """Keep the old-release reminder silent and out of the user's cache.

    The reminder counts as shown in this session, so that no test's output
    depends on the calendar: a year after a release, the reminder would
    otherwise be logged at the first run started in an interactive session
    and take the place of that run's tip. Its state file lies in pytest's
    temporary directory. ``test_release_reminder.py`` resets the flag to
    test the reminder; the flag is set again after each test.
    """
    _make_inert(monkeypatch, _release_reminder_directory)
