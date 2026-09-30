"""The old-release reminder at the start of a run.

PyBADS ships the date of its release (``pybads._release.RELEASE_DATE``).
When a run starts in an interactive session and the installed release is
more than a year old, a note says that a newer version may exist and how to
find out, with PyPI's address on a line of its own. It is one INFO record
of the run's logger, in the slot of a runtime tip (``BADS._init_mesh_``),
which it takes. The reminder opens no network connection and writes no file
but its state file.

The state file, ``update_reminder.json`` in PyBADS's user cache directory,
maps each installed version to the dates of its showings, which caps the
reminder at three showings per version, at least 90 days apart. A file
whose content is malformed is started afresh. A process-local flag allows
one showing per Python session. When the file cannot be read, that flag is
the only cap; when it can be read but not written, the showings it records
still count. A showing counts when it is logged, wherever the logger's
handlers send it.
"""

import datetime
import json
import logging
import os
import re
import sys
import tempfile
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pybads import _release

# The FAQ (docsrc/source/faq.md) quotes this wording; change both together.
REMINDER_TEMPLATE = (
    "Note: PyBADS {version} was released {age}. Run "
    "pybads.check_for_updates() to see whether a newer version is available."
)
LAST_REMINDER_TEMPLATE = "This is the last reminder for PyBADS {version}."
AGE_ONE_YEAR = "more than a year ago"
AGE_YEARS_TEMPLATE = "more than {years} years ago"
PYPI_URL = "https://pypi.org/project/pybads/"

THRESHOLD_YEARS = 1
SPACING_DAYS = 90
MAX_SHOWINGS = 3
STATE_FILE_NAME = "update_reminder.json"
CACHE_DIR_VARIABLE = "PYBADS_CACHE_DIR"
OPT_OUT_VARIABLES = ("CI", "PYBADS_NO_UPDATE_REMINDER", "NO_UPDATE_NOTIFIER")
# Values of an opt-out variable that leave it unset, as CI detection reads
# them.
UNSET_VALUES = frozenset({"", "0", "false"})

_RELEASE_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_MAX_STATE_BYTES = 64 * 1024

_STATE_LOCK = threading.Lock()
_SHOWN_THIS_SESSION = False


def _today():
    """The date of the run, from the clock."""
    return datetime.date.today()


def _installed_version():
    """The installed version of PyBADS, or ``None`` if unknown."""
    try:
        return version("pybads")
    except PackageNotFoundError:
        return None


def _is_interactive():
    """Whether output reaches a terminal or an IPython kernel.

    IPython is consulted only when the session has imported it already.
    """
    stdout = sys.stdout
    try:
        if stdout is not None and stdout.isatty():
            return True
    except (AttributeError, OSError, ValueError):
        pass
    get_ipython = getattr(sys.modules.get("IPython"), "get_ipython", None)
    if get_ipython is None:
        return False
    try:
        shell = get_ipython()
    except Exception:
        return False
    return getattr(shell, "kernel", None) is not None


def _cache_root(environ=None, platform=None):
    """PyBADS's user cache directory, or ``None`` when none is found.

    The directory that ``PYBADS_CACHE_DIR`` names, or else PyBADS's
    directory in the user cache, at the place where, on a usual setup,
    ``platformdirs.user_cache_dir("pybads", appauthor=False,
    opinion=False)`` puts it (PyBADS does not depend on platformdirs):
    ``%LOCALAPPDATA%\\pybads`` on Windows, ``~/Library/Caches/pybads`` on
    macOS, ``$XDG_CACHE_HOME/pybads`` or, when that variable is not an
    absolute path, ``~/.cache/pybads`` elsewhere. ``None`` when
    ``LOCALAPPDATA`` is not an absolute path on Windows, or the home
    directory cannot be found. The directory is not created.
    """
    if environ is None:
        environ = os.environ
    if platform is None:
        platform = sys.platform
    override = environ.get(CACHE_DIR_VARIABLE, "").strip()
    try:
        if override:
            return Path(override).expanduser()
        if platform == "win32":
            root = Path(environ.get("LOCALAPPDATA", "").strip())
        elif platform == "darwin":
            root = Path("~/Library/Caches").expanduser()
        else:
            root = Path(environ.get("XDG_CACHE_HOME", "").strip())
            # The XDG specification ignores a relative path
            if not root.is_absolute():
                root = Path("~/.cache").expanduser()
    except RuntimeError:
        # Path.expanduser raises when it cannot find the home directory
        return None
    if not root.is_absolute():
        return None
    return root / "pybads"


def _default_state_path():
    """The state file in PyBADS's user cache directory, or ``None``."""
    root = _cache_root()
    return None if root is None else root / STATE_FILE_NAME


def _parse_date(value):
    """The date of an ISO ``YYYY-MM-DD`` string, or ``None``."""
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _years_passed(released, today):
    """The number of anniversaries of the release before *today*."""
    years = today.year - released.year
    if (today.month, today.day) <= (released.month, released.day):
        years -= 1
    return years


def _age_phrase(released, today):
    """How long ago the release was, in whole years exceeded."""
    years = _years_passed(released, today)
    if years >= 2:
        return AGE_YEARS_TEMPLATE.format(years=years)
    return AGE_ONE_YEAR


def _opted_out(environ):
    """Whether an opt-out variable is set to a value that counts."""
    return any(
        environ.get(name, "").strip().lower() not in UNSET_VALUES
        for name in OPT_OUT_VARIABLES
    )


def _read_state(path):
    """The recorded showings, or ``None`` when the file is unusable.

    A missing file holds no showings, and so does a file whose content is
    anything but a mapping from release versions to lists of ISO dates: the
    next write replaces it. A file that cannot be read is unusable.
    """
    if path is None:
        return None
    try:
        with open(path, "rb") as stream:
            payload = stream.read(_MAX_STATE_BYTES + 1)
    except FileNotFoundError:
        return {}
    except Exception:
        return None
    if len(payload) > _MAX_STATE_BYTES:
        return {}
    try:
        state = json.loads(payload.decode("utf-8"))
    except Exception:
        return {}
    if not isinstance(state, dict):
        return {}
    for key, dates in state.items():
        if (
            not _RELEASE_VERSION.fullmatch(key)
            or not isinstance(dates, list)
            or any(_parse_date(date) is None for date in dates)
        ):
            return {}
    return state


def _write_state(path, state):
    """Replace the state file through a temporary file and a rename.

    Every error is swallowed; the return value says whether the file was
    replaced.
    """
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".update_reminder.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(state, stream, sort_keys=True)
        os.replace(temporary, path)
        temporary = None
        return True
    except Exception:
        return False
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def format_reminder(installed, released, today, last):
    """The message of the reminder: its sentence, the last-reminder
    sentence on the last showing, PyPI's address on a line of its own, and
    an empty line before the display that follows, as a tip ends."""
    message = REMINDER_TEMPLATE.format(
        version=installed, age=_age_phrase(released, today)
    )
    if last:
        message += " " + LAST_REMINDER_TEMPLATE.format(version=installed)
    return f"{message}\n{PYPI_URL}\n"


def consider_release_reminder(
    *,
    logger,
    enabled,
    today=None,
    installed=None,
    release_date=None,
    interactive=None,
    environ=None,
    state_path=None,
):
    """
    Consider the old-release reminder for a run that starts, and log it if
    it is due.

    The reminder is logged when the installed version is a final release
    ``X.Y.Z`` released more than a year before the date of the run, in an
    interactive session without an opt-out environment variable, and when
    neither the session nor the state file's cap has used it up. A start
    at which it is not logged changes nothing.

    Parameters
    ----------
    logger : logging.Logger
        The logger of the run, which shows the reminder at INFO.
    enabled : bool
        ``options['show_tips']``, which silences the reminder too.
    today : datetime.date, optional
        The date of the run. The default reads the clock.
    installed : str, optional
        The installed version. The default reads the package metadata.
    release_date : str, optional
        The ISO date of the installed release. The default is
        ``pybads._release.RELEASE_DATE``.
    interactive : bool, optional
        Whether the session is interactive. The default checks whether
        standard output is a terminal or the code runs in an IPython kernel.
    environ : Mapping[str, str], optional
        The environment holding the opt-out variables. The default is
        ``os.environ``.
    state_path : str or os.PathLike, optional
        The state file that records the showings. The default is
        ``update_reminder.json`` in PyBADS's user cache directory.

    Returns
    -------
    shown : bool
        Whether the reminder was logged.
    """
    global _SHOWN_THIS_SESSION

    if not enabled or not logger.isEnabledFor(logging.INFO):
        return False

    with _STATE_LOCK:
        if _SHOWN_THIS_SESSION:
            return False
        if environ is None:
            environ = os.environ
        if _opted_out(environ):
            return False
        if installed is None:
            installed = _installed_version()
        if not isinstance(installed, str) or not _RELEASE_VERSION.fullmatch(
            installed
        ):
            return False
        if release_date is None:
            release_date = _release.RELEASE_DATE
        released = _parse_date(release_date)
        if released is None:
            return False
        if today is None:
            today = _today()
        if _years_passed(released, today) < THRESHOLD_YEARS:
            return False
        if interactive is None:
            interactive = _is_interactive()
        if not interactive:
            return False

        if state_path is None:
            try:
                path = _default_state_path()
            except Exception:
                path = None
        else:
            path = Path(state_path)
        state = _read_state(path)
        dates = [] if state is None else state.get(installed, [])
        if len(dates) >= MAX_SHOWINGS:
            return False
        if dates:
            last = max(_parse_date(date) for date in dates)
            if (today - last).days < SPACING_DAYS:
                return False

        # A handler whose stream fails reports the error through logging
        # and raises nothing, so the reminder cannot stop a run
        logger.info(
            format_reminder(
                installed,
                released,
                today,
                last=len(dates) + 1 == MAX_SHOWINGS,
            )
        )
        _SHOWN_THIS_SESSION = True
        if state is not None:
            state[installed] = [*dates, today.isoformat()]
            _write_state(path, state)
        return True


def _reset_release_reminder_state():
    """Reset the process-local flag, for tests that need it isolated."""
    global _SHOWN_THIS_SESSION
    with _STATE_LOCK:
        _SHOWN_THIS_SESSION = False


def _after_fork_in_child():
    """In a forked child, a lock of its own; the session flag stays the
    parent's."""
    global _STATE_LOCK
    _STATE_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
