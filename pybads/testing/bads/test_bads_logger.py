"""The messages of a run go to the `BADS` logger, whose level the `display`
option sets."""

import logging
import re
from pathlib import Path

import numpy as np
import pytest

import pybads
from pybads import BADS

D = 3


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


# With the root logger at DEBUG, the poll leaves NumPy's warnings on, and its
# probability of improvement divides by a zero predicted SD.
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_messages_on_bads_logger(caplog):
    """Every message that PyBADS's modules log during this seeded run comes
    from the `BADS` logger, including the debug message of a stalling run,
    which the run reaches. The root logger is at DEBUG, so that a message on
    another logger would be captured too."""
    bads = BADS(
        _sphere,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={"display": "full", "max_fun_evals": 100, "random_seed": 0},
    )
    with caplog.at_level(logging.DEBUG):
        bads.optimize()
    package_dir = Path(pybads.__file__).resolve().parent
    records = [
        record
        for record in caplog.records
        if package_dir in Path(record.pathname).resolve().parents
    ]
    assert any(
        "optimization is stalling" in record.getMessage() for record in records
    )
    assert {record.name for record in records} == {"BADS"}


def _kind(message):
    """The kind of a message of the display: the opening message, an
    iteration line (or their header), the final message, or None."""
    if message.startswith("Beginning optimization"):
        return "opening"
    if message.startswith(" Iteration") or re.match(
        r"\s*\d+\s+\d+\s", message
    ):
        return "iteration"
    if message.startswith("Optimization terminated") or (
        "value at minimum" in message
    ):
        return "final"
    return None


def _display_levels(display, caplog):
    """The levels of the records of each kind that a short run with the
    given `display` logs, by kind."""
    bads = BADS(
        _sphere,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={"display": display, "max_fun_evals": 30, "random_seed": 0},
    )
    caplog.clear()
    bads.optimize()
    levels = {}
    for record in caplog.records:
        kind = _kind(record.getMessage())
        if record.name == "BADS" and kind is not None:
            levels.setdefault(kind, set()).add(record.levelno)
    return levels


def test_display_message_levels(caplog):
    """The opening message and the final message are logged above the
    iteration lines, at INFO, and below the warnings."""
    levels = _display_levels("iter", caplog)
    assert levels["iteration"] == {logging.INFO}
    (final,) = levels["final"]
    (opening,) = levels["opening"]
    assert logging.INFO < final < opening < logging.WARNING


# The kinds of messages that each display shows, besides the warnings
_SHOWN = {
    "off": set(),
    "notify": {"opening"},
    "final": {"opening", "final"},
    "iter": {"opening", "iteration", "final"},
    "full": {"opening", "iteration", "final"},
    "none": set(),
    "OFF": set(),
    "Final": {"opening", "final"},
    "all": {"opening", "iteration", "final"},
}


@pytest.mark.parametrize("display", list(_SHOWN))
def test_display_levels(display, caplog):
    """`display` is read from its first three letters, lower case, as in
    MATLAB BADS (`bads.m`): "off" or "none" shows no message but the
    warnings, "notify" the opening message, "final" also the final message,
    "iter" or "all" also the iteration lines, and "full" the debug messages
    too."""
    assert set(_display_levels(display, caplog)) == _SHOWN[display]
    if display == "full":
        assert logging.getLogger("BADS").level == logging.DEBUG
