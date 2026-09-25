"""The messages of a run go to the `BADS` logger, whose level the `display`
option sets."""

import logging
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
