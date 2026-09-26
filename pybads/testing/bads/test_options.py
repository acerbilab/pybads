"""The checks of the options that `BADS` is given, as MATLAB BADS's
`setupoptions.m` makes them."""

import logging

import numpy as np
import pytest

from pybads import BADS

D = 3


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def _make_bads(**options):
    return BADS(
        _sphere,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={"display": "off", "random_seed": 3, **options},
    )


@pytest.mark.parametrize("max_fun_evals", [0, -5, 30.5, np.nan, "200*D"])
def test_max_fun_evals_must_be_a_positive_integer(max_fun_evals):
    with pytest.raises(ValueError, match="max_fun_evals.*positive integer"):
        _make_bads(max_fun_evals=max_fun_evals)


@pytest.mark.parametrize(
    "max_fun_evals",
    [30, 30.0, np.float64(30.0)],
    ids=["int", "float", "float64"],
)
def test_max_fun_evals_whole_number_is_an_integer(max_fun_evals):
    bads = _make_bads(max_fun_evals=max_fun_evals)
    assert bads.options["max_fun_evals"] == 30
    assert type(bads.options["max_fun_evals"]) is int


def test_max_fun_evals_can_be_infinite():
    """MATLAB BADS's check accepts an infinite budget too."""
    bads = _make_bads(max_fun_evals=np.inf)
    assert bads.options["max_fun_evals"] == np.inf


def test_improvement_quantile_above_half_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="BADS"):
        _make_bads(improvement_quantile=0.9)
    assert any(
        record.name == "BADS"
        and "improvement_quantile'] is greater than 0.5" in record.getMessage()
        for record in caplog.records
    )


def test_improvement_quantile_default_is_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="BADS"):
        _make_bads()
    assert not any(
        "improvement_quantile" in record.getMessage()
        for record in caplog.records
    )
