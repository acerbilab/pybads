"""The options that `BADS` is given: a value of `None` stands for the
default, the boolean options take only booleans, the checks that MATLAB
BADS's `setupoptions.m` makes, and the options that are not supported; and
the option files, whose comment lines describe the options."""

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


def test_none_stands_for_the_default():
    """A user value of `None` leaves the option at its default, as an empty
    value does in MATLAB BADS (`setupoptions.m`): here the log transform of
    a variable whose bounds are positive and span two decades."""
    bads = BADS(
        _sphere,
        np.array([10.0, 1.0]),
        np.array([0.5, -10.0]),
        np.array([200.0, 10.0]),
        np.array([1.0, -5.0]),
        np.array([100.0, 5.0]),
        options={
            "display": "off",
            "random_seed": 3,
            "nonlinear_scaling": None,
        },
    )
    assert bads.options["nonlinear_scaling"] is True
    assert "nonlinear_scaling" not in bads.options["useroptions"]
    assert bads.var_transf.apply_log_t.tolist() == [[True, False]]


def test_none_budget_is_the_default_budget():
    bads = _make_bads(max_fun_evals=None, max_iter=3)
    assert bads.options["max_fun_evals"] == 500 * D
    assert bads.optimize()["iterations"] == 3


def test_none_for_an_unknown_option_is_refused():
    with pytest.raises(ValueError, match="max_fun_eval does not exist"):
        _make_bads(max_fun_eval=None)


@pytest.mark.parametrize("value", ["off", "on", 0, 1], ids=repr)
@pytest.mark.parametrize("name", ["uncertainty_handling", "nonlinear_scaling"])
def test_boolean_options_refuse_other_values(name, value):
    """A MATLAB-style string is not converted: `"off"` would be true."""
    with pytest.raises(
        ValueError, match=rf"options\['{name}'\] needs to be True or False"
    ):
        _make_bads(**{name: value})


@pytest.mark.parametrize("value", [True, False, np.True_, np.False_], ids=repr)
def test_boolean_options_take_booleans(value):
    bads = _make_bads(nonlinear_scaling=value, uncertainty_handling=value)
    assert bads.options["nonlinear_scaling"] is value
    assert bads.optim_state["uncertainty_handling_level"] == int(value)


def test_plot_takes_the_names_of_plots():
    """`plot`, whose default is `False`, also takes the names of MATLAB
    BADS's plots, as its description says."""
    assert _make_bads(plot="scatter").options["plot"] == "scatter"


def test_fun_values_is_not_supported():
    """Prior evaluations, which MATLAB BADS imports (`setupvars.m`), are
    refused; the empty default passes."""
    fun_values = {"X": np.ones((2, D)), "Y": np.array([[3.0], [3.0]])}
    with pytest.raises(ValueError, match="fun_values'] is not supported"):
        _make_bads(fun_values=fun_values)
    assert _make_bads(fun_values={}).options["fun_values"] == {}


def test_f_vals_is_not_supported():
    """`f_vals`, PyBADS's own, filled a cache that nothing reads and stopped
    the run at its first display line."""
    with pytest.raises(ValueError, match="f_vals'] is not supported"):
        _make_bads(f_vals=[48.0])


def test_descriptions_are_whole_comment_lines():
    """An option's description is the whole comment line above it, an `=`
    or a `:` included."""
    descriptions = _make_bads().options.descriptions
    assert descriptions["noise_size"] == (
        "Base observation noise magnitude (SD), e.g. noise_size = 1.0, or a "
        "pair [SD, SD of the prior over log SD]; ignored with "
        "specify_target_noise"
    )
    assert descriptions["periodic_vars"] == (
        "Array with indices of periodic variables, like periodic_vars = [1, 2]"
    )
    assert descriptions["gp_samples"] == (
        "Hyperparameters samples (0 = optimize)"
    )
    assert descriptions["stobads_frame_size_scaling_power"].startswith(
        "Power value of the Sto-BADS incumbent decision rule:  \\gamma"
    )
