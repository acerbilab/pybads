"""The options that `BADS` is given: a value of `None` stands for the
default, the boolean options take only booleans, the checks that MATLAB
BADS's `setupoptions.m` makes, those of the run's limits and counts, and
the options that are not supported; and the option files, whose comment
lines describe the options."""

import logging

import numpy as np
import pytest

from pybads import BADS
from pybads.bads.option_configs import get_pybads_option_dir_path
from pybads.bads.options import Options

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


LIMITS = ["max_fun_evals", "max_iter", "tol_stall_iters"]


@pytest.mark.parametrize(
    "value",
    [
        0,
        -5,
        30.5,
        np.nan,
        -np.inf,
        True,
        "200*D",
        np.array(30),
        np.array([30]),
        30 + 0j,
    ],
    ids=repr,
)
@pytest.mark.parametrize("name", LIMITS)
def test_limits_must_be_positive_integers_or_inf(name, value):
    """The run's limits take a positive integer or inf, and anything else is
    refused when `BADS` is created. MATLAB BADS checks `max_fun_evals` so;
    a `max_iter` or a `tol_stall_iters` given as a string stopped the run
    with `TypeError` at the end of its first iteration, a `tol_stall_iters`
    of 0 or not whole with `TypeError` or `IndexError`, and a `max_iter` not
    whole ended the run at the next whole number of iterations."""
    with pytest.raises(
        ValueError,
        match=rf"options\['{name}'\] needs to be a positive integer or inf, "
        r"not ",
    ):
        _make_bads(**{name: value})


@pytest.mark.parametrize(
    "value, stored",
    [
        (3, 3),
        (3.0, 3),
        (np.int64(3), 3),
        (np.float64(3.0), 3),
        (2**70, np.inf),
        (np.inf, np.inf),
    ],
    ids=repr,
)
@pytest.mark.parametrize("name", ["max_iter", "tol_stall_iters"])
def test_limits_are_stored_as_ints_or_inf(name, value, stored):
    bads = _make_bads(**{name: value})
    assert bads.options[name] == stored
    assert type(bads.options[name]) is (float if stored == np.inf else int)


def test_max_iter_counts_whole_iterations():
    """A whole-number float `max_iter` ends the run at that iteration, as
    the integer does."""
    result = _make_bads(max_iter=3.0).optimize()
    assert result["iterations"] == 3
    assert "options['max_iter']" in result["message"]


def test_infinite_tol_stall_iters_turns_the_stall_criterion_off():
    """At its default, the stall criterion ends this run; `tol_stall_iters
    = inf` leaves it to the other criteria."""
    assert _make_bads().optimize()["status"] == 2
    result = _make_bads(tol_stall_iters=np.inf).optimize()
    assert result["status"] != 2
    assert "tol_fun" not in result["message"]


@pytest.mark.parametrize(
    "search_n_try",
    [-1, 2.5, np.nan, np.inf, True, "D", np.array([3]), 3 + 0j],
    ids=repr,
)
def test_search_n_try_must_be_an_integer_at_least_zero(search_n_try):
    """A `search_n_try` that is not an integer at least 0 is refused when
    `BADS` is created. One that is not whole ended no round of searches, and
    the run turned without evaluating, forever."""
    with pytest.raises(
        ValueError,
        match=r"options\['search_n_try'\] needs to be an integer at least 0",
    ):
        _make_bads(search_n_try=search_n_try)


def test_search_n_try_whole_number_is_an_int():
    bads = _make_bads(search_n_try=4.0)
    assert bads.options["search_n_try"] == 4
    assert type(bads.options["search_n_try"]) is int
    assert type(bads.optim_state["search_count"]) is int


def test_search_n_try_zero_runs_no_search(monkeypatch):
    """`search_n_try = 0` is a run without searches, whose every iteration
    is a poll, as in MATLAB BADS."""

    def no_search(self, gp):
        raise AssertionError("searched with search_n_try = 0")

    monkeypatch.setattr(BADS, "_search_step_", no_search)
    result = _make_bads(search_n_try=0, max_fun_evals=60).optimize()
    assert np.isfinite(result["fval"])
    assert result["iterations"] > 1


@pytest.mark.parametrize(
    "noise_final_samples",
    [-1, 2.5, np.nan, np.inf, True, "10", np.array([10]), 10 + 0j],
    ids=repr,
)
def test_noise_final_samples_must_be_an_integer_at_least_zero(
    noise_final_samples,
):
    """A `noise_final_samples` that is not an integer at least 0 is refused
    when `BADS` is created: one that is not whole stopped a noisy run with
    `TypeError` after its last iteration, and a string at its start."""
    with pytest.raises(
        ValueError,
        match=r"options\['noise_final_samples'\] needs to be an integer at "
        r"least 0",
    ):
        _make_bads(noise_final_samples=noise_final_samples)


def test_noise_final_samples_whole_number_is_an_int():
    """A whole-number float is taken as its integer, which sets the number
    of final samples of a noisy run."""
    noise = np.random.default_rng(0)
    bads = BADS(
        lambda x: _sphere(x) + noise.standard_normal(),
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={
            "display": "off",
            "random_seed": 3,
            "uncertainty_handling": True,
            "max_fun_evals": 60,
            "noise_final_samples": 4.0,
        },
    )
    assert bads.options["noise_final_samples"] == 4
    assert type(bads.options["noise_final_samples"]) is int
    assert np.shape(bads.optimize()["yval_vec"]) == (4,)


@pytest.mark.parametrize(
    "max_fun_evals",
    [30, 30.0, np.float64(30.0)],
    ids=["int", "float", "float64"],
)
def test_max_fun_evals_whole_number_is_an_integer(max_fun_evals):
    bads = _make_bads(max_fun_evals=max_fun_evals)
    assert bads.options["max_fun_evals"] == 30
    assert type(bads.options["max_fun_evals"]) is int


@pytest.mark.parametrize(
    "max_fun_evals, stored",
    [(2**63 - 1, 2**63 - 1), (2**70, np.inf), (1e308, np.inf)],
    ids=["int64 max", "2**70", "1e308"],
)
def test_max_fun_evals_beyond_64_bits_stands_for_inf(max_fun_evals, stored):
    """A whole number too large for NumPy's 64-bit integers, which the run's
    NumPy arithmetic does not take, is no budget: it stands for inf. A
    Python integer that large raised `TypeError` from `np.isfinite` in the
    check, and 1e308 was converted to such an integer, which stopped the run
    with `OverflowError`."""
    bads = _make_bads(max_fun_evals=max_fun_evals, max_iter=2)
    assert bads.options["max_fun_evals"] == stored
    result = bads.optimize()
    assert np.isfinite(result["fval"])


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
    BADS's plots, which have no effect."""
    assert _make_bads(plot="scatter").options["plot"] == "scatter"


def test_fun_values_is_not_supported():
    """MATLAB BADS's option of prior evaluations (`setupvars.m`) is refused,
    with a pointer to the argument that takes them; the empty default
    passes."""
    fun_values = {"X": np.ones((2, D)), "Y": np.array([[3.0], [3.0]])}
    with pytest.raises(
        ValueError, match="fun_values'] is not supported.*precomputed_eval"
    ):
        _make_bads(fun_values=fun_values)
    assert _make_bads(fun_values={}).options["fun_values"] == {}


def test_f_vals_is_not_supported():
    """`f_vals`, PyBADS's own, filled a cache that nothing reads, and a
    finite value in it stopped 1.1.0's run, at its first display line or,
    with several values, when `BADS` was created; it is refused. A value
    without a finite element stands for the default `None`, as an empty or
    one-element one did in 1.1.0."""
    for f_vals in ([48.0], [np.nan, 48.0], ["a"]):
        with pytest.raises(ValueError, match="f_vals'] is not supported"):
            _make_bads(f_vals=f_vals)
    for f_vals in ([], [np.nan], [np.inf], np.full(2, np.nan)):
        _make_bads(f_vals=f_vals)


def test_descriptions_are_whole_comment_lines():
    """An option's description is the whole comment line above it, an `=`
    or a `:` included."""
    descriptions = _make_bads().options.descriptions
    assert descriptions["noise_size"] == (
        "Base observation noise magnitude (SD), e.g. noise_size = 1.0, or a "
        "pair [SD, SD of the prior over log SD]; None is 1.0 in a noisy run "
        "and sqrt(tol_fun) in a deterministic one; ignored with "
        "specify_target_noise"
    )
    assert descriptions["periodic_vars"] == (
        "Indices (from 0, counting the fixed variables) of the periodic "
        "variables, such as angles, e.g. "
        "periodic_vars = [2, 3]; each wraps around its hard bounds, which "
        "need to be finite and are its period"
    )
    assert descriptions["gp_samples"] == (
        "Hyperparameter samples (unused: PyBADS optimizes one set of GP "
        "hyperparameters, where MATLAB BADS samples them above 1)"
    )
    assert descriptions["stobads_frame_size_scaling_power"] == (
        "Deprecated along with stobads: power of the mesh size in the "
        "Sto-BADS interval gamma * SD * mesh_size**power: at 2, a difference "
        "of a fraction of its SD counts as certain at a small mesh; 0 makes "
        "the rule a z-test"
    )


def test_user_options_keep_their_descriptions():
    """An option that the user sets has the description of its file, the
    basic one's and the advanced one's alike, which `str(options)` prints
    beside its value."""
    options = _make_bads(n_search=2**10, noise_size=2.0).options
    defaults = _make_bads().options
    for name in ("n_search", "noise_size"):
        assert options.descriptions[name] == defaults.descriptions[name]
        assert options.descriptions[name] != ""
    lines = str(options).splitlines()
    assert f"n_search: 1024 ({defaults.descriptions['n_search']}) " in lines
    # useroptions, the set of the user's names, is not an option
    assert [line for line in lines if "(None)" in line] == [
        f"useroptions: {options['useroptions']} (None) "
    ]


def _default_options(D):
    """The default options of both option files, for `D` variables."""
    option_dir = get_pybads_option_dir_path()
    options = Options(
        option_dir + "/basic_bads_options.ini", evaluation_parameters={"D": D}
    )
    options.load_options_file(
        option_dir + "/advanced_bads_options.ini",
        evaluation_parameters={"D": D},
    )
    return options


def test_every_option_has_a_description():
    """The documentation shows the option files verbatim: every option has a
    description, the comment line above it, and none ends in the closing
    quote of MATLAB BADS's defaults."""
    options = _default_options(2)
    names = [name for name in options if name != "useroptions"]
    assert len(names) > 100
    for name in names:
        description = options.descriptions[name]
        assert description != "", name
        assert not description.endswith(("'", ";")), name


@pytest.mark.parametrize("D", [1, 2, 6, 20])
def test_search_n_try_is_an_int(D):
    search_n_try = _default_options(D)["search_n_try"]
    assert type(search_n_try) is int
    assert search_n_try == max(D, 3 + D // 2)
