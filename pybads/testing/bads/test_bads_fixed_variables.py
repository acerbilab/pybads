"""Fixed variables, whose four bounds are equal: `BADS` optimizes the other
variables, as a run of the problem reduced to them does, seed for seed, and
gives the target, `non_box_cons`, `output_fcn`, the function log, the
iteration history and the result points of all the variables, the fixed ones
at their values.

On macOS arm64, NumPy's and SciPy's linear algebra (Accelerate) gives results
whose last bits depend on where the arrays lie in memory, so two runs of one
seed need not match once the first Gaussian process is computed: there, a run
and its reduced problem are compared on what the seed alone decides, the
start and the initial design."""

import logging
import platform
import sys

import numpy as np
import pytest

from pybads import BADS

_REPEATS_BIT_FOR_BIT = not (
    sys.platform == "darwin" and platform.machine() == "arm64"
)

# Five variables, the first and the fourth fixed; the last takes the log
# transform, and the third is bounded above only by its hard bound
D_ORIG = 5
FIXED = [0, 3]
FREE = [1, 2, 4]
LB = np.array([2.0, -10.0, -10.0, -1.5, 0.01])
UB = np.array([2.0, 10.0, 10.0, -1.5, 100.0])
PLB = np.array([2.0, -5.0, -5.0, -1.5, 0.1])
PUB = np.array([2.0, 5.0, 5.0, -1.5, 10.0])
X0 = np.array([2.0, 3.0, -2.0, -1.5, 5.0])
CENTER = np.array([0.0, 0.5, -1.0, 0.0, 1.0])


@pytest.fixture(autouse=True)
def _restore_global_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _expand(x_free):
    """Points of all the variables from points of the free ones."""
    x_free = np.asarray(x_free, dtype=float)
    x = np.empty(x_free.shape[:-1] + (D_ORIG,))
    x[...] = LB
    x[..., FREE] = x_free
    return x


def _target(noise=None, noise_seed=0):
    """The target, which depends on the fixed variables too, and the list
    of the points it receives, copied. With `noise`, "inferred" or
    "specified", Gaussian noise from its own generator."""
    calls = []
    rng = np.random.default_rng(noise_seed)

    def fun(x):
        calls.append(np.array(x, copy=True))
        y = float(np.sum((x - CENTER) ** 2) + 0.1 * x[0] * x[3] * x[1])
        if noise is None:
            return y
        sd = 0.5
        y += sd * rng.standard_normal()
        return (y, sd) if noise == "specified" else y

    return fun, calls


def _non_box_cons(X):
    return np.atleast_2d(X)[:, 1] + np.atleast_2d(X)[:, 3] > 4.0


def _runs(x0=X0, noise=None, precomputed=None, **options):
    """The run with fixed variables and the run of the problem reduced to
    the free ones, from the same seed, each with its target's calls and its
    output function's points."""
    non_box_cons = options.pop("non_box_cons", None)
    options = {
        "display": "off",
        "random_seed": 7,
        "max_fun_evals": 40,
        **options,
    }
    if noise is not None:
        options["uncertainty_handling"] = True
        options["specify_target_noise"] = noise == "specified"
    runs = []
    for reduced in (False, True):
        fun, calls = _target(noise)
        outputs = []

        def output_fcn(x, optim_state, state, outputs=outputs):
            outputs.append(np.array(x, copy=True))
            return False

        run_options = {**options, "output_fcn": output_fcn}
        if reduced:
            run_fun = lambda x, fun=fun: fun(_expand(x))  # noqa: E731
            run_nbc = (
                None
                if non_box_cons is None
                else lambda X, nbc=non_box_cons: nbc(_expand(X))
            )
            run_options["output_fcn"] = lambda x, s, t, f=output_fcn: f(
                _expand(x), s, t
            )
            periodic = run_options.get("periodic_vars")
            if periodic is not None:
                run_options["periodic_vars"] = [
                    FREE.index(i) for i in periodic if i in FREE
                ]
            args = [
                None if x0 is None else np.asarray(x0)[FREE],
                LB[FREE],
                UB[FREE],
                PLB[FREE],
                PUB[FREE],
            ]
            run_precomputed = (
                None
                if precomputed is None
                else (precomputed[0][:, FREE],) + tuple(precomputed[1:])
            )
        else:
            run_fun, run_nbc = fun, non_box_cons
            args = [x0, LB, UB, PLB, PUB]
            run_precomputed = precomputed
        bads = BADS(
            run_fun,
            *args,
            non_box_cons=run_nbc,
            options=run_options,
            precomputed_evaluations=run_precomputed,
        )
        result = bads.optimize()
        runs.append((bads, result, calls, outputs))
    return runs


def _assert_same_run(runs):
    (bads, result, calls, outputs), (
        bads_r,
        result_r,
        calls_r,
        outputs_r,
    ) = runs
    log, log_r = bads.function_logger, bads_r.function_logger
    assert bads.D == bads_r.D == len(FREE)
    assert np.array_equal(result["x0"], _expand(result_r["x0"]))
    if not _REPEATS_BIT_FOR_BIT:
        # The rows of the evaluations made before the run (distinct points
        # without uncertainty handling), of the start and of the design
        n = bads.optim_state["eff_starting_points"]
        assert n == bads_r.optim_state["eff_starting_points"]
        n += bads.optim_state["precomputed_locations"]
        assert np.array_equal(log.X[:n], log_r.X[:n])
        assert np.array_equal(log.X_orig[:n], _expand(log_r.X_orig[:n]))
        assert np.array_equal(log.Y[:n], log_r.Y[:n])
        return
    # The target received the same points, of all the variables
    assert len(calls) == len(calls_r) == result["func_count"]
    assert np.array_equal(np.array(calls), np.array(calls_r))
    for key in (
        "fval",
        "fsd",
        "func_count",
        "iterations",
        "status",
        "message",
        "mesh_size",
    ):
        assert result[key] == result_r[key], key
    for key in ("yval_vec", "ysd_vec"):
        assert (result[key] is None and result_r[key] is None) or (
            np.array_equal(result[key], result_r[key])
        ), key
    assert np.array_equal(result["x"], _expand(result_r["x"]))
    # The log holds the points of all the variables in the original space,
    # and those of the run in the transformed one
    rows, rows_r = log.X_flag, log_r.X_flag
    assert np.array_equal(rows, rows_r)
    assert np.array_equal(log.X_orig[rows], _expand(log_r.X_orig[rows_r]))
    assert np.array_equal(log.X[rows], log_r.X[rows_r])
    assert np.array_equal(log.Y[rows], log_r.Y[rows_r])
    # So do the iteration history and the output function
    history, history_r = bads.iteration_history, bads_r.iteration_history
    assert np.array_equal(
        np.vstack(history["x"]), _expand(np.vstack(history_r["x"]))
    )
    assert np.array_equal(np.vstack(history["u"]), np.vstack(history_r["u"]))
    assert np.array_equal(history["fval"], history_r["fval"])
    assert np.array_equal(np.array(outputs), np.array(outputs_r))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"x0": None},
        {"x0": np.where(np.isin(np.arange(D_ORIG), FIXED), np.nan, X0)},
        {"noise": "inferred"},
        {"noise": "specified"},
        {"non_box_cons": _non_box_cons},
        {"periodic_vars": [0, 2]},
    ],
    ids=[
        "deterministic",
        "random_x0",
        "x0_nan_at_fixed",
        "inferred_noise",
        "specified_noise",
        "non_box_cons",
        "periodic",
    ],
)
def test_run_is_that_of_the_reduced_problem(kwargs):
    """A run with fixed variables is the run of the problem reduced to the
    other variables, seed for seed: the options are evaluated with D the
    number of free variables, and every draw and computation is the same,
    while the points that leave the run have all the variables. A random
    or non-finite `x0` is drawn over the free variables alone; a
    non-finite `x0` at a fixed variable is its value; a fixed periodic
    variable is left out of `periodic_vars`, whose other indices count all
    the variables."""
    _assert_same_run(_runs(**kwargs))


def test_run_with_evaluations_made_before_is_that_of_the_reduced_problem():
    """`precomputed_evaluations` holds points of all the variables, which
    the run takes as the reduced problem takes their free coordinates, a
    point given twice included."""
    rng = np.random.default_rng(3)
    X = _expand(rng.uniform(-3, 3, (8, len(FREE))) + [0, 0, 4])
    # Two points given twice, which the log holds once
    X = np.vstack([X, X[[5, 2]]])
    y = np.array([_target()[0](x) for x in X])
    _assert_same_run(_runs(precomputed=(X, y)))


def test_callbacks_and_result_see_all_the_variables():
    """The target gets 1-D points of all the variables, `non_box_cons`
    arrays of them, and the output function and the result points of them,
    the fixed ones at their values; the result's `fun` and `non_box_cons`
    are those given, and the options' defaults take D as the number of free
    variables."""
    fun, calls = _target()
    constraint_inputs = []

    def non_box_cons(X):
        constraint_inputs.append(np.array(X, copy=True))
        return _non_box_cons(X)

    bads = BADS(
        fun,
        X0,
        LB,
        UB,
        PLB,
        PUB,
        non_box_cons=non_box_cons,
        options={"display": "off", "random_seed": 0, "max_fun_evals": 20},
    )
    assert bads.D == len(FREE)
    assert bads.options["max_fun_evals"] == 20
    defaults = BADS(fun, X0, LB, UB, PLB, PUB, options={"display": "off"})
    assert defaults.options["max_iter"] == 200 * len(FREE)
    assert defaults.options["max_fun_evals"] == 500 * len(FREE)
    result = bads.optimize()
    assert all(x.shape == (D_ORIG,) for x in calls)
    assert all(np.all(x[FIXED] == LB[FIXED]) for x in calls)
    assert all(X.ndim == 2 and X.shape[1] == D_ORIG for X in constraint_inputs)
    assert all(np.all(X[:, FIXED] == LB[FIXED]) for X in constraint_inputs)
    assert result["x"].shape == (D_ORIG,)
    assert np.all(result["x"][FIXED] == LB[FIXED])
    assert np.array_equal(result["x0"], X0[None])
    assert result["fun"] is fun
    assert result["non_box_cons"] is non_box_cons


def test_setup_reports_the_indices_of_all_the_variables(caplog):
    """The reports of the setup name the variables by their indices among
    all the variables: the fixed ones, the one with an infinite bound, the
    one on a log scale and the periodic one."""
    ub = UB.copy()
    ub[2] = np.inf
    with caplog.at_level(logging.DEBUG, logger="BADS"):
        BADS(
            _target()[0],
            X0,
            LB,
            ub,
            PLB,
            PUB,
            options={"display": "notify", "periodic_vars": [0, 1]},
        )
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "left out of the optimization: [0, 3]" in messages
    assert "infinite bound(s), in variables (index) [2]" in messages
    assert "log coordinates: [4]" in messages
    assert "periodic boundaries: [1]" in messages


def test_periodic_vars_count_all_the_variables():
    """`periodic_vars` counts all the variables, is checked against them all
    and kept as given, sorted; the run's mask of periodic variables and its
    transform cover the free ones, and leave a fixed one out."""
    fun = _target()[0]

    def make(periodic_vars):
        return BADS(
            fun,
            X0,
            LB,
            UB,
            PLB,
            PUB,
            options={"display": "off", "periodic_vars": periodic_vars},
        )

    # The last variable is on a log scale, which a periodic one is not
    bads = make([4, 0, 1])
    assert bads.options["periodic_vars"] == [0, 1, 4]
    assert bads.optim_state["periodic_vars"].tolist() == [[True, False, True]]
    assert not bads.var_transf.apply_log_t[0, 2]
    only_fixed = make([0, 3])
    assert only_fixed.options["periodic_vars"] == [0, 3]
    assert not np.any(only_fixed.optim_state["periodic_vars"])
    assert only_fixed.var_transf.apply_log_t[0, 2]
    with pytest.raises(ValueError, match="outside 0 to D - 1 = 4"):
        make([5])


def test_periodic_fixed_variables_need_no_gpyreg_with_periods(monkeypatch):
    """A `periodic_vars` that names only fixed variables makes no variable
    of the run periodic, and needs no gpyreg whose kernels take periods."""
    import pybads.bads.bads as bads_module

    monkeypatch.setattr(bads_module, "_gpyreg_takes_periods", lambda: False)
    bads = BADS(
        _target()[0],
        X0,
        LB,
        UB,
        PLB,
        PUB,
        options={"display": "off", "periodic_vars": [0]},
    )
    assert not np.any(bads.optim_state["periodic_vars"])


def test_x0_off_a_fixed_value_is_refused():
    """A finite `x0` that differs from the value of a fixed variable is
    refused, with the indices of the variables where it does."""
    x0 = X0.copy()
    x0[3] = 0.0
    with pytest.raises(ValueError, match=r"bads:FixedVariables.*\[3\]"):
        BADS(_target()[0], x0, LB, UB, PLB, PUB)


def test_every_variable_fixed_is_refused():
    with pytest.raises(ValueError, match="fixes them all"):
        BADS(_target()[0], LB, LB, LB, LB, LB)


def test_equal_plausible_bounds_without_equal_hard_bounds_are_refused():
    """A variable is fixed only when its four bounds are equal; equal
    plausible bounds between distinct hard bounds are refused."""
    plb, pub = PLB.copy(), PUB.copy()
    plb[1] = pub[1] = 1.0
    with pytest.raises(ValueError, match="bads:MatchingPB"):
        BADS(_target()[0], X0, LB, UB, plb, pub)


@pytest.mark.parametrize(
    "precomputed, match",
    [
        (
            (_expand(np.zeros((2, 3))) + [0, 0, 0, 1, 1], [1.0, 2.0]),
            r"within the hard bounds; rows \[0, 1\]",
        ),
        ((np.ones((2, 3)), [1.0, 2.0]), r"shape \(N, 5\)"),
    ],
    ids=["off_fixed_value", "free_variables_only"],
)
def test_evaluations_made_before_have_all_the_variables(precomputed, match):
    """The points of `precomputed_evaluations` have all the variables, and
    one whose fixed coordinate differs from its value lies outside the hard
    bounds."""
    with pytest.raises(ValueError, match=match):
        BADS(
            _target()[0],
            X0,
            LB,
            UB,
            PLB,
            PUB,
            precomputed_evaluations=precomputed,
        )


def test_one_free_variable_among_lists_of_integers():
    """A run with one free variable, from lists of integers and without
    plausible bounds, which are then the hard bounds, is the run of the
    problem reduced to that variable."""
    options = {"display": "off", "random_seed": 3, "max_fun_evals": 30}
    calls = []

    def fun(x):
        calls.append(np.array(x, copy=True))
        return float((x[1] - 0.5) ** 2 + 0.1 * x[0] * x[1])

    result = BADS(fun, [2, 3], [2, -10], [2, 10], options=dict(options))
    result = result.optimize()
    calls_full, calls[:] = calls[:], []
    result_r = BADS(
        lambda x: fun(np.r_[2.0, x]), [3], [-10], [10], options=dict(options)
    ).optimize()
    assert np.array_equal(result["x0"], [[2.0, 3.0]])
    assert result["x"][0] == 2.0
    if _REPEATS_BIT_FOR_BIT:
        assert np.array_equal(np.array(calls_full), np.array(calls))
        assert result["x"][1] == result_r["x"][0]
        assert result["fval"] == result_r["fval"]
    else:
        assert np.array_equal(calls_full[0], calls[0])


def test_unbounded_free_variables_are_reported_as_unconstrained(caplog):
    """Beside fixed variables, free variables without bounds make a fully
    unconstrained optimization, as the reduced problem is."""
    lb, ub = np.full(D_ORIG, -np.inf), np.full(D_ORIG, np.inf)
    lb[FIXED], ub[FIXED] = LB[FIXED], UB[FIXED]
    plb, pub = PLB.copy(), PUB.copy()
    plb[4], pub[4] = -5.0, 5.0
    with caplog.at_level(logging.DEBUG, logger="BADS"):
        BADS(_target()[0], X0, lb, ub, plb, pub, options={"display": "notify"})
    messages = [record.getMessage() for record in caplog.records]
    assert "Detected fully unconstrained optimization." in messages


def test_x0_of_several_rows_beside_fixed_variables_is_refused():
    with pytest.raises(ValueError, match="bads:StartingSet"):
        BADS(_target()[0], np.vstack([X0, X0]), LB, UB, PLB, PUB)


@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
def test_complex_inputs_of_real_values_fix_variables():
    """Inputs of a complex type whose values are real are taken as real
    numbers, as without fixed variables."""
    bads = BADS(
        _target()[0],
        X0.astype(complex),
        LB.astype(complex),
        UB,
        PLB,
        PUB,
        options={"display": "off"},
    )
    assert bads.D == len(FREE)
    assert np.array_equal(bads.x0, X0[None])
