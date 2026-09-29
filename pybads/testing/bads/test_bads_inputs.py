"""The inputs of `BADS` as MATLAB BADS checks them (`boundscheck.m`,
`setupvars.m`): the bounds, the starting point and `non_box_cons`."""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from pybads import BADS

OPTIONS = {"display": "off", "random_seed": 1}


def _quadratic(x):
    x = np.asarray(x).ravel()
    return float((x[0] - 0.3) ** 2 + 0.1 * (x[1] - 2.0) ** 2)


def test_mixed_bounded_and_unbounded_variables():
    """A bounded variable beside an unbounded one is accepted, as in MATLAB
    BADS, and the run finds the minimum."""
    bads = BADS(
        _quadratic,
        np.array([0.5, 0.0]),
        np.array([0.0, -np.inf]),
        np.array([1.0, np.inf]),
        np.array([0.1, -3.0]),
        np.array([0.9, 3.0]),
        options={**OPTIONS, "max_fun_evals": 100},
    )
    assert np.isinf(bads.optim_state["lb"]).tolist() == [[False, True]]
    result = bads.optimize()
    assert result["func_count"] <= 100
    np.testing.assert_allclose(result["x"], [0.3, 2.0], atol=0.05)


@pytest.mark.parametrize(
    "lb, ub, plb, pub, log",
    [
        ([0.0, -np.inf], [np.inf, np.inf], [0.1, -3.0], [0.9, 3.0], False),
        ([-np.inf, -np.inf], [1.0, 10.0], [0.1, -3.0], [0.9, 3.0], False),
        ([1e-3, 0.5], [np.inf, np.inf], [1e-2, 1.0], [1.0, 20.0], True),
    ],
    ids=["bounded_below", "bounded_above", "bounded_below_log"],
)
def test_half_bounded_variables(lb, ub, plb, pub, log):
    """A variable bounded on one side only is accepted, as in MATLAB BADS
    (`setupvars.m` only cautions), log-transformed when its bounds are all
    positive and `pub / plb >= 10` (which a variable bounded above only
    cannot be), and the run finds the minimum."""
    bads = BADS(
        _quadratic,
        np.array([0.5, 1.0]),
        *[np.array(bound) for bound in (lb, ub, plb, pub)],
        options={**OPTIONS, "max_fun_evals": 100},
    )
    state = bads.optim_state
    assert np.isinf(state["lb"]).tolist() == [np.isinf(lb).tolist()]
    assert np.isinf(state["ub"]).tolist() == [np.isinf(ub).tolist()]
    assert bads.var_transf.apply_log_t.tolist() == [[log, log]]
    result = bads.optimize()
    assert result["func_count"] <= 100
    np.testing.assert_allclose(result["x"], [0.3, 2.0], atol=0.05)


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def _bounds_of(bads):
    keys = ["lb", "ub", "plb", "pub", "lb_orig", "ub_orig"]
    keys += ["plb_orig", "pub_orig", "u"]
    return [bads.x0] + [bads.optim_state[key] for key in keys]


@pytest.mark.parametrize(
    "plausible", [True, False], ids=["with_plausible", "without_plausible"]
)
def test_scalar_bounds_are_replicated(plausible):
    """Scalar bounds stand for the same bound in each dimension, as in
    MATLAB BADS (`boundscheck.m`)."""
    x0, D = np.array([0.5, -0.2, 0.1]), 3
    scalars = (-5.0, 5.0, -2.0, 2.0) if plausible else (-5.0, 5.0)
    vectors = [np.full(D, bound) for bound in scalars]
    by_scalars = BADS(_sphere, x0, *scalars, options=OPTIONS)
    by_vectors = BADS(_sphere, x0, *vectors, options=OPTIONS)
    assert by_scalars.optim_state["lb"].shape == (1, D)
    for scalar, vector in zip(_bounds_of(by_scalars), _bounds_of(by_vectors)):
        np.testing.assert_array_equal(scalar, vector)


@pytest.mark.parametrize(
    "x0, bounds, log",
    [
        ([5, 3], ([1, -10], [1000, 10], [2, -5], [500, 5]), [True, False]),
        ([5, 3], ([1, -10], [1000, 10]), [True, False]),
        ([5, 30], (1, 1000, 2, 500), [True, True]),
        ([1, -2], (-10, 10, -5, 5), [False, False]),
    ],
    ids=[
        "arrays",
        "arrays_plausible_omitted",
        "scalars_log",
        "scalars_linear",
    ],
)
def test_integer_bounds_are_taken_as_floats(x0, bounds, log):
    """Integer-typed bounds and `x0`, arrays or scalars, are taken as the
    same values as floats, as MATLAB's numbers are doubles: the log of the
    bounds of a log-scaled variable is not truncated."""

    def inputs(dtype):
        return [
            np.array(b, dtype) if isinstance(b, list) else dtype(b)
            for b in (x0, *bounds)
        ]

    by_ints = BADS(_sphere, *inputs(int), options=OPTIONS)
    by_floats = BADS(_sphere, *inputs(float), options=OPTIONS)
    for bads in (by_ints, by_floats):
        assert bads.var_transf.apply_log_t.tolist() == [log]
    ints, floats = _bounds_of(by_ints), _bounds_of(by_floats)
    for by_int, by_float in zip(ints, floats):
        np.testing.assert_array_equal(by_int, by_float)
    assert [by_int.dtype for by_int in ints] == [float] * len(ints)


@pytest.mark.parametrize(
    "bounds, log, u_bounds, u0",
    [
        ((0.0, -5.0, 5.0), False, (-1.0, 1.0), 0.0),
        ((-5.0, -5.0, 5.0), False, (-1.0, 1.0), -1.0),
        ((-5.0, -5.0, 5.0, -2.0, 2.0), False, (-2.5, 2.5), -2.5),
        ((9.995, 0.0, 10.0, 2.0, 8.0), False, (-5 / 3, 5 / 3), 1705 / 1024),
        ((2.0, 1.0, 10.0), True, (-1.0, 1.0), -407 / 1024),
        ((1e-2, 1e-3, 1e3, 1e-2, 1e2), True, (-1.5, 1.5), -1.0),
        ((2e-4, 0.0, 1.0, 1e-4, 5e-4), False, (-1.5, 4998.5), -0.5),
        (
            (0.5, -1000.0, 1.0, 0.0, 0.99),
            False,
            (-1000.495 / 0.495, 0.505 / 0.495),
            10 / 1024,
        ),
    ],
    ids=[
        "plausible_omitted",
        "start_on_bound_plausible_omitted",
        "start_on_bound",
        "start_near_bound_outside_plausible",
        "log_plausible_omitted",
        "log_start_on_plausible_bound",
        "plausible_near_lower_bound",
        "plausible_near_upper_bound",
    ],
)
def test_start_and_plausible_bounds_are_kept(bounds, log, u_bounds, u0):
    """As in MATLAB BADS (`boundscheck.m`, `setupvars.m`), neither `x0` nor
    the plausible bounds are moved: omitted plausible bounds are the hard
    bounds, a start on a hard bound or outside the plausible box stays where
    it is, and plausible bounds close to a hard bound are accepted. The
    transformed hard bounds and the start on the grid are MATLAB's."""
    x0, lb, ub, plb, pub = (bounds + (None, None))[:5]
    bads = BADS(
        _sphere,
        *[
            None if b is None else np.array([b])
            for b in (x0, lb, ub, plb, pub)
        ],
        options=OPTIONS,
    )
    state = bads.optim_state
    np.testing.assert_array_equal(bads.x0, [[x0]])
    plb, pub = (lb if plb is None else plb), (ub if pub is None else pub)
    np.testing.assert_array_equal(state["plb_orig"], [[plb]])
    np.testing.assert_array_equal(state["pub_orig"], [[pub]])
    assert bads.var_transf.apply_log_t.tolist() == [[log]]
    np.testing.assert_allclose(
        np.hstack([state["lb"], state["ub"]]), [u_bounds], rtol=1e-12
    )
    np.testing.assert_allclose(state["u"], [[u0]], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "x0", [[np.inf, 0.0], [0.0, -np.inf]], ids=["plus_inf", "minus_inf"]
)
def test_infinite_x0_is_drawn_at_random(x0):
    """A start with an infinite element is replaced by a random point in the
    plausible box, the one drawn for a missing `x0`, also within finite hard
    bounds, as in MATLAB BADS (`setupvars.m`)."""
    bounds = (-2 * np.ones(2), 2 * np.ones(2), -np.ones(2), np.ones(2))
    bads = BADS(_sphere, np.array(x0), *bounds, options=OPTIONS)
    missing = BADS(_sphere, None, *bounds, options=OPTIONS)
    assert np.all(np.isfinite(bads.x0)) and np.all(np.abs(bads.x0) <= 1)
    np.testing.assert_array_equal(bads.x0, missing.x0)


@pytest.mark.parametrize(
    "plb, pub",
    [([-1.0, -1.0], [1.0, 1.0]), (-1.0, 1.0), (-1, 1)],
    ids=["lists", "float_scalars", "int_scalars"],
)
def test_missing_x0_takes_its_size_from_any_plausible_bounds(plb, pub):
    """Without `x0`, plausible bounds given as a list or a Python scalar size
    the random start as the same bounds given as NumPy arrays or scalars do,
    and give the same start."""
    as_numpy = (np.asarray(plb, dtype=float), np.asarray(pub, dtype=float))
    bounds = (-2 * np.ones(np.size(plb)), 2 * np.ones(np.size(plb)))
    bads = BADS(_sphere, None, *bounds, plb, pub, options=OPTIONS)
    reference = BADS(_sphere, None, *bounds, *as_numpy, options=OPTIONS)
    assert bads.D == np.size(plb)
    np.testing.assert_array_equal(bads.x0, reference.x0)


@pytest.mark.parametrize(
    "bounds",
    [(-np.ones(2), np.ones(2)), ()],
    ids=["with_bounds", "without_bounds"],
)
def test_starting_set_is_refused(bounds):
    """`x0` is a single point, as in MATLAB BADS: a set of starting points
    is refused, and the plausible bounds are not estimated from it."""
    x0 = np.array([[0.1, 0.2], [0.3, -0.4]])
    with pytest.raises(ValueError, match="bads:StartingSet"):
        BADS(_sphere, x0, *bounds, options=OPTIONS)


def _shifted_sphere(x):
    return float(np.sum((np.atleast_2d(x) - 1.0) ** 2))


def _constrained_bads(non_box_cons, **options):
    return BADS(
        _shifted_sphere,
        np.array([0.3, 0.2]),
        -2 * np.ones(2),
        2 * np.ones(2),
        -np.ones(2),
        np.ones(2),
        non_box_cons=non_box_cons,
        options={**OPTIONS, **options},
    )


@pytest.mark.parametrize(
    "non_box_cons",
    [
        lambda x: float(np.sum(x**2) > 1),
        lambda x: bool(np.sum(x**2) > 1),
        lambda x: np.zeros((len(x), 2)),
        lambda x: float(x[0] ** 2 + x[1] ** 2 > 1),
    ],
    ids=["scalar", "bool", "two_columns", "raises"],
)
def test_non_box_cons_output_is_checked(non_box_cons):
    """`non_box_cons` takes an N x D array and returns N violations, or
    `BADS` raises a `ValueError` that says so, as in MATLAB BADS
    (`setupvars.m`), also when the constraint raises on such an array."""
    with pytest.raises(ValueError, match="one point per row"):
        _constrained_bads(non_box_cons)


def test_non_box_cons_output_of_shape_n_or_n_by_1():
    """The N violations may come as an (N,) or an (N, 1) array, which give
    the same run."""
    results = []
    for shape in [(-1,), (-1, 1)]:

        def disc(x, shape=shape):
            outside = np.sum(np.atleast_2d(x) ** 2, axis=1) > 1
            return np.reshape(outside, shape)

        results.append(_constrained_bads(disc, max_fun_evals=50).optimize())
    flat, column = results
    assert np.sum(flat["x"] ** 2) <= 1
    np.testing.assert_array_equal(column["x"], flat["x"])
    assert column["func_count"] == flat["func_count"]


def _outside_disc(x):
    return np.sum(np.atleast_2d(x) ** 2, axis=1) > 1


@pytest.mark.parametrize(
    "seed, first_draw_feasible",
    [(1, True), (8, False)],
    ids=["first_draw_feasible", "first_draw_infeasible"],
)
def test_random_x0_is_drawn_again_until_feasible(seed, first_draw_feasible):
    """A missing `x0` that violates `non_box_cons` is drawn again from the
    run's generator, in the transformed plausible box (here the identity
    map of [-1, 1]^2). A feasible first draw is kept, and no draw is added."""
    rng = np.random.default_rng(seed)
    draws = 0
    while True:
        u = rng.uniform(-1.0, 1.0, size=(1, 2))
        draws += 1
        if not _outside_disc(u):
            break
    assert (draws == 1) == first_draw_feasible
    bads = BADS(
        _shifted_sphere,
        None,
        -2 * np.ones(2),
        2 * np.ones(2),
        -np.ones(2),
        np.ones(2),
        non_box_cons=_outside_disc,
        options={**OPTIONS, "random_seed": seed},
    )
    np.testing.assert_allclose(bads.x0, u, rtol=1e-12)
    assert bads.rng.random() == rng.random()


def _negative_on_mesh(x):
    """Violated where the first coordinate is negative and on the initial
    search mesh of the identity map of [-1, 1]^2, whose step is 2**-10: a
    random draw is feasible as drawn, and the point on the mesh may not
    be."""
    x = np.atleast_2d(x)
    on_mesh = np.isclose(
        x[:, 0] * 2**10, np.round(x[:, 0] * 2**10), rtol=0, atol=1e-9
    )
    return (on_mesh & (x[:, 0] < 0)).astype(float)


def test_random_x0_is_drawn_again_until_feasible_on_the_mesh():
    """A missing `x0` is tested against `non_box_cons` once it is put on the
    mesh, as MATLAB BADS tests its random start (`setupvars.m:83-85`,
    `evalinitmesh.m:22-26`), and drawn again while that point violates it:
    here the first draw is feasible as drawn and not on the mesh."""
    seed = 2
    rng = np.random.default_rng(seed)
    draws = 0
    while True:
        u = rng.uniform(-1.0, 1.0, size=(1, 2))
        draws += 1
        if np.round(u[0, 0] * 2**10) / 2**10 >= 0:
            break
    assert draws > 1
    bads = BADS(
        _shifted_sphere,
        None,
        -2 * np.ones(2),
        2 * np.ones(2),
        -np.ones(2),
        np.ones(2),
        non_box_cons=_negative_on_mesh,
        options={**OPTIONS, "random_seed": seed},
    )
    np.testing.assert_allclose(bads.x0, u, rtol=1e-12)
    assert _negative_on_mesh(bads.optim_state["u"])[0] == 0
    assert bads.rng.random() == rng.random()


def test_random_x0_that_stays_infeasible_is_refused():
    """After 1000 draws that all violate `non_box_cons`, `BADS` raises."""
    with pytest.raises(ValueError, match="does not satisfy non-bound"):
        BADS(
            _shifted_sphere,
            None,
            -2 * np.ones(2),
            2 * np.ones(2),
            -np.ones(2),
            np.ones(2),
            non_box_cons=lambda x: np.ones(len(x)),
            options=OPTIONS,
        )


def _bads_with_improvement_quantile(improvement_quantile):
    return BADS(
        _quadratic,
        np.array([0.5, 0.0]),
        -5 * np.ones(2),
        5 * np.ones(2),
        -3 * np.ones(2),
        3 * np.ones(2),
        options={**OPTIONS, "improvement_quantile": improvement_quantile},
    )


@pytest.mark.parametrize(
    "improvement_quantile",
    [0, 1, -0.2, 1.5, np.nan, True, "0.3", 0.3 + 0j, np.array([0.3, 0.4])],
)
def test_improvement_quantile_outside_zero_one_is_refused(
    improvement_quantile,
):
    """An `improvement_quantile` that is not a number greater than 0 and
    less than 1, whose improvements are NaN or infinite, is refused with
    `ValueError` when `BADS` is created; MATLAB BADS refuses it when it
    evaluates an improvement (`EvalImprovement` in `bads.m`)."""
    with pytest.raises(
        ValueError,
        match=r"improvement_quantile'\] needs to be greater than 0 and less",
    ):
        _bads_with_improvement_quantile(improvement_quantile)


@pytest.mark.parametrize("improvement_quantile", [1e-6, 0.25, 0.9])
def test_improvement_quantile_inside_zero_one_is_accepted(
    improvement_quantile,
):
    bads = _bads_with_improvement_quantile(improvement_quantile)
    assert bads.options["improvement_quantile"] == improvement_quantile


def _bads_with_accelerate_mesh_steps(accelerate_mesh_steps):
    return BADS(
        _quadratic,
        np.array([0.5, 0.0]),
        -5 * np.ones(2),
        5 * np.ones(2),
        -3 * np.ones(2),
        3 * np.ones(2),
        options={**OPTIONS, "accelerate_mesh_steps": accelerate_mesh_steps},
    )


@pytest.mark.parametrize(
    "accelerate_mesh_steps", [0, -1, 2.5, np.nan, np.inf, True, "3"]
)
def test_accelerate_mesh_steps_not_a_positive_integer_is_refused(
    accelerate_mesh_steps,
):
    """An `accelerate_mesh_steps` that is not a positive integer is refused
    when `BADS` is created: below 1, the accelerated mesh reduction read an
    iteration not recorded yet and the run stopped with `TypeError` at its
    first failed poll, as MATLAB BADS stops."""
    with pytest.raises(
        ValueError,
        match=r"accelerate_mesh_steps'\] needs to be a positive integer",
    ):
        _bads_with_accelerate_mesh_steps(accelerate_mesh_steps)


def test_accelerate_mesh_steps_refusal_names_accelerate_mesh():
    """`inf`, which MATLAB BADS and 1.1.0 ran without the accelerated
    reduction of the mesh, is refused, and the message names the switch
    that turns the reduction off, `accelerate_mesh=False`."""
    with pytest.raises(
        ValueError, match=r"options\['accelerate_mesh'\] = False turns"
    ):
        _bads_with_accelerate_mesh_steps(np.inf)


def test_acq_hedge_true_is_refused():
    """`acq_hedge=True`, MATLAB BADS's acquisition hedge, which PyBADS does
    not implement and MATLAB BADS labels unsupported, is refused when
    `BADS` is created; a run with it stopped with `UnboundLocalError` at
    its first improving search."""
    with pytest.raises(
        ValueError, match=r"options\['acq_hedge'\] should be False"
    ):
        BADS(
            _quadratic,
            np.array([0.5, 0.0]),
            -5 * np.ones(2),
            5 * np.ones(2),
            -3 * np.ones(2),
            3 * np.ones(2),
            options={**OPTIONS, "acq_hedge": True},
        )


@pytest.mark.parametrize("accelerate_mesh_steps", [1, 3, 3.0, np.int64(2)])
def test_accelerate_mesh_steps_positive_integer_is_accepted(
    accelerate_mesh_steps,
):
    bads = _bads_with_accelerate_mesh_steps(accelerate_mesh_steps)
    assert bads.options["accelerate_mesh_steps"] == accelerate_mesh_steps
    assert type(bads.options["accelerate_mesh_steps"]) is int
    result = bads.optimize()
    assert np.isfinite(result["fval"])


def _bads_with_hedge_gamma(hedge_gamma, n_search_methods=2):
    search_method = [("ES-wcm", 1), ("ES-ell", 1), ("ES-wcm", 1)]
    return BADS(
        _quadratic,
        np.array([0.5, 0.0]),
        -5 * np.ones(2),
        5 * np.ones(2),
        -3 * np.ones(2),
        3 * np.ones(2),
        options={
            **OPTIONS,
            "hedge_gamma": hedge_gamma,
            "search_method": search_method[:n_search_methods],
        },
    )


@pytest.mark.parametrize(
    "hedge_gamma, n_search_methods",
    [
        (-0.1, 2),
        (-1e-12, 2),
        (0.5 + 1e-12, 2),
        (1.25, 2),
        (0.34, 3),
        (1.5, 1),
        (np.nan, 2),
        (np.inf, 2),
        (True, 2),
        ("0.1", 2),
        (np.array([0.1, 0.2]), 2),
    ],
)
def test_hedge_gamma_outside_zero_to_one_over_n_is_refused(
    hedge_gamma, n_search_methods
):
    """A `hedge_gamma` outside [0, 1 / n], n the number of search methods,
    is refused when `BADS` is created: above 1 / n the hedge's
    probabilities favor the search of lower gain, and above 1 / (n - 1)
    some of them are negative; MATLAB BADS runs with them."""
    with pytest.raises(
        ValueError, match=r"hedge_gamma'\] needs to lie between 0 and 1 / n"
    ):
        _bads_with_hedge_gamma(hedge_gamma, n_search_methods)


@pytest.mark.parametrize(
    "hedge_gamma, n_search_methods",
    [(0, 2), (0.125, 2), (0.5, 2), (1 / 3, 3), (1, 1), (np.float64(0.25), 2)],
)
def test_hedge_gamma_from_zero_to_one_over_n_is_accepted(
    hedge_gamma, n_search_methods
):
    bads = _bads_with_hedge_gamma(hedge_gamma, n_search_methods)
    assert bads.options["hedge_gamma"] == hedge_gamma


def _bads_with_options(options, D=2):
    return BADS(
        _sphere,
        np.full(D, 0.5),
        -5 * np.ones(D),
        5 * np.ones(D),
        -3 * np.ones(D),
        3 * np.ones(D),
        options={**OPTIONS, **options},
    )


@pytest.mark.parametrize(
    "hedge_beta",
    [
        -1.0,
        -1e-12,
        -1000.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        "1",
        1 + 0j,
        np.array([1.0, 2.0]),
    ],
)
def test_hedge_beta_not_a_finite_number_at_least_zero_is_refused(hedge_beta):
    """A `hedge_beta` that is not a finite number at least 0 is refused when
    `BADS` is created: below 0 the hedge favors the search of lower gain,
    and at inf or NaN its probabilities and gains are NaN, so that every
    choice is at random; MATLAB BADS runs with them."""
    with pytest.raises(
        ValueError,
        match=r"hedge_beta'\] needs to be a finite number greater than or "
        r"equal to 0",
    ):
        _bads_with_options({"hedge_beta": hedge_beta})


def test_hedge_beta_refusal_names_its_default():
    """The refusal of `hedge_beta` names its default, `1e-3 / tol_fun`."""
    with pytest.raises(
        ValueError,
        match=r"not -1\.0; its default is 1e-3 / options\['tol_fun'\]",
    ):
        _bads_with_options({"hedge_beta": -1.0})


@pytest.mark.parametrize(
    "tol_fun",
    [
        0,
        0.0,
        -1e-3,
        np.float64(-1.0),
        np.nan,
        np.inf,
        -np.inf,
        410.0,
        1e10,
        False,
        True,
        np.True_,
        "1e-3",
        np.array(1e-3),
        np.array([1e-3]),
        1e-3 + 0j,
    ],
)
@pytest.mark.parametrize("hedge_beta", [None, 1.0])
def test_tol_fun_not_a_positive_number_at_most_e6_is_refused(
    tol_fun, hedge_beta
):
    """A `tol_fun` that is not a positive real number at most e^6 (a
    boolean, a string, an array, of one element or none, or a complex number
    included) is refused when `BADS` is created, whatever `hedge_beta`: 0 and
    False stopped with a bare `ZeroDivisionError` at the default `hedge_beta
    = 1e-3 / tol_fun`, which refused a negative value or NaN but not -inf,
    and not beside a user's `hedge_beta`; above e^6, where the bounds of the
    GP's noise cross, inf included, the run stopped at its first fit of the
    GP."""
    with pytest.raises(
        ValueError,
        match=r"tol_fun'\] needs to be a positive number at most e\^6",
    ):
        _bads_with_options({"tol_fun": tol_fun, "hedge_beta": hedge_beta})


@pytest.mark.parametrize("tol_fun", [1e-12, 0.1, 400, np.exp(6)])
def test_tol_fun_up_to_e6_runs(tol_fun):
    """A `tol_fun` up to e^6 runs, the bounds of the GP's noise meeting at
    e^6."""
    result = _bads_with_options(
        {"tol_fun": tol_fun, "max_fun_evals": 20}
    ).optimize()
    assert result["func_count"] <= 20


@pytest.mark.parametrize(
    "tol_mesh", [0, 0.0, -1.0, -1e-12, np.nan, np.inf, -np.inf], ids=repr
)
def test_tol_mesh_not_a_positive_finite_number_is_refused(tol_mesh):
    """A `tol_mesh` that is not a positive finite number is refused when
    `BADS` is created: at 0 or below, the mesh criterion never ended the
    run, with a `RuntimeWarning` from the logarithm of `tol_mesh`."""
    with pytest.raises(
        ValueError,
        match=r"options\['tol_mesh'\] needs to be a positive finite number",
    ):
        _bads_with_options({"tol_mesh": tol_mesh})


@pytest.mark.parametrize(
    "tol_mesh, on_mesh",
    [(1e-6, 2.0**-19), (1e-5, 2.0**-16), (3e-3, 2.0**-8), (1, 1.0)],
    ids=repr,
)
def test_tol_mesh_is_put_on_the_mesh(tol_mesh, on_mesh):
    """`tol_mesh` is stored as a float, and the run's tolerance is the
    smallest power of `poll_mesh_multiplier` (2) at least `tol_mesh`, as in
    MATLAB BADS (`setupvars.m`)."""
    bads = _bads_with_options({"tol_mesh": tol_mesh})
    assert type(bads.options["tol_mesh"]) is float
    assert bads.optim_state["tol_mesh"] == on_mesh


@pytest.mark.parametrize(
    "search_method",
    [
        [],
        "ES-wcm",
        [("ES-wcm",)],
        [("ES-cma", 1)],
        [("ES-wcm", 1), ("ES-foo", 1)],
        [("ES-wcm", 1), "ES-ell"],
        [(np.array(["ES-wcm", "ES-ell"]), 1)],
        [("ES-wcm", 1, 1), ("ES-ell", 1)],
        np.array([["ES-wcm", 1, 1], ["ES-ell", 2, 1]], dtype=object),
    ],
    ids=[
        "empty",
        "string",
        "no_flag",
        "unknown",
        "one_unknown",
        "one_not_a_pair",
        "array_of_two_names",
        "three_elements",
        "array_of_triples",
    ],
)
def test_search_method_is_checked(search_method):
    """`search_method` is a non-empty list of pairs (name, sum-rule flag),
    each name "ES-wcm" or "ES-ell", checked when `BADS` is created; 1.1.0
    stopped at the first search, or at the first that chose an unknown
    name, and ignored the elements of an entry beyond its pair."""
    with pytest.raises(ValueError, match=r"search_method'\] needs to be"):
        _bads_with_options({"search_method": search_method})


@pytest.mark.parametrize(
    "search_method",
    [[("ES-ell", 1)], [["ES-wcm", 0], ["ES-ell", 1]], (("ES-wcm", True),)],
)
def test_search_method_of_known_searches_is_accepted(search_method):
    bads = _bads_with_options({"search_method": search_method})
    assert bads.options["search_method"] == search_method


@pytest.mark.parametrize(
    "name, value, as_list",
    [
        (
            "search_method",
            [(np.array(["ES-wcm"]), 1), ("ES-ell", 1)],
            [("ES-wcm", 1), ("ES-ell", 1)],
        ),
        (
            "search_method",
            np.array([["ES-wcm", "1"], ["ES-ell", "1"]]),
            [("ES-wcm", "1"), ("ES-ell", "1")],
        ),
        (
            "search_method",
            np.array([("ES-wcm", 1), ("ES-ell", 1)], dtype=object),
            [("ES-wcm", 1), ("ES-ell", 1)],
        ),
        (
            "search_acq_fcn",
            np.array(["acq_LCB", None], dtype=object),
            ("acq_LCB", None),
        ),
        ("search_acq_fcn", (np.array(["acq_LCB"]), None), ("acq_LCB", None)),
    ],
    ids=[
        "array_name",
        "string_array",
        "object_array",
        "acq_object_array",
        "acq_array_name",
    ],
)
def test_search_options_given_as_arrays_run_as_lists(name, value, as_list):
    """A `search_method` or `search_acq_fcn` given as a NumPy array, or with
    a NumPy array of one element for a name, which the searches compare as
    they compare a string and 1.1.0 ran, runs as the same list does."""
    bads = _bads_with_options({name: value, "max_fun_evals": 40})
    result = bads.optimize()
    reference = _bads_with_options({name: as_list, "max_fun_evals": 40})
    reference_result = reference.optimize()
    assert len(bads.optim_state["search_stats"]["success"]) > 0
    np.testing.assert_array_equal(result["x"], reference_result["x"])
    assert result["fval"] == reference_result["fval"]
    assert result["func_count"] == reference_result["func_count"]


@pytest.mark.parametrize(
    "search_acq_fcn",
    [
        "acq_LCB",
        ("acq_LCB",),
        ("acq_EI", None),
        [None, None],
        2.0,
        (np.array(["acq_LCB", "acq_LCB"]), None),
        ("acq_LCB", None, 1.0),
    ],
    ids=[
        "string",
        "one_element",
        "another_name",
        "no_name",
        "number",
        "array_of_two_names",
        "three_elements",
    ],
)
def test_search_acq_fcn_other_than_lcb_is_refused(search_acq_fcn):
    """`search_acq_fcn` is the pair ("acq_LCB", sqrt_beta), checked when
    `BADS` is created; 1.1.0 stopped at the first search, and PyBADS
    stopped there too for another name, or with an unrelated error when
    `BADS` was created for a value that is not a sequence of two, and
    ignored the elements beyond the pair."""
    with pytest.raises(ValueError, match=r"search_acq_fcn'\] needs to be"):
        _bads_with_options({"search_acq_fcn": search_acq_fcn})


@pytest.mark.parametrize("hedge_beta", [0, 0.0, 1, 1e3, np.float64(0.5)])
def test_hedge_beta_finite_number_at_least_zero_is_accepted(hedge_beta):
    bads = _bads_with_options({"hedge_beta": hedge_beta})
    assert bads.options["hedge_beta"] == hedge_beta


@pytest.mark.parametrize(
    "hedge_decay",
    [
        -0.1,
        -1e-12,
        1 + 1e-12,
        2.0,
        50.0,
        np.nan,
        np.inf,
        True,
        "0.5",
        0.5 + 0j,
        np.array([0.5, 0.6]),
    ],
)
def test_hedge_decay_outside_zero_one_is_refused(hedge_decay):
    """A `hedge_decay` outside [0, 1] is refused when `BADS` is created:
    above 1 the hedge's gains grow until they overflow, and below 0 they
    alternate in sign; MATLAB BADS runs with them."""
    with pytest.raises(
        ValueError, match=r"hedge_decay'\] needs to lie between 0 and 1"
    ):
        _bads_with_options({"hedge_decay": hedge_decay})


@pytest.mark.parametrize("hedge_decay", [0, 0.5, 1, 1.0, np.float64(0.9)])
def test_hedge_decay_from_zero_to_one_is_accepted(hedge_decay):
    bads = _bads_with_options({"hedge_decay": hedge_decay})
    assert bads.options["hedge_decay"] == hedge_decay


@pytest.mark.parametrize(
    "name, value",
    [
        ("hedge_gamma", np.array([0.1])),
        ("hedge_gamma", np.array([[0.1]])),
        ("hedge_gamma", np.complex128(0.1 - 5j)),
        ("hedge_gamma", Decimal("0.25")),
        ("hedge_beta", np.array([[1.0]])),
        ("hedge_beta", np.array([True])),
        ("hedge_beta", np.complex128(1.0)),
        ("hedge_beta", Fraction(1, 4)),
        ("hedge_beta", 10**400),
        ("hedge_decay", np.array([0.5])),
        ("hedge_decay", np.array([True])),
        ("hedge_decay", np.complex128(0.5 + 1j)),
        ("improvement_quantile", np.array([0.3])),
        ("improvement_quantile", np.complex128(0.3)),
        ("improvement_quantile", Fraction(1, 3)),
        ("tol_mesh", "1e-6"),
        ("tol_mesh", True),
        ("tol_mesh", np.array(1e-6)),
        ("tol_mesh", np.array([1e-6])),
        ("tol_mesh", np.complex128(1e-6)),
        ("tol_mesh", Decimal("1e-6")),
    ],
)
def test_real_valued_options_refuse_what_is_not_a_real_number(name, value):
    """`hedge_gamma`, `hedge_beta`, `hedge_decay`, `improvement_quantile`
    and `tol_mesh` take a real number, a Python or NumPy integer or float
    that is not a boolean: an array, of one element too, a NumPy complex
    number, which NumPy orders, a `Decimal`, a `Fraction` or an integer too
    large for a float is refused when `BADS` is created. Some of them passed
    the range checks and stopped the run at its first search with an
    unrelated error (a `hedge_decay` of `[0.5]`, a `hedge_gamma` of shape
    (1, 1)), and a string `tol_mesh` stopped the creation of `BADS` with
    NumPy's `TypeError`."""
    with pytest.raises(ValueError, match=rf"{name}'\] needs to"):
        _bads_with_options({name: value})


@pytest.mark.parametrize(
    "name, value",
    [
        ("hedge_gamma", np.float32(0.25)),
        ("hedge_gamma", np.int64(0)),
        ("hedge_beta", 1),
        ("hedge_beta", np.float64(0.5)),
        ("hedge_decay", np.float16(0.5)),
        ("improvement_quantile", np.float64(0.25)),
        ("tol_mesh", np.float32(1e-4)),
        ("tol_mesh", np.int64(1)),
    ],
)
def test_real_valued_options_are_stored_as_floats(name, value):
    bads = _bads_with_options({name: value})
    assert bads.options[name] == value
    assert type(bads.options[name]) is float


@pytest.mark.parametrize("tol_fun", [1e-3, 1e-6, 0.1])
@pytest.mark.parametrize("D", [1, 2, 5, 20, 60])
def test_hedge_beta_and_hedge_decay_defaults_are_accepted(D, tol_fun):
    """The defaults, `hedge_beta = 1e-3 / tol_fun` and `hedge_decay =
    0.1 ** (1 / (2 * D))`, are accepted at every `D`."""
    bads = _bads_with_options({"tol_fun": tol_fun}, D)
    assert bads.options["hedge_beta"] == 1e-3 / tol_fun
    assert bads.options["hedge_decay"] == 0.1 ** (1 / (2 * D))


@pytest.mark.parametrize(
    "sqrt_beta",
    [0, -1.0, np.nan, np.inf, "acq_schedule", np.array([1.0, 2.0]), True],
    ids=["zero", "negative", "nan", "inf", "name", "2 elements", "bool"],
)
def test_search_sqrt_beta_refused_before_any_evaluation(sqrt_beta):
    """A `sqrt_beta` of `search_acq_fcn` that is not None, a callable or a
    positive finite real number, which the search's LCB refuses, is refused
    when `BADS` is created, before the target is evaluated."""
    calls = []

    def target(x):
        calls.append(x)
        return _quadratic(x)

    with pytest.raises(
        ValueError,
        match=r"options\['search_acq_fcn'\]\[1\] \(sqrt_beta\) needs to be "
        r"None \(the default schedule\), a callable",
    ):
        BADS(
            target,
            np.array([0.5, 0.0]),
            -5 * np.ones(2),
            5 * np.ones(2),
            -3 * np.ones(2),
            3 * np.ones(2),
            options={**OPTIONS, "search_acq_fcn": ("acq_LCB", sqrt_beta)},
        )
    assert calls == []


@pytest.mark.parametrize(
    "sqrt_beta",
    [None, 2.0, np.float64(0.5), np.array([1.0]), lambda t, n_vars: -1.0],
    ids=["None", "float", "np.float64", "1-element", "callable"],
)
def test_search_sqrt_beta_accepted(sqrt_beta):
    """None, a positive finite real number and a callable are accepted when
    `BADS` is created; a callable's value is checked at each call."""
    bads = BADS(
        _quadratic,
        np.array([0.5, 0.0]),
        -5 * np.ones(2),
        5 * np.ones(2),
        -3 * np.ones(2),
        3 * np.ones(2),
        options={**OPTIONS, "search_acq_fcn": ("acq_LCB", sqrt_beta)},
    )
    assert bads.options["search_acq_fcn"][1] is sqrt_beta


def _bads_with_n_search_iter(n_search_iter):
    return BADS(
        _quadratic,
        np.array([0.5, 0.0]),
        -5 * np.ones(2),
        5 * np.ones(2),
        -3 * np.ones(2),
        3 * np.ones(2),
        options={
            **OPTIONS,
            "n_search_iter": n_search_iter,
            "max_fun_evals": 60,
        },
    )


@pytest.mark.parametrize(
    "n_search_iter", [0, -1, 0.5, 2.5, np.nan, np.inf, True, "2"]
)
def test_n_search_iter_not_a_positive_integer_is_refused(n_search_iter):
    """An `n_search_iter` that is not a positive integer is refused when
    `BADS` is created: the run stopped at its first search, with
    `ZeroDivisionError` at 0, `TypeError` at 0.5 or 2.5 and NumPy's
    `ValueError` at -1."""
    with pytest.raises(
        ValueError, match=r"n_search_iter'\] needs to be a positive integer"
    ):
        _bads_with_n_search_iter(n_search_iter)


@pytest.mark.parametrize("n_search_iter", [1, 2, 3.0, np.int64(4)])
def test_n_search_iter_positive_integer_is_accepted(n_search_iter):
    bads = _bads_with_n_search_iter(n_search_iter)
    assert bads.options["n_search_iter"] == n_search_iter
    assert type(bads.options["n_search_iter"]) is int
    result = bads.optimize()
    assert np.isfinite(result["fval"])


@pytest.mark.parametrize(
    "n_search_iter, n_search", [(4097, 4096), (2**70, 4096), (11, 10)]
)
def test_n_search_iter_above_n_search_is_refused(n_search_iter, n_search):
    """An `n_search_iter` above `n_search`, which leaves each generation of
    the ES search without a candidate (`n_search / n_search_iter` rounded
    down), is refused when `BADS` is created; 1.1.0 stopped the run at its
    first search with `IndexError`, and a Python integer beyond 64 bits
    raised `TypeError` from `np.isfinite` in the check."""
    with pytest.raises(
        ValueError,
        match=r"n_search_iter'\] needs to be a positive integer, at most "
        r"options\['n_search'\]",
    ):
        _bads_with_options(
            {"n_search_iter": n_search_iter, "n_search": n_search}
        )


@pytest.mark.parametrize(
    "n_search", [0, -1, 2.5, np.nan, np.inf, True, "4096", np.array([4096])]
)
def test_n_search_not_a_positive_integer_is_refused(n_search):
    """An `n_search`, the number of candidates of the ES search, that is not
    a positive integer is refused when `BADS` is created; 1.1.0 stopped the
    run at its first search (`IndexError` below 1, `TypeError` for a string,
    `ValueError` for NaN) or ran with a fraction."""
    with pytest.raises(
        ValueError, match=r"n_search'\] needs to be a positive integer"
    ):
        _bads_with_options({"n_search": n_search})


def test_n_search_iter_equal_to_n_search_runs():
    """At `n_search_iter = n_search` each generation of the ES search draws
    one candidate, and the run completes; a whole-number `n_search` is
    stored as an integer."""
    bads = _bads_with_options(
        {"n_search": 10.0, "n_search_iter": 10, "max_fun_evals": 40}
    )
    assert bads.options["n_search"] == 10
    assert type(bads.options["n_search"]) is int
    result = bads.optimize()
    assert np.isfinite(result["fval"])


def test_accelerate_mesh_steps_takes_an_integer_beyond_64_bits():
    """A Python integer too large for NumPy's 64-bit integers is a positive
    integer, which the check refused with `TypeError` from `np.isfinite`."""
    bads = _bads_with_options({"accelerate_mesh_steps": 2**70})
    assert bads.options["accelerate_mesh_steps"] == 2**70
    assert type(bads.options["accelerate_mesh_steps"]) is int


def _bads_with_periodic_vars(periodic_vars, x0=None, lb=None, ub=None):
    D = 3
    lb = -5 * np.ones(D) if lb is None else lb
    ub = 5 * np.ones(D) if ub is None else ub
    return BADS(
        _quadratic,
        x0,
        lb,
        ub,
        -3 * np.ones(D),
        3 * np.ones(D),
        options={**OPTIONS, "periodic_vars": periodic_vars},
    )


@pytest.mark.parametrize(
    "x0", [np.array([0.5, 0.0, 1.0]), None], ids=["x0", "random_x0"]
)
@pytest.mark.parametrize(
    "periodic_vars",
    [[2, 0], (0, 2), np.array([2, 0]), np.array([0, 2], dtype=np.uint8)],
    ids=["list", "tuple", "array", "uint8"],
)
def test_periodic_vars_are_indices(x0, periodic_vars):
    """`periodic_vars` takes the 0-based indices of the periodic variables,
    in any order, and stores them sorted; `optim_state` holds their mask."""
    bads = _bads_with_periodic_vars(periodic_vars, x0)
    assert bads.options["periodic_vars"] == [0, 2]
    assert all(type(i) is int for i in bads.options["periodic_vars"])
    assert bads.optim_state["periodic_vars"].tolist() == [[True, False, True]]


@pytest.mark.parametrize("periodic_vars", [1, np.int64(1)])
def test_periodic_vars_takes_one_index(periodic_vars):
    """A single index, a Python or NumPy integer, names one variable."""
    bads = _bads_with_periodic_vars(periodic_vars)
    assert bads.options["periodic_vars"] == [1]


@pytest.mark.parametrize(
    "x0", [np.array([0.5, 0.0, 1.0]), None], ids=["x0", "random_x0"]
)
@pytest.mark.parametrize(
    "periodic_vars, match",
    [
        ([3], "outside 0 to D - 1 = 2"),
        ([-1], "outside 0 to D - 1 = 2"),
        ([1, 1], "more than once"),
        ([True], "list of the indices"),
        ([0, True], "list of the indices"),
        (np.array([True, False, True]), "list of the indices"),
        ([0.0], "list of the indices"),
        ("1", "list of the indices"),
        ([[0, 1]], "list of the indices"),
    ],
    ids=[
        "out",
        "negative",
        "repeated",
        "bool",
        "int and bool",
        "mask",
        "float",
        "string",
        "2d",
    ],
)
def test_periodic_vars_refused(x0, periodic_vars, match):
    """A `periodic_vars` that is not a list of distinct indices from 0 to
    `D - 1` is refused with `ValueError` when `BADS` is created, before the
    variables are transformed, also with a random `x0`, whose draw
    transforms them. A boolean mask is refused rather than read as the
    indices 0 and 1."""
    with pytest.raises(ValueError, match=match):
        _bads_with_periodic_vars(periodic_vars, x0)


@pytest.mark.parametrize("x0_first", [2 * np.pi, 0.0], ids=["ub", "lb"])
def test_start_on_a_periodic_bound_is_the_lower_bound(x0_first):
    """A start on either bound of a periodic variable whose plausible bounds
    are its hard bounds, where the grid holds both, is the same point: the
    run starts from the lower bound, where the candidates are wrapped."""
    bads = BADS(
        _quadratic,
        np.array([x0_first, 0.5]),
        np.array([0.0, -5.0]),
        np.array([2 * np.pi, 5.0]),
        np.array([0.0, -2.0]),
        np.array([2 * np.pi, 2.0]),
        options={**OPTIONS, "periodic_vars": [0]},
    )
    assert bads.u[0] == bads.optim_state["lb"][0, 0]


def test_periodic_vars_need_a_gpyreg_with_periods(monkeypatch):
    """With a gpyreg whose kernels take no periods (before 1.4.0), a
    periodic_vars that names a variable is refused when `BADS` is created,
    before a run spends evaluations; without periodic variables `BADS` runs
    as before."""
    import pybads.bads.bads as bads_module

    monkeypatch.setattr(bads_module, "_gpyreg_takes_periods", lambda: False)
    with pytest.raises(ImportError, match="gpyreg 1.4.0 or later"):
        _bads_with_periodic_vars([0], np.array([0.5, 0.0, 1.0]))
    assert _bads_with_periodic_vars(None).options["periodic_vars"] is None


def test_periodic_vars_need_finite_bounds():
    """The hard bounds of a periodic variable set its period, and must be
    finite, as in MATLAB BADS."""
    lb = np.array([-5.0, -np.inf, -5.0])
    with pytest.raises(ValueError, match=r"variables \[1\] .* not finite"):
        _bads_with_periodic_vars([0, 1], np.array([0.5, 0.0, 1.0]), lb=lb)


def test_periodic_vars_never_log_transformed():
    """A periodic variable stays in linear coordinates, where its period is
    the width of its bounds, even when its bounds would take it to log
    coordinates, as in MATLAB BADS (`setupvars.m`)."""
    D = 2
    bads = BADS(
        _quadratic,
        np.array([2.0, 2.0]),
        1e-3 * np.ones(D),
        1e3 * np.ones(D),
        1e-2 * np.ones(D),
        1e2 * np.ones(D),
        options={**OPTIONS, "periodic_vars": [1]},
    )
    assert bads.var_transf.apply_log_t.ravel().tolist() == [True, False]


@pytest.mark.parametrize(
    "x0", [np.array([0.5, 0.0]), None], ids=["x0", "random_x0"]
)
@pytest.mark.parametrize(
    "periodic_vars", [[], np.array([], dtype=int)], ids=["list", "array"]
)
def test_empty_periodic_vars_stands_for_none(x0, periodic_vars):
    """An empty `periodic_vars` names no periodic variable, as in MATLAB
    BADS (`setupvars.m`), and stands for `None`, its default."""
    bads = BADS(
        _quadratic,
        x0,
        -5 * np.ones(2),
        5 * np.ones(2),
        -3 * np.ones(2),
        3 * np.ones(2),
        options={**OPTIONS, "periodic_vars": periodic_vars},
    )
    assert bads.options["periodic_vars"] is None
    assert not np.any(bads.optim_state["periodic_vars"])


@pytest.mark.parametrize(
    "name, value",
    [
        ("variational_sampler", "malasample"),
        ("warp_every_iters", 5),
        ("min_iter", 2),
        ("diagnostics", False),
        ("gp_cov_fun", 1),
    ],
)
def test_options_of_pyvbmc_without_effect_are_unknown(name, value):
    """The options that no code of PyBADS read and that MATLAB BADS does not
    have, most of them PyVBMC's leftovers, are not options of PyBADS:
    setting one raises `ValueError`, as for any unknown name."""
    with pytest.raises(ValueError, match=f"The option {name} does not exist"):
        _bads_with_options({name: value})


@pytest.mark.parametrize(
    "name, value",
    [
        ("gp_samples", 0),
        ("gp_method", "nearest"),
        ("chol_attempts", 0),
        ("poll_method", "poll_mads_2n"),
    ],
)
def test_options_of_matlab_without_effect_are_accepted(name, value):
    """The options named after MATLAB BADS's that PyBADS does not use stay
    options, so that setting one is not an error, and their descriptions
    say that they are unused."""
    bads = _bads_with_options({name: value})
    assert bads.options[name] == value
    assert "unused" in bads.options.descriptions[name]
