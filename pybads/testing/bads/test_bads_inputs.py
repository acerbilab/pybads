"""The inputs of `BADS` as MATLAB BADS checks them (`boundscheck.m`,
`setupvars.m`): the bounds, the starting point and `non_box_cons`."""

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
