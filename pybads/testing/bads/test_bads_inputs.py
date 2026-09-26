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
    "lb, ub",
    [([-10.0, -np.inf], [np.inf, np.inf]), ([-1.0, -np.inf], [1.0, 10.0])],
    ids=["bounded_below", "bounded_above"],
)
def test_half_bounded_variable_is_refused(lb, ub):
    with pytest.raises(ValueError, match="bads:HalfBounds"):
        BADS(
            _quadratic,
            np.array([0.5, 0.0]),
            np.array(lb),
            np.array(ub),
            np.array([0.1, -3.0]),
            np.array([0.9, 3.0]),
            options=OPTIONS,
        )


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
