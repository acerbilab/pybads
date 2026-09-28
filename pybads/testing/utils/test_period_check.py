import numpy as np
import pytest

from pybads.utils.period_check import period_check

LB = np.array([[-1.0, 0.0, -2.0]])
UB = np.array([[1.0, 5.0, 2.0]])
MASK = np.array([[True, False, True]])


def test_no_periodic_variable_returns_the_input():
    """Without periodic variables `period_check` returns its input itself,
    so that a run without them computes exactly what it did before."""
    u = np.array([[3.0, 7.0, -9.0]])
    assert period_check(u, LB, UB, None) is u
    assert period_check(u, LB, UB, np.zeros((1, 3), dtype=bool)) is u


@pytest.mark.parametrize(
    "u, expected",
    [
        ([0.5, 7.0, 1.0], [0.5, 7.0, 1.0]),
        ([1.5, 7.0, 2.5], [-0.5, 7.0, -1.5]),
        ([-1.5, -3.0, -2.5], [0.5, -3.0, 1.5]),
        ([5.5, 0.0, -10.0], [-0.5, 0.0, -2.0]),
        ([1.0, 5.0, 2.0], [-1.0, 5.0, -2.0]),
        ([-1.0, 0.0, -2.0], [-1.0, 0.0, -2.0]),
    ],
    ids=["inside", "above", "below", "periods away", "upper bound", "lower"],
)
def test_wraps_the_periodic_coordinates(u, expected):
    """A periodic coordinate is wrapped into `[lb, ub)`, the upper bound
    to the lower one, whatever the number of periods it lies away; the
    other coordinates are left as they are, even out of bounds."""
    u = np.array([u])
    u_before = u.copy()
    wrapped = period_check(u, LB, UB, MASK)
    np.testing.assert_allclose(wrapped, [expected], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(u, u_before)


@pytest.mark.parametrize(
    "lb, ub",
    [(0.1, 2.3), (1.7260070718156868, 4.180697584676979)],
    ids=["mod", "rounding"],
)
def test_just_below_the_lower_bound_wraps_below_the_upper_one(lb, ub):
    """A coordinate one step of rounding below the lower bound comes out in
    `[lb, ub)`: `np.mod` returns the period itself for it, or the sum of
    `lb` and the shift rounds to `ub`, and both are the point `lb`."""
    u = np.array([[np.nextafter(lb, -np.inf)]])
    wrapped = period_check(u, np.array([[lb]]), np.array([[ub]]), [[True]])
    assert lb <= wrapped[0, 0] < ub


def test_one_dimensional_point_and_flat_mask():
    """A single point may be given as a 1-D array, and the bounds and the
    mask flat."""
    wrapped = period_check(
        np.array([1.5, 7.0, 2.5]), LB.ravel(), UB.ravel(), MASK.ravel()
    )
    assert wrapped.shape == (3,)
    np.testing.assert_allclose(wrapped, [-0.5, 7.0, -1.5], atol=1e-12)
