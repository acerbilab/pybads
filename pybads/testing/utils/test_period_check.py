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


def test_tiny_negative_offset_wraps_to_the_lower_bound():
    """A coordinate just below the lower bound, where `np.mod` rounds to the
    period itself, lands on the lower bound or just below the upper one,
    within the bounds."""
    u = np.array([[-1.0 - 1e-17, 0.0, -2.0 - 1e-16]])
    wrapped = period_check(u, LB, UB, MASK)
    assert np.all(wrapped[:, [0, 2]] >= LB[:, [0, 2]])
    assert np.all(wrapped[:, [0, 2]] <= UB[:, [0, 2]])


def test_one_dimensional_point_and_flat_mask():
    """A single point may be given as a 1-D array, and the bounds and the
    mask flat."""
    wrapped = period_check(
        np.array([1.5, 7.0, 2.5]), LB.ravel(), UB.ravel(), MASK.ravel()
    )
    assert wrapped.shape == (3,)
    np.testing.assert_allclose(wrapped, [-0.5, 7.0, -1.5], atol=1e-12)
