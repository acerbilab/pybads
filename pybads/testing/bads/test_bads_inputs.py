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
