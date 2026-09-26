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
