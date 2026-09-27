import numpy as np
import pytest

from pybads.init_functions import init_sobol


def _design(D, fun_eval_start, u0=None, rng=0):
    u0 = np.zeros(D) if u0 is None else u0
    return init_sobol(
        u0,
        -2 * np.ones(D),
        2 * np.ones(D),
        -np.ones(D),
        np.ones(D),
        fun_eval_start,
        rng=np.random.default_rng(rng),
    )


@pytest.mark.parametrize(
    "D, fun_eval_start, n_points",
    [(3, 3, 4), (3, 5, 8), (2, 2, 4), (4, 3, 8), (5, 20, 32)],
)
def test_init_sobol_returns_number_of_points(D, fun_eval_start, n_points):
    u_init, n_samples = _design(D, fun_eval_start)
    assert u_init.shape == (n_points, D)
    assert n_samples == n_points
