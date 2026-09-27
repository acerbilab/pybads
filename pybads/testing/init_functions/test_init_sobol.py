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


def test_init_sobol_design_follows_the_generator_not_the_start():
    """The generator decides the design, and the start does not: starts
    inside the plausible box and on its bounds give one design for one
    seed, and two seeds give two designs."""
    D = 3
    starts = [
        np.zeros(D),
        np.array([0.5, -0.25, 0.75]),
        -np.ones(D),
        np.array([1.0, 0.3, -1.0]),
    ]
    design, _ = _design(D, D, u0=starts[0], rng=0)
    for u0 in starts[1:]:
        assert np.array_equal(_design(D, D, u0=u0, rng=0)[0], design)
    other, _ = _design(D, D, u0=starts[0], rng=1)
    assert not np.array_equal(other, design)
    assert np.all((other >= -1) & (other <= 1))
