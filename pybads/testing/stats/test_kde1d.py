import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm

from pybads.stats import kde1d


def test_kde1d_standard_normal():
    """The estimate from 10,000 standard normal samples is a density close
    to the normal smoothed by the kernel, N(0, 1 + h**2) for a bandwidth
    h (at seeds 0-4 the largest deviation is below 0.02)."""
    samples = np.random.default_rng(0).standard_normal(10_000)
    density, xmesh, bandwidth = kde1d(samples, 2**10, -6, 6)
    assert density.shape == (2**10,)
    assert xmesh.shape == (2**10,)
    h = float(bandwidth)
    assert h > 0
    assert np.all(density >= 0)
    assert np.isclose(trapezoid(density, xmesh), 1, atol=1e-2)
    expected = norm.pdf(xmesh, scale=np.sqrt(1 + h**2))
    assert np.max(np.abs(density - expected)) < 0.05
