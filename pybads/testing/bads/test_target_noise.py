"""Options and results specific to a target that returns the standard
deviation of its noise (`specify_target_noise`)."""

import numpy as np
import pytest

from pybads import BADS

D = 3


def _noisy_sphere_with_estimated_sd(noise_seed):
    """Sphere with heteroskedastic noise, returning an estimate of its
    standard deviation that varies from call to call, as the estimate of a
    stochastic target does."""
    rng = np.random.default_rng(noise_seed)

    def fun(x):
        y = float(np.sum(np.atleast_2d(x) ** 2))
        sd = 1.0 + 0.1 * np.sqrt(y)
        sd_estimate = sd * np.exp(0.3 * rng.standard_normal())
        return y + sd * rng.standard_normal(), sd_estimate

    return fun


def _make_bads(fun, **options):
    opts = {
        "display": "off",
        "max_fun_evals": 60,
        "random_seed": 7,
        "uncertainty_handling": True,
        "specify_target_noise": True,
    }
    opts.update(options)
    return BADS(
        fun,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=opts,
    )


def _warns_noise_size_ignored(caplog):
    return any(
        record.name == "BADS"
        and "options['noise_size'] is ignored" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("noise_size", [1.0, np.array([1.0])], ids=str)
def test_noise_size_ignored_warns(noise_size, caplog):
    with caplog.at_level("WARNING", logger="BADS"):
        _make_bads(_noisy_sphere_with_estimated_sd(0), noise_size=noise_size)
    assert _warns_noise_size_ignored(caplog)


@pytest.mark.parametrize("noise_size", [None, 0.0], ids=str)
def test_noise_size_empty_or_zero_is_silent(noise_size, caplog):
    with caplog.at_level("WARNING", logger="BADS"):
        _make_bads(_noisy_sphere_with_estimated_sd(0), noise_size=noise_size)
    assert not _warns_noise_size_ignored(caplog)


def test_final_estimate_weights_samples_by_precision():
    """`fval` and `fsd` combine the final samples at the returned point by
    the precisions that the target returns, as MATLAB BADS's
    `FinalEstimate` does."""
    result = _make_bads(_noisy_sphere_with_estimated_sd(0)).optimize()
    y = np.asarray(result["yval_vec"], dtype=float)
    sd = np.asarray(result["ysd_vec"], dtype=float)
    assert y.shape == sd.shape == (10,)
    precision = 1 / sd**2
    assert result["fval"] == pytest.approx(
        np.sum(y * precision) / np.sum(precision), rel=1e-12
    )
    assert result["fsd"] == pytest.approx(
        1 / np.sqrt(np.sum(precision)), rel=1e-12
    )
    # The weighting matters: the plain mean of these samples differs.
    assert abs(result["fval"] - np.mean(y)) > 1e-6


def test_final_estimate_from_one_sample():
    """With one final sample, `fval` and `fsd` are the sample and the
    standard deviation that the target returns with it."""
    result = _make_bads(
        _noisy_sphere_with_estimated_sd(0), noise_final_samples=1
    ).optimize()
    y = np.asarray(result["yval_vec"], dtype=float)
    sd = np.asarray(result["ysd_vec"], dtype=float)
    assert y.shape == sd.shape == (1,)
    assert result["fval"] == pytest.approx(y[0], rel=1e-12)
    assert result["fsd"] == pytest.approx(sd[0], rel=1e-12)
