"""Reproducibility of seeded runs: the `random_seed` option fixes a run, and
`np.random.seed` before construction fixes a run without it."""

import numpy as np
import pytest

from pybads import BADS

D = 3


@pytest.fixture(autouse=True)
def _restore_global_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def _noisy_sphere(noise_seed):
    """Sphere with Gaussian noise from its own generator."""
    rng = np.random.default_rng(noise_seed)

    def fun(x):
        return _sphere(x) + rng.standard_normal()

    return fun


def _noisy_sphere_with_sd(noise_seed):
    """Sphere with heteroskedastic noise, returning its standard deviation."""
    rng = np.random.default_rng(noise_seed)

    def fun(x):
        y = _sphere(x)
        sd = 1.0 + 0.1 * np.sqrt(y)
        return y + sd * rng.standard_normal(), sd

    return fun


def _make_bads(fun, seed, **options):
    opts = {"display": "off", "max_fun_evals": 60}
    if seed is not None:
        opts["random_seed"] = seed
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


def _summary(bads, result):
    logger = bads.function_logger
    yval_vec = result["yval_vec"]
    return {
        "x": np.asarray(result["x"]).copy(),
        "fval": result["fval"],
        "func_count": result["func_count"],
        "yval_vec": None if yval_vec is None else np.asarray(yval_vec).copy(),
        "X": logger.X[logger.X_flag].copy(),
        "Y": logger.Y[logger.X_flag].copy(),
    }


def _run(fun, seed, **options):
    bads = _make_bads(fun, seed, **options)
    return _summary(bads, bads.optimize())


def _same(a, b):
    return all(
        (a[k] is None and b[k] is None)
        or (
            a[k] is not None
            and b[k] is not None
            and np.array_equal(a[k], b[k])
        )
        for k in a
    )


@pytest.fixture(scope="module")
def seeded_run():
    # Module scope runs outside the per-test snapshot of the global state.
    state = np.random.get_state()
    try:
        return _run(_sphere, 42)
    finally:
        np.random.set_state(state)


def test_seed_fixes_run(seeded_run):
    assert _same(seeded_run, _run(_sphere, 42))
    assert not _same(seeded_run, _run(_sphere, 43))


def test_seed_ignores_global_draws(seeded_run):
    np.random.seed(12345)
    bads = _make_bads(_sphere, 42)
    np.random.rand(7)
    assert _same(seeded_run, _summary(bads, bads.optimize()))


def test_seed_none_follows_global_seed():
    np.random.seed(5)
    first = _run(_sphere, None)
    np.random.seed(5)
    assert _same(first, _run(_sphere, None))


@pytest.mark.parametrize(
    "make_fun, options",
    [
        (_noisy_sphere, {"uncertainty_handling": True}),
        (
            _noisy_sphere_with_sd,
            {"uncertainty_handling": True, "specify_target_noise": True},
        ),
    ],
    ids=["inferred_noise", "specified_noise"],
)
def test_seed_fixes_noisy_run(make_fun, options):
    first = _run(make_fun(0), 7, **options)
    assert _same(first, _run(make_fun(0), 7, **options))
