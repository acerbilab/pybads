"""Reproducibility of seeded runs: the `random_seed` option fixes a run, and
`np.random.seed` before construction fixes a run without it. Every draw of a
run comes from its generator `bads.rng`, so that after construction a run
draws nothing from NumPy's global random state."""

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


def _make_bads(fun, seed, random_x0=False, **options):
    opts = {"display": "off", "max_fun_evals": 60}
    if seed is not None:
        opts["random_seed"] = seed
    opts.update(options)
    return BADS(
        fun,
        None if random_x0 else np.ones(D) * 4,
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


@pytest.fixture(scope="module")
def unseeded_run():
    """A run with `random_seed=None` after `np.random.seed(5)`."""
    state = np.random.get_state()
    try:
        np.random.seed(5)
        return _run(_sphere, None)
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


def test_seed_none_follows_global_seed(unseeded_run):
    np.random.seed(5)
    assert _same(unseeded_run, _run(_sphere, None))


def test_seed_none_ignores_draws_after_construction(unseeded_run):
    np.random.seed(5)
    bads = _make_bads(_sphere, None)
    np.random.rand(7)
    assert _same(unseeded_run, _summary(bads, bads.optimize()))


@pytest.mark.parametrize(
    "random_x0", [False, True], ids=["given_x0", "random_x0"]
)
def test_seed_none_does_not_reseed(random_x0):
    """Construction draws from the global state only the four integers that
    derive the generator (`pybads.rng.get_rng`), and does not reseed it."""
    np.random.seed(11)
    _make_bads(_sphere, None, random_x0=random_x0)
    after_construction = np.random.random()
    np.random.seed(11)
    np.random.randint(0, 2**32, size=4, dtype=np.uint32)
    assert np.random.random() == after_construction


def test_seed_accepts_generator():
    rng = np.random.default_rng(3)
    bads = _make_bads(_sphere, rng)
    assert bads.rng is rng
    assert bads.optim_state["random_seed"] is None


@pytest.mark.parametrize(
    "seed, reported",
    [
        (42, 42),
        (42.0, 42),
        (np.int64(42), 42),
        (np.random.SeedSequence(42), None),
    ],
    ids=["int", "whole_float", "numpy_int", "seed_sequence"],
)
def test_seed_types(seed, reported):
    """Each of these seeds gives the generator of `default_rng(42)`; the
    result reports the seed (`optim_state["random_seed"]`, which
    `OptimizeResult` copies) when it is an integer."""
    bads = _make_bads(_sphere, seed)
    expected = np.random.default_rng(42).bit_generator.state
    assert bads.rng.bit_generator.state == expected
    assert bads.optim_state["random_seed"] == reported
    assert type(bads.optim_state["random_seed"]) is type(reported)


@pytest.mark.parametrize("seed", [42.5, "42"], ids=["float", "string"])
def test_seed_rejects_other_values(seed):
    with pytest.raises(TypeError):
        _make_bads(_sphere, seed)


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


@pytest.mark.parametrize(
    "make_fun, options",
    [
        (lambda noise_seed: _sphere, {"random_x0": True}),
        (_noisy_sphere, {"uncertainty_handling": True}),
        (lambda noise_seed: _sphere, {"double_refit": True}),
        (
            lambda noise_seed: _sphere,
            {"double_refit": True, "use_slice_sampler": True},
        ),
    ],
    ids=[
        "deterministic_random_x0",
        "inferred_noise",
        "prior_samples",
        "slice_sampler",
    ],
)
def test_seeded_run_leaves_global_state_untouched(make_fun, options):
    """Every draw of a seeded run, construction included, comes from
    `bads.rng`: NumPy's global random state is the same before and after,
    so no draw site of PyBADS and no gpyreg call (the GP fits, the slice
    sampler) uses it. `double_refit` makes each refit draw a second starting
    point, from the priors or from the slice sampler."""
    np.random.seed(7)
    before = np.random.get_state()
    bads = _make_bads(make_fun(0), 3, **options)
    bads.optimize()
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
