import copy

import gpyreg as gpr
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import norm

from pybads import BADS
from pybads.bads import gaussian_process_train
from pybads.bads.gaussian_process_train import (
    _cov_identifier_to_covariance_function,
    _get_fevals_data,
    _get_gp_training_options,
    _get_random_samples_from_priors_,
    _meanfun_name_to_mean_function,
    _robust_gp_fit_,
    add_and_update_gp,
    get_grid_search_neighbors,
    init_and_train_gp,
    local_gp_fitting,
)
from pybads.stats import get_hpd


def test_get_fevals_data_no_noise():
    D = 3
    f = lambda x: np.sum(x + 2, axis=1)
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D)) * 1

    bads = BADS(f, x0, None, None, plb, pub)

    # Make sure we get nothing out before data has not been added.
    X_train, y_train, s2_train, t_train = _get_fevals_data(
        bads.function_logger
    )

    assert X_train.shape == (0, 3)
    assert y_train.shape == (0, 1)
    assert s2_train is None
    assert t_train.shape == (0, 1)

    # Create dummy data.
    sample_count = 10
    window = bads.optim_state["pub"] - bads.optim_state["plb"]
    rnd_tmp = np.random.rand(sample_count, window.shape[1])
    Xs = window * rnd_tmp + bads.optim_state["plb"]
    ys = f(Xs)

    # Add dummy training data explicitly since function_logger
    # has a parameter transformer which makes everything hard.
    for sample_idx in range(sample_count):
        bads.function_logger.X_flag[sample_idx] = True
        bads.function_logger.X[sample_idx] = Xs[sample_idx]
        bads.function_logger.Y[sample_idx] = ys[sample_idx]
        bads.function_logger.fun_eval_time[sample_idx] = 1e-5

    # Then make sure we get that data back.
    X_train, y_train, s2_train, t_train = _get_fevals_data(
        bads.function_logger
    )

    assert np.all(X_train == Xs)
    assert np.all(y_train.flatten() == ys)
    assert s2_train is None
    assert np.all(t_train == 1e-5)


def test_get_fevals_data_noise():
    D = 3
    f = lambda x: (np.sum(np.atleast_2d(x) + 2, axis=1), 0)
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * -5
    pub = np.ones((1, D)) * 5
    options = {"specify_target_noise": True, "uncertainty_handling": True}

    bads = BADS(
        f,
        x0,
        plausible_lower_bounds=plb,
        plausible_upper_bounds=pub,
        options=options,
    )

    # Make sure we get nothing out before data has not been added.
    X_train, y_train, s2_train, t_train = _get_fevals_data(
        bads.function_logger
    )

    assert X_train.shape == (0, 3)
    assert y_train.shape == (0, 1)
    assert s2_train.shape == (0, 1)
    assert t_train.shape == (0, 1)

    # Create dummy data.
    sample_count = 10
    window = bads.optim_state["pub"] - bads.optim_state["plb"]
    rnd_tmp = np.random.rand(sample_count, window.shape[1])
    Xs = window * rnd_tmp + bads.optim_state["plb"]
    ys = []
    for x_idx in range(Xs.shape[0]):
        f_i, _ = f(Xs[x_idx])
        ys.append(f_i)
    ys = np.array(ys)

    # Add dummy training data explicitly since function_logger
    # has a parameter transformer which makes everything hard.
    for sample_idx in range(sample_count):
        bads.function_logger.X_flag[sample_idx] = True
        bads.function_logger.X[sample_idx] = Xs[sample_idx]
        bads.function_logger.Y[sample_idx] = ys[sample_idx]
        bads.function_logger.S[sample_idx] = 0.5 + sample_idx
        bads.function_logger.fun_eval_time[sample_idx] = 1e-5

    # Then make sure we get that data back.
    X_train, y_train, s2_train, t_train = _get_fevals_data(
        bads.function_logger
    )

    assert np.all(X_train == Xs)
    assert np.all(y_train.flatten() == ys.flatten())
    # The noise standard deviations, squared into variances.
    assert np.all(s2_train.flatten() == (0.5 + np.arange(sample_count)) ** 2)
    assert np.all(t_train == 1e-5)


def test_meanfun_name_to_mean_function():
    m1 = _meanfun_name_to_mean_function("zero")
    m2 = _meanfun_name_to_mean_function("const")
    m3 = _meanfun_name_to_mean_function("negquad")

    assert isinstance(m1, gpr.mean_functions.ZeroMean)
    assert isinstance(m2, gpr.mean_functions.ConstantMean)
    assert isinstance(m3, gpr.mean_functions.NegativeQuadratic)

    with pytest.raises(ValueError):
        m4 = _meanfun_name_to_mean_function("linear")
    with pytest.raises(ValueError):
        m5 = _meanfun_name_to_mean_function("quad")
    with pytest.raises(ValueError):
        m6 = _meanfun_name_to_mean_function("posquad")
    with pytest.raises(ValueError):
        m7 = _meanfun_name_to_mean_function("se")
    with pytest.raises(ValueError):
        m8 = _meanfun_name_to_mean_function("negse")
    with pytest.raises(ValueError):
        m9 = _meanfun_name_to_mean_function("linear")


def test_cov_identifier_to_covariance_function():
    c1 = _cov_identifier_to_covariance_function(2)
    c2 = _cov_identifier_to_covariance_function(3)
    c3 = _cov_identifier_to_covariance_function([3, 1])
    c4 = _cov_identifier_to_covariance_function([3, 3])
    c5 = _cov_identifier_to_covariance_function([3, 5])

    assert isinstance(c1, gpr.covariance_functions.SquaredExponential)
    assert isinstance(c2, gpr.covariance_functions.Matern)
    assert isinstance(c3, gpr.covariance_functions.Matern)
    assert isinstance(c4, gpr.covariance_functions.Matern)
    assert isinstance(c5, gpr.covariance_functions.Matern)

    assert c2.degree == 5
    assert c3.degree == 1
    assert c4.degree == 3
    assert c5.degree == 5

    with pytest.raises(ValueError):
        c6 = _cov_identifier_to_covariance_function(0)


def test_get_gp_training_options_samplers():
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    f = lambda x: np.sum(x + 2)
    bads = BADS(f, x0, lb, ub, plb, pub)

    hyp_dict = {"run_cov": np.eye(3)}
    hyp_dict_none = {"run_cov": None}
    bads.optim_state["eff_starting_points"] = 10
    bads.optim_state["ntrain"] = 10
    bads.optim_state["iter"] = 1
    bads.options["weighted_hyp_cov"] = False

    res1 = _get_gp_training_options(
        bads.optim_state,
        bads.iteration_history,
        bads.options,
        hyp_dict,
        8,
        bads.function_logger,
    )
    assert res1["sampler"] == "slicesample"


def test_get_gp_training_options_opts_N():
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    f = lambda x: np.sum(x + 2)
    bads = BADS(f, x0, lb, ub, plb, pub)

    bads.optim_state["eff_starting_points"] = 10
    bads.optim_state["ntrain"] = 10
    bads.optim_state["iter"] = 2
    bads.options["weighted_hyp_cov"] = False
    hyp_dict = {"run_cov": np.eye(3)}
    hyp_dict_none = {"run_cov": None}
    bads.options["gpretrainthreshold"] = 10

    res1 = _get_gp_training_options(
        bads.optim_state,
        bads.iteration_history,
        bads.options,
        hyp_dict,
        0,
        bads.function_logger,
    )
    assert res1["opts_N"] == 1


@pytest.mark.parametrize("max_fun_evals", [2, 3, 4, 5])
@pytest.mark.parametrize("D", [2, 3])
def test_get_gp_training_options_small_budget(monkeypatch, D, max_fun_evals):
    """With a budget no larger than the initial design, the budget counts as
    used up: the GP fits start from `gp_train_n_init_final` points, within
    the range of the schedule, from `gp_train_n_init` down."""
    import pybads.bads.gaussian_process_train as gpt

    seen = []
    original = gpt._get_gp_training_options

    def spy(*args, **kwargs):
        gp_train = original(*args, **kwargs)
        seen.append(gp_train["init_N"])
        return gp_train

    monkeypatch.setattr(gpt, "_get_gp_training_options", spy)
    bads = BADS(
        lambda x: float(np.sum(np.ravel(x) ** 2)),
        0.5 * np.ones(D),
        -5 * np.ones(D),
        5 * np.ones(D),
        -2 * np.ones(D),
        2 * np.ones(D),
        options={
            "display": "off",
            "max_fun_evals": max_fun_evals,
            "random_seed": 0,
        },
    )
    result = bads.optimize()
    assert bads.optim_state["eff_starting_points"] >= max_fun_evals
    assert seen
    assert all(n == bads.options["gp_train_n_init_final"] for n in seen)
    assert np.isfinite(result["fval"])


def test_gp_noise_variances_with_target_noise(monkeypatch):
    """With `specify_target_noise`, the GP holds the squares of the noise
    standard deviations that the target returns, after each rebuild of the
    local GP and each added point of a run."""
    import pybads.bads.bads as bads_module

    rng = np.random.default_rng(1000)

    def fun(x):
        y = np.sum(np.atleast_2d(x) ** 2)
        sd = 2 + np.sqrt(y)
        return y + sd * rng.standard_normal(), sd

    checked = {"local": 0, "add": 0}
    original_local = bads_module.local_gp_fitting
    original_add = bads_module.add_and_update_gp

    def spy_local(gp, current_point, function_logger, *args, **kwargs):
        out = original_local(
            gp, current_point, function_logger, *args, **kwargs
        )
        X = function_logger.X[function_logger.X_flag]
        S = function_logger.S[function_logger.X_flag]
        rows = [np.flatnonzero(np.all(X == x, axis=1))[0] for x in gp.X]
        assert np.array_equal(gp.s2, S[rows] ** 2)
        checked["local"] += 1
        return out

    def spy_add(function_logger, gp, x_new, y_new, sd_new=None, options=None):
        n_train = gp.X.shape[0]
        out = original_add(function_logger, gp, x_new, y_new, sd_new, options)
        assert gp.X.shape[0] == n_train + 1
        assert np.array_equal(gp.s2[-1:], np.atleast_2d(sd_new) ** 2)
        checked["add"] += 1
        return out

    monkeypatch.setattr(bads_module, "local_gp_fitting", spy_local)
    monkeypatch.setattr(bads_module, "add_and_update_gp", spy_add)
    D = 3
    BADS(
        fun,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={
            "display": "off",
            "max_fun_evals": 60,
            "random_seed": 0,
            "uncertainty_handling": True,
            "specify_target_noise": True,
        },
    ).optimize()
    assert checked["local"] > 0 and checked["add"] > 0


def test_gp_mean_prior_recentred_at_each_rebuild(monkeypatch):
    """After each rebuild of the local GP, the prior of the constant mean is
    MATLAB's empirical prior (gpdefBads.m): centred at the 90th percentile
    of the training targets (MATLAB's prctile1), with the standard deviation
    `(y90 - median(y)) / 5`."""
    import pybads.bads.bads as bads_module

    checked = {"local": 0}
    original_local = bads_module.local_gp_fitting

    def spy_local(gp, *args, **kwargs):
        out = original_local(gp, *args, **kwargs)
        y = gp.y.ravel()
        y90 = np.percentile(y, 90, method="hazen")
        kind, (mu, sigma) = gp.get_priors()["mean_const"]
        assert kind == "gaussian"
        assert np.isclose(mu.item(), y90, rtol=1e-12)
        assert np.isclose(sigma.item(), (y90 - np.median(y)) / 5, rtol=1e-12)
        checked["local"] += 1
        return out

    monkeypatch.setattr(bads_module, "local_gp_fitting", spy_local)
    D = 3
    BADS(
        lambda x: np.sum(np.atleast_2d(x) ** 2),
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={"display": "off", "max_fun_evals": 60, "random_seed": 0},
    ).optimize()
    assert checked["local"] > 0


def test_gp_noise_prior_follows_mesh_at_each_rebuild(monkeypatch):
    """In a deterministic run, after each rebuild of the local GP, the prior
    of the GP noise is centred at `log(noise_size) + mesh_noise_multiplier *
    log(mesh_size)`, as in MATLAB's gpdefBads.m, with the SD of the
    definition prior, 1."""
    import pybads.bads.bads as bads_module

    mesh_sizes = []
    original_local = bads_module.local_gp_fitting

    def spy_local(
        gp,
        current_point,
        function_logger,
        options,
        optim_state,
        *args,
        **kwargs,
    ):
        out = original_local(
            gp,
            current_point,
            function_logger,
            options,
            optim_state,
            *args,
            **kwargs,
        )
        mesh_size = optim_state["mesh_size"]
        centre = np.log(np.ravel(options["noise_size"])[0]) + options[
            "mesh_noise_multiplier"
        ] * np.log(mesh_size)
        kind, (mu, sigma) = gp.get_priors()["noise_log_scale"]
        assert kind == "gaussian"
        assert np.isclose(mu.item(), centre, rtol=1e-12)
        assert sigma.item() == 1.0
        mesh_sizes.append(mesh_size)
        return out

    monkeypatch.setattr(bads_module, "local_gp_fitting", spy_local)
    D = 3
    bads = BADS(
        lambda x: np.sum(np.atleast_2d(x) ** 2),
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={"display": "off", "max_fun_evals": 60, "random_seed": 0},
    )
    bads.optimize()
    assert bads.optim_state["uncertainty_handling_level"] == 0
    assert bads.options["mesh_noise_multiplier"] > 0
    assert min(mesh_sizes) < 1


def test_gp_log_lengthscale_bounds(monkeypatch):
    """The bounds of the GP log length scales are the logs of `tol_mesh`
    and of the maximum length scale, `min(100, 10 * (ub - lb) / scale)` in
    normalized units, as in MATLAB's gpdefBads.m."""
    import pybads.bads.bads as bads_module

    checked = {"local": 0}
    original_local = bads_module.local_gp_fitting

    def spy_local(
        gp,
        current_point,
        function_logger,
        options,
        optim_state,
        *args,
        **kwargs,
    ):
        out = original_local(
            gp,
            current_point,
            function_logger,
            options,
            optim_state,
            *args,
            **kwargs,
        )
        cov_range = np.minimum(
            100,
            10
            * (optim_state["ub"] - optim_state["lb"])
            / optim_state["scale"],
        ).ravel()
        lower, upper = gp.get_bounds()["covariance_log_lengthscale"]
        assert np.allclose(lower, np.log(optim_state["tol_mesh"]))
        assert np.allclose(upper, np.log(cov_range))
        checked["local"] += 1
        return out

    monkeypatch.setattr(bads_module, "local_gp_fitting", spy_local)
    D = 3
    BADS(
        lambda x: np.sum(np.atleast_2d(x) ** 2),
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={"display": "off", "max_fun_evals": 60, "random_seed": 0},
    ).optimize()
    assert checked["local"] > 0


def _make_bads(D=2, **options):
    return BADS(
        lambda x: float(np.sum(np.atleast_2d(x) ** 2)),
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options={"display": "off", "random_seed": 3, **options},
    )


def _initialized_bads(D=2, **options):
    bads = _make_bads(D, **options)
    gp, _, _, _ = bads._init_optimization_()
    return bads, gp


def test_gp_mean_starts_at_median_of_lowest_ceil_fraction(monkeypatch):
    """The constant mean of the GP starts at the median of the lowest
    `ceil(0.8 N)` initial targets, as in MATLAB's gpdefBads.m, and its
    prior stays centred on the high-density set, the lowest `round(0.8 N)`.
    The two differ at D = 4, whose N = 9 initial targets give 8 and 7."""
    import pybads.bads.gaussian_process_train as gpt_module

    seen = {}
    original_gp_hyp = gpt_module._gp_hyp

    def spy_gp_hyp(optim_state, options, plb, pub, gp, X, y, *args):
        gp, hyp0, gp_s_N = original_gp_hyp(
            optim_state, options, plb, pub, gp, X, y, *args
        )
        seen.update(y=y.ravel().copy(), hyp0=hyp0.copy())
        seen["prior"] = gp.get_priors()["mean_const"][1][0].item()
        return gp, hyp0, gp_s_N

    monkeypatch.setattr(gpt_module, "_gp_hyp", spy_gp_hyp)
    _initialized_bads(D=4)
    y = np.sort(seen["y"])
    assert y.size == 9
    assert seen["hyp0"][-1] == np.median(y[:8])
    assert seen["prior"] == np.median(y[:7])
    assert np.median(y[:8]) != np.median(y[:7])


def test_rebuild_substitutes_ill_defined_values():
    """An ill-defined target value in the training set of the local GP is
    replaced by the highest well-defined one, as in MATLAB's gpupdate.m.
    (The function logger refuses such values today.)"""
    bads, gp = _initialized_bads()
    logger = bads.function_logger
    rows = np.flatnonzero(logger.X_flag)
    logger.Y[rows[np.argmin(logger.Y[rows].ravel())]] = np.inf
    gp, _ = local_gp_fitting(
        gp,
        bads.u,
        logger,
        bads.options,
        bads.optim_state,
        bads.iteration_history,
        False,
        rng=bads.rng,
    )
    err = gp.temporary_data["err_y"].ravel()
    assert err.sum() == 1
    assert np.all(np.isfinite(gp.y))
    assert gp.y.ravel()[err].item() == np.max(gp.y.ravel()[~err])


def test_add_enters_ill_defined_value_as_highest():
    """A point added with an ill-defined value enters the GP with the
    highest value of its training set, as in MATLAB's gpupdate.m."""
    bads, gp = _initialized_bads()
    n, y_max = gp.y.shape[0], np.max(gp.y)
    gp = add_and_update_gp(
        bads.function_logger,
        gp,
        np.full((1, 2), 0.5),
        np.inf,
        None,
        bads.options,
    )
    assert gp.y.shape[0] == n + 1
    assert gp.y.ravel()[-1] == y_max
    assert np.all(np.isfinite(gp.predict(np.zeros((1, 2)))[0]))


def test_nearest_neighbors_keep_log_order_at_equal_distance():
    """The training set of the local GP is sorted by distance from the
    incumbent with a stable sort, as MATLAB's gpupdate.m: points at equal
    distance keep the order of the function log."""
    bads, gp = _initialized_bads()
    logger = bads.function_logger
    # A grid around the origin in shuffled order: the points that differ
    # only in the signs of their coordinates are at equal distance.
    grid = np.arange(-3, 4)
    U = np.array([(i, j) for i in grid for j in grid], dtype=float)
    U = U[np.random.default_rng(0).permutation(len(U))]
    n = len(U)
    logger.X[:n] = U
    logger.Y[:n, 0] = np.arange(n)  # the row of each point in the log
    logger.X_flag[:] = False
    logger.X_flag[:n] = True
    logger.X_max_idx = n - 1
    X, Y, _ = get_grid_search_neighbors(
        logger, np.zeros((1, 2)), gp, bads.options, bads.optim_state
    )
    rows = Y.ravel().astype(int)
    dist = np.sum(U**2, axis=1)
    assert len(rows) == n
    assert np.array_equal(X, U[rows])
    assert np.all(np.diff(dist[rows]) >= 0)
    ties = np.diff(dist[rows]) == 0
    assert np.any(ties)
    assert np.all(np.diff(rows)[ties] > 0)


@pytest.mark.parametrize(
    "name, mean",
    [
        ("zero", gpr.mean_functions.ZeroMean),
        ("const", gpr.mean_functions.ConstantMean),
    ],
)
def test_gp_mean_fun_accepted(name, mean):
    """`gp_mean_fun` accepts the constant mean, MATLAB's, and the zero
    mean, and the GP is built with it."""
    _, gp = _initialized_bads(gp_mean_fun=name)
    assert isinstance(gp.mean, mean)


@pytest.mark.parametrize("name", ["negquad", "se"])
def test_gp_mean_fun_refused(name):
    """Every other mean function is refused when `BADS` is created: the
    negative quadratic has the wrong shape for a minimizer, and the others
    cannot be built."""
    with pytest.raises(
        ValueError, match=r"options\['gp_mean_fun'\] should be 'const'"
    ):
        _make_bads(gp_mean_fun=name)


@pytest.mark.parametrize("value", ["ard", "foo"])
def test_gp_cov_prior_refused(value):
    """`gp_cov_prior` accepts only `"iso"`, the default: MATLAB's `"ard"`
    is not ported, and an unknown value is refused as MATLAB does, when
    `BADS` is created."""
    with pytest.raises(ValueError, match="'ard' is not supported"):
        _make_bads(gp_cov_prior=value)


def test_fit_lik_false_refused():
    """A fixed noise level (`fit_lik=False`) is refused when `BADS` is
    created, with MATLAB's message."""
    with pytest.raises(ValueError, match="Fixed noise not supported"):
        _make_bads(fit_lik=False)


def test_plateau_initial_design_runs():
    """A target equal on the whole initial design (a plateau outside a small
    ball) runs: the prior of the GP mean takes the SD 1 where the targets
    have no spread."""
    D = 3

    def fun(x):
        r2 = np.sum((np.asarray(x) - 1.9) ** 2)
        return float(r2) if r2 < 0.05 else 1e3

    bads = BADS(
        fun,
        np.zeros(D),
        -5 * np.ones(D),
        5 * np.ones(D),
        -2 * np.ones(D),
        2 * np.ones(D),
        options={"display": "off", "max_fun_evals": 100, "random_seed": 0},
    )
    result = bads.optimize()
    n_init = bads.optim_state["eff_starting_points"]
    assert np.all(bads.function_logger.Y[:n_init] == 1e3)
    assert np.isfinite(result["fval"])


@pytest.mark.parametrize("refit_flag", [False, True])
def test_rebuild_with_equal_targets_keeps_output_scale_prior(refit_flag):
    """A rebuild of the local GP on targets with no spread keeps the centre
    of the previous prior of the output scale, where MATLAB's gpdefBads.m
    takes log(0)."""
    bads, gp = _initialized_bads()
    previous = gp.get_priors()["covariance_log_outputscale"]
    logger = bads.function_logger
    logger.Y[logger.X_flag] = 7.0
    gp, _ = local_gp_fitting(
        gp,
        bads.u,
        logger,
        bads.options,
        bads.optim_state,
        bads.iteration_history,
        refit_flag,
        rng=bads.rng,
    )
    assert np.all(gp.y == 7.0)
    kind, (mu, sigma) = gp.get_priors()["covariance_log_outputscale"]
    assert kind == "gaussian"
    assert mu.item() == previous[1][0].item()
    assert sigma.item() == 2.0
    assert np.all(np.isfinite(gp.predict(np.zeros((1, 2)))[0]))


def test_rebuild_output_scale_prior_from_sample_sd():
    """At a rebuild of the local GP, the prior of the output scale is
    centred at the log of the targets' SD normalized by N - 1, as MATLAB's
    `std` in gpdefBads.m, with the SD 2."""
    bads, gp = _initialized_bads()
    gp, _ = local_gp_fitting(
        gp,
        bads.u,
        bads.function_logger,
        bads.options,
        bads.optim_state,
        bads.iteration_history,
        False,
        rng=bads.rng,
    )
    assert gp.y.size > 1
    kind, (mu, sigma) = gp.get_priors()["covariance_log_outputscale"]
    assert kind == "gaussian"
    assert np.isclose(mu.item(), np.log(np.std(gp.y, ddof=1)), rtol=1e-12)
    assert sigma.item() == 2.0


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_rebuild_with_single_target_keeps_output_scale_prior():
    """A rebuild of the local GP on a single training point keeps the centre
    of the previous prior of the output scale, as targets with no spread
    do, with no warning from an SD normalized by N - 1 = 0."""
    bads, gp = _initialized_bads()
    previous = gp.get_priors()["covariance_log_outputscale"]
    logger = bads.function_logger
    logger.X_flag[1:] = False
    logger.X_max_idx = 0
    gp, _ = local_gp_fitting(
        gp,
        bads.u,
        logger,
        bads.options,
        bads.optim_state,
        bads.iteration_history,
        False,
        rng=bads.rng,
    )
    assert gp.y.size == 1
    kind, (mu, sigma) = gp.get_priors()["covariance_log_outputscale"]
    assert kind == "gaussian"
    assert mu.item() == previous[1][0].item()
    assert sigma.item() == 2.0


def test_refit_on_targets_below_initial_design_fits_mean_below_them():
    """The constant mean of the GP is unbounded, as in MATLAB's
    gpdefBads.m: a refit on local targets far below those of the initial
    design fits a mean below the lower bound that gpyreg recommends for the
    initial design, which bounded it all run."""
    bads, gp = _initialized_bads()
    logger = bads.function_logger
    X, Y = logger.X[logger.X_flag], logger.Y[logger.X_flag]
    hpd_X, hpd_y, _, _ = get_hpd(X, Y, bads.options["hpd_frac"])
    initial_lower = gp.mean.get_bounds_info(hpd_X, hpd_y)["LB"].item()
    logger.Y[logger.X_flag] -= 1e3
    gp, exit_flag = local_gp_fitting(
        gp,
        bads.u,
        logger,
        bads.options,
        bads.optim_state,
        bads.iteration_history,
        True,
        rng=bads.rng,
    )
    assert exit_flag == 1
    assert np.max(gp.y) < initial_lower
    mean = gp.get_hyperparameters()[0]["mean_const"].item()
    assert np.min(gp.y) < mean < initial_lower
    lower, upper = gp.get_bounds()["mean_const"]
    assert lower.item() == -np.inf and upper.item() == np.inf


def test_thin_feasible_region_runs():
    """A feasible region too thin for the initial design (`non_box_cons`
    |x1 - x2| <= 0.005) leaves the local GP one training point, whose
    targets have no spread: the rebuild keeps the previous prior of the
    output scale, and the run goes on."""
    D = 3
    bads = BADS(
        lambda x: float(np.sum((np.ravel(x) - 1.0) ** 2)),
        np.zeros(D),
        -5 * np.ones(D),
        5 * np.ones(D),
        -2 * np.ones(D),
        2 * np.ones(D),
        non_box_cons=lambda x: np.abs(x[:, 0] - x[:, 1]) > 0.005,
        options={"display": "off", "max_fun_evals": 100, "random_seed": 0},
    )
    result = bads.optimize()
    assert bads.optim_state["eff_starting_points"] == 1
    assert abs(result["x"][0] - result["x"][1]) <= 0.005
    assert result["fval"] < 1e-3


def test_rebuild_len_scale_is_mean_over_samples(monkeypatch):
    """With several hyperparameter samples, the GP length scale is MATLAB's
    sum of their length scales weighted by `hypweight` (gpupdate.m), with
    equal weights: their mean. (The refit returns one sample today.)"""
    import pybads.bads.gaussian_process_train as gpt_module

    def two_sample_fit(gp, x_train, y_train, s2_train, hyp_gp, *args, **kw):
        first = gp.hyperparameters_to_dict(hyp_gp)[-1]
        second = {name: value.copy() for name, value in first.items()}
        second["covariance_log_lengthscale"] += np.log(3.0)
        hyp = gp.hyperparameters_from_dict([first, second])
        gp.set_hyperparameters(hyp, compute_posterior=False)
        return gp, hyp, None, 1

    monkeypatch.setattr(gpt_module, "_robust_gp_fit_", two_sample_fit)
    bads, gp = _initialized_bads()
    gp, _ = local_gp_fitting(
        gp,
        bads.u,
        bads.function_logger,
        bads.options,
        bads.optim_state,
        bads.iteration_history,
        True,
        rng=bads.rng,
    )
    samples = gp.get_hyperparameters()
    assert len(samples) == 2
    first = np.exp(samples[0]["covariance_log_lengthscale"])
    np.testing.assert_allclose(
        gp.temporary_data["len_scale"], 2 * first, rtol=1e-12
    )


# --- _robust_gp_fit_ ------------------------------------------------------


@pytest.fixture(scope="module")
def refit_case():
    """The GP of the last iteration of a short deterministic run (D = 3, 69
    training points), with the run and the training options of a refit."""
    D = 3
    bads = BADS(
        lambda x: float(np.sum((np.ravel(x) - 0.3) ** 2 * [1.0, 4.0, 9.0])),
        np.array([[1.0, -1.0, 0.5]]),
        -5 * np.ones((1, D)),
        5 * np.ones((1, D)),
        -2 * np.ones((1, D)),
        2 * np.ones((1, D)),
        options={"display": "off", "max_fun_evals": 70, "random_seed": 5},
    )
    bads.optimize()
    gp = bads.iteration_history["gp"][bads.optim_state["iter"]]
    gp_train = _get_gp_training_options(
        bads.optim_state,
        bads.iteration_history,
        bads.options,
        gp.get_hyperparameters(as_array=True),
        0,
        bads.function_logger,
    )
    return bads, gp, gp_train


def _inject_fit_failures(monkeypatch, n_fail=np.inf):
    """Makes `GP.fit` raise `LinAlgError` at its first `n_fail` calls and
    then return its first start, and records what each call is given."""
    calls = []

    def fit(gp, X=None, y=None, s2=None, hyp0=None, options=None, rng=None):
        calls.append(
            {
                "X": X.copy(),
                "lower_bounds": gp.lower_bounds.copy(),
                "hyp0": np.atleast_2d(hyp0).copy(),
            }
        )
        if len(calls) <= n_fail:
            raise np.linalg.LinAlgError("injected failure")
        return np.atleast_2d(hyp0)[:1].copy(), None, None

    monkeypatch.setattr(gpr.GP, "fit", fit)
    return calls


def _robust_fit(case, hyp=None, **options):
    """`_robust_gp_fit_` on a copy of the GP of `case`, from `hyp` (the GP's
    hyperparameters by default), with `options` over the run's options."""
    bads, gp, gp_train = case
    gp = copy.deepcopy(gp)
    opts = copy.deepcopy(bads.options)
    for key, value in options.items():
        opts[key] = value
    if hyp is None:
        hyp = gp.get_hyperparameters(as_array=True)
    return _robust_gp_fit_(
        gp,
        gp.X,
        gp.y,
        gp.s2,
        hyp,
        gp_train,
        copy.deepcopy(bads.optim_state),
        opts,
        np.random.default_rng(1),
    )


def test_robust_fit_slice_sampler_samples_on_retry_data(
    monkeypatch, refit_case
):
    """With `use_slice_sampler`, the start of each retry is sampled on the
    data that the retry fits, from which the second failure removes
    points."""
    calls = _inject_fit_failures(monkeypatch, 2)
    sampled = []
    original = gaussian_process_train._get_samples_from_slice_sampler_

    def spy(gp, *args, **kwargs):
        sampled.append((gp.X.copy(), gp.y.copy()))
        return original(gp, *args, **kwargs)

    monkeypatch.setattr(
        gaussian_process_train, "_get_samples_from_slice_sampler_", spy
    )
    _robust_fit(refit_case, use_slice_sampler=True)
    assert len(calls) == 3 and len(sampled) == 2
    assert calls[2]["X"].shape[0] < calls[1]["X"].shape[0]
    _, gp, _ = refit_case
    for (X, y), call in zip(sampled, calls[1:]):
        assert X.shape == call["X"].shape
        assert np.array_equal(X, call["X"])
        rows = [np.flatnonzero(np.all(gp.X == x, axis=1))[0] for x in X]
        assert np.array_equal(y, gp.y[rows])


def test_robust_fit_every_try_failed_returns_best_start(
    monkeypatch, refit_case
):
    """When every try fails, the fit returns the best of its starts, taken
    into the bounds and ranked by the log posterior on the data at entry,
    with exit flag -1, as MATLAB's gpHyperOptimize.m; a start whose log
    posterior cannot be computed ranks last. `noise_nudge=[0, 0]` leaves
    the bounds of the noise in place over the ten tries."""
    _, gp, _ = refit_case
    fitted = gp.get_hyperparameters(as_array=True)
    low_output_scale = fitted.copy()
    low_output_scale[0, 3] -= 3.0
    above_bound = fitted.copy()
    above_bound[0, 0] = gp.upper_bounds[0] + 1.0
    starts = np.vstack((low_output_scale, above_bound))
    in_bounds = np.minimum(
        np.maximum(starts, gp.lower_bounds), gp.upper_bounds
    )
    assert in_bounds[1, 0] == gp.upper_bounds[0]
    assert gp.log_posterior(in_bounds[1]) > gp.log_posterior(in_bounds[0])

    nudge = np.array([0, 0])
    calls = _inject_fit_failures(monkeypatch)
    out, hyp, res, flag = _robust_fit(refit_case, starts, noise_nudge=nudge)
    assert len(calls) == 10
    assert flag == -1 and res is None
    assert np.array_equal(hyp, in_bounds[1:])
    assert np.array_equal(out.get_hyperparameters(as_array=True), hyp)

    original = gpr.GP.log_posterior

    def log_posterior(self, hyp, compute_grad=False):
        if np.array_equal(hyp, in_bounds[1]):
            raise np.linalg.LinAlgError("injected failure")
        return original(self, hyp, compute_grad)

    monkeypatch.setattr(gpr.GP, "log_posterior", log_posterior)
    _, hyp, _, flag = _robust_fit(refit_case, starts, noise_nudge=nudge)
    assert flag == -1
    assert np.array_equal(hyp, in_bounds[:1])


@pytest.mark.parametrize(
    "noise_nudge, bound_nudge",
    [(np.array([1, 0]), 0.0), (np.array([1]), 0.5)],
    ids=["default", "one_element"],
)
def test_robust_fit_noise_nudge(
    monkeypatch, refit_case, noise_nudge, bound_nudge
):
    """After each failure, the start of the noise is raised by the
    cumulative `noise_nudge[0]`, and its lower bound from the bound at entry
    by `noise_nudge[1]` per failure, as in MATLAB's gpHyperOptimize.m, which
    completes a one-element nudge with half of it for the bound. At the
    default [1, 0] the bound stays in place, and the fifth failure no longer
    raises the bound above the upper one."""
    _, gp, _ = refit_case
    i_noise = gp.covariance.hyperparameter_count(gp.D)
    lb0 = gp.lower_bounds[i_noise]
    # A start with the noise at its lower bound, from which each retry
    # starts, with the noise raised by its nudge
    hyp = gp.get_hyperparameters(as_array=True)
    hyp[0, i_noise] = lb0
    monkeypatch.setattr(
        gaussian_process_train,
        "_get_random_samples_from_priors_",
        lambda gp, rng=None: hyp.copy(),
    )
    calls = _inject_fit_failures(monkeypatch, 6)
    _robust_fit(refit_case, hyp, noise_nudge=noise_nudge)
    assert len(calls) == 7
    for k, call in enumerate(calls):
        assert np.isclose(call["lower_bounds"][i_noise], lb0 + k * bound_nudge)
        assert np.isclose(call["hyp0"][0, i_noise], lb0 + k)


def test_robust_fit_slice_sampler_starts_within_bounds(
    monkeypatch, refit_case
):
    """With `use_slice_sampler`, the start of each retry lies within the
    bounds, which the slice sampler requires ("The initial starting point
    X0 is outside the bounds"): the lower bound of the noise no longer
    rises faster than its start, which stopped the run at the third
    consecutive failure, and the start is taken into the bounds, as in
    MATLAB's gpHyperOptimize.m, so that its noise, raised at each failure,
    stays below the upper bound."""
    _, gp, _ = refit_case
    i_noise = gp.covariance.hyperparameter_count(gp.D)
    hyp = gp.get_hyperparameters(as_array=True)
    hyp[0, i_noise] = gp.upper_bounds[i_noise]
    calls = _inject_fit_failures(monkeypatch, 9)
    _robust_fit(refit_case, hyp, use_slice_sampler=True)
    assert len(calls) == 10
    for call in calls:
        assert np.all(call["hyp0"] >= call["lower_bounds"])
        assert np.all(call["hyp0"] <= gp.upper_bounds)


def test_robust_fit_removes_points_above_matlab_percentile(
    monkeypatch, refit_case
):
    """From the second failure, a retry removes the worse point of the
    closest pair and the points above the 95th percentile of the targets as
    MATLAB's prctile1 computes it (NumPy's "hazen"), and a fit that
    succeeds after retries has exit flag 1, as in MATLAB's
    gpHyperOptimize.m."""
    calls = _inject_fit_failures(monkeypatch, 2)
    _, _, _, flag = _robust_fit(refit_case)
    assert len(calls) == 3
    _, gp, _ = refit_case
    X, y = gp.X, gp.y.ravel()
    assert np.array_equal(calls[1]["X"], X)
    dist = cdist(X, X)
    dist[np.tril_indices(dist.shape[0])] = np.inf
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    removed = np.union1d(
        [i if y[i] > y[j] else j],
        np.flatnonzero(y > np.percentile(y, 95, method="hazen")),
    )
    assert calls[2]["X"].shape[0] == X.shape[0] - removed.size
    assert np.array_equal(calls[2]["X"], np.delete(X, removed, axis=0))
    assert flag == 1


def test_robust_fit_stops_below_D_points(monkeypatch):
    """The retries stop once fewer training points than dimensions remain,
    and the fit then returns its start, taken into the bounds, with exit
    flag -1, as when every try fails, as in MATLAB's gpHyperOptimize.m."""
    D = 3
    bads, gp = _initialized_bads(D)
    hyp = gp.get_hyperparameters(as_array=True)
    gp_train = _get_gp_training_options(
        bads.optim_state,
        bads.iteration_history,
        bads.options,
        hyp,
        0,
        bads.function_logger,
    )
    calls = _inject_fit_failures(monkeypatch)
    _, hyp_out, _, flag = _robust_gp_fit_(
        gp,
        gp.X,
        gp.y,
        gp.s2,
        hyp,
        gp_train,
        bads.optim_state,
        bads.options,
        np.random.default_rng(1),
    )
    # From the second failure, each retry removes one of the 5 points (none
    # lies above the percentile of 5 or 4), and 2 are fewer than D
    assert [call["X"].shape[0] for call in calls] == [5, 5, 4, 3]
    assert flag == -1
    in_bounds = np.minimum(np.maximum(hyp, gp.lower_bounds), gp.upper_bounds)
    assert np.array_equal(hyp_out, in_bounds)


# --- _get_random_samples_from_priors_ ------------------------------------


def _prior_draws(gp, n=4000):
    rng = np.random.default_rng(7)
    draws = np.vstack(
        [_get_random_samples_from_priors_(gp, rng) for _ in range(n)]
    )
    return gp.hyperparameters_to_dict(draws)


def test_prior_samples_follow_gaussian_priors(refit_case):
    """The draws of each block of hyperparameters have the mean and the
    standard deviation of its Gaussian prior, in the units of the
    hyperparameter (log units for a log hyperparameter), as in MATLAB's
    gppriorrnd.m."""
    _, gp, _ = refit_case
    priors = gp.get_priors()
    draws = _prior_draws(gp)
    for key, (kind, (mu, sigma)) in priors.items():
        assert kind == "gaussian"
        values = np.array([draw[key] for draw in draws])
        se = sigma / np.sqrt(values.shape[0])
        assert np.all(np.abs(values.mean(axis=0) - mu) < 4 * se), key
        assert np.allclose(values.std(axis=0), sigma, rtol=0.05), key


def test_prior_samples_keep_block_without_prior(refit_case):
    """A block of hyperparameters without a prior keeps its value, as in
    MATLAB's gppriorrnd.m."""
    _, gp, _ = refit_case
    gp = copy.deepcopy(gp)
    priors = gp.get_priors()
    priors["covariance_log_shape"] = None
    gp.set_priors(priors)
    current = gp.get_hyperparameters()[-1]["covariance_log_shape"]
    draws = _prior_draws(gp, 10)
    for draw in draws:
        assert np.array_equal(draw["covariance_log_shape"], current)


def _normal_quantiles(n):
    """`n` z-scores that a normality test accepts: the quantiles of the
    standard normal at the midpoints of `n` equal bins."""
    return norm.ppf((np.arange(1, n + 1) - 0.5) / n)


def _refit_verdict(z, func_count, last_fit):
    """The verdicts (refit, unreliable) of `_is_gp_refit_time_` at D = 2
    (refit period 10, `min_refit_time` 4) on statistics of the GP
    prediction whose z-scores are `z` (targets `z`, predicted with mean 0
    and SD 1), at `func_count` evaluations with the last refit at
    `last_fit`."""
    bads = _make_bads()
    bads._record_gp_refit_()
    for z_i in z:
        bads._save_gp_stats_(z_i, 0.0, 1.0)
    bads.function_logger.func_count = func_count
    bads.optim_state["lastfitgp"] = last_fit
    refit, unreliable = bads._is_gp_refit_time_(
        bads.options["normalpha_level"]
    )
    return bool(refit), bool(unreliable)


# The verdicts are those of MATLAB's IsRefitTime (bads.m) and gppredcheck.m
@pytest.mark.parametrize(
    "z, func_count, last_fit, verdict",
    [
        ([0.1], 30, 29, (False, False)),
        ([0.1], 30, 20, (False, False)),
        (_normal_quantiles(9), 60, 30, (False, False)),
        (_normal_quantiles(10), 60, 30, (True, False)),
    ],
    ids=["n=1", "n=1, refit allowed", "n=period-1", "n=period"],
)
def test_gp_refit_time_counts_statistics(z, func_count, last_fit, verdict):
    """The calibration test of the GP counts its statistics as MATLAB BADS
    does: one statistic is tested (with the chi-square test), not taken
    for none, and the periodic refit is due once the statistics number
    the refit period."""
    assert _refit_verdict(z, func_count, last_fit) == verdict


# The verdicts are those of MATLAB's IsRefitTime (bads.m) and gppredcheck.m;
# the bounds of the sum of squares are 3.9e-13 and 25.26 at n = 1, 1.0e-6
# and 29.02 at n = 2
@pytest.mark.parametrize(
    "z, func_count, last_fit, verdict",
    [
        ([4.0], 30, 29, (False, False)),
        ([3.0, 3.2], 30, 29, (False, False)),
        ([3.0, 3.2], 40, 20, (False, False)),
        ([6e-4, 6e-4], 30, 29, (False, True)),
        ([4.0, 4.0], 30, 29, (False, True)),
    ],
    ids=[
        "n=1, 16",
        "n=2, 19.24",
        "n=2, 19.24, refit allowed",
        "n=2, 7.2e-7",
        "n=2, 32",
    ],
)
def test_gp_refit_time_chi_square_bounds(z, func_count, last_fit, verdict):
    """Fewer than three statistics are tested by their sum of squares
    against the quantiles alpha/2 and 1 - alpha/2 of the chi-square
    distribution with n degrees of freedom, as in MATLAB's gppredcheck.m."""
    assert _refit_verdict(z, func_count, last_fit) == verdict


@pytest.mark.parametrize("level", [0, 2])
def test_gp_stats_hold_sd_of_observation(monkeypatch, level):
    """The statistics of the GP prediction that the calibration test reads
    hold, for each point that the search or the poll evaluates, the SD of
    the observation there, MATLAB's `ys` (acqLCB.m, from gppred.m): the
    latent SD with the noise of the GP's posterior added, its noise
    hyperparameter times the factor by which gpyreg raises the noise of a
    posterior that fails to factorize. At level 2 that noise is the GP's
    base noise, without the target's own SD, which is not known where the
    GP predicts (mygp.m calls likGaussHe.m without it)."""
    import pybads.bads.bads as bads_module

    rng = np.random.default_rng(1000)

    def fun(x):
        y = np.sum(np.atleast_2d(x) ** 2)
        if level == 0:
            return y
        sd = 2 + np.sqrt(y)
        return y + sd * rng.standard_normal(), sd

    predictions, stats = [], []
    original_acq = bads_module.acq_fcn_lcb
    original_save = BADS._save_gp_stats_

    def spy_acq(xi, func_count, gp, *args, **kwargs):
        out = original_acq(xi, func_count, gp, *args, **kwargs)
        predictions.append((xi, gp, out))
        return out

    def spy_save(self, fval, ymu, ys):
        # The point evaluated is the one of lowest LCB of the last batch
        xi, gp, (z, f_mu, _) = predictions[-1]
        index = np.argmin(z)
        _, f_s2 = gp.predict(xi[index : index + 1])
        noise_log_scale = gp.get_hyperparameters()[0]["noise_log_scale"]
        sn2_mult = gp.posteriors[0].sn2_mult or 1
        s2 = f_s2.item() + np.exp(2 * noise_log_scale.item()) * sn2_mult
        stats.append((ymu, f_mu[index].item(), ys, np.sqrt(s2)))
        return original_save(self, fval, ymu, ys)

    monkeypatch.setattr(bads_module, "acq_fcn_lcb", spy_acq)
    monkeypatch.setattr(BADS, "_save_gp_stats_", spy_save)
    options = {"display": "off", "max_fun_evals": 60, "random_seed": 0}
    if level == 2:
        options.update(uncertainty_handling=True, specify_target_noise=True)
    D = 3
    bads = BADS(
        fun,
        np.ones(D) * 4,
        -100 * np.ones(D),
        100 * np.ones(D),
        -8 * np.ones(D),
        12 * np.ones(D),
        options=options,
    )
    bads.optimize()
    assert bads.optim_state["uncertainty_handling_level"] == level
    ymu, f_mu, ys, expected = np.array(stats).T
    assert len(ys) > 0
    assert np.array_equal(ymu, f_mu)
    np.testing.assert_allclose(ys, expected, rtol=1e-10)


def test_poll_scale_follows_length_scales_when_unbounded():
    """On a problem unbounded in every variable, the poll scale of a refit
    follows the GP length scales, within the width of the plausible box,
    which takes the place of the infinite bounds (2 in normalized units), as
    in MATLAB's gpupdate.m."""
    D = 3
    bads = BADS(
        lambda x: float(np.sum((np.array([1, 5, 30]) * np.ravel(x)) ** 2)),
        np.array([1.0, -1.2, 0.8]),
        None,
        None,
        -2 * np.ones(D),
        2 * np.ones(D),
        options={"display": "off", "random_seed": 3},
    )
    gp, _, _, _ = bads._init_optimization_()
    gp, _ = local_gp_fitting(
        gp,
        bads.u,
        bads.function_logger,
        bads.options,
        bads.optim_state,
        bads.iteration_history,
        True,
        rng=bads.rng,
    )
    log_ls = gp.get_hyperparameters()[0]["covariance_log_lengthscale"]
    ll = np.exp(bads.options["gp_rescale_poll"] * (log_ls - np.mean(log_ls)))
    poll_scale = gp.temporary_data["poll_scale"]
    assert np.all(poll_scale > 0)
    np.testing.assert_allclose(
        poll_scale,
        np.clip(ll, bads.optim_state["search_mesh_size"], 2.0),
        rtol=1e-12,
    )
