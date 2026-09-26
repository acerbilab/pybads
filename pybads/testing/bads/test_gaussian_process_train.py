import copy

import gpyreg as gpr
import numpy as np
import pytest
from scipy.stats import norm

from pybads import BADS
from pybads.bads import gaussian_process_train
from pybads.bads.gaussian_process_train import (
    _cov_identifier_to_covariance_function,
    _get_fevals_data,
    _get_gp_training_options,
    _meanfun_name_to_mean_function,
    _robust_gp_fit_,
    add_and_update_gp,
    init_and_train_gp,
    local_gp_fitting,
)


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
