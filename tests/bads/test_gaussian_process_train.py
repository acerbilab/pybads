

import gpyreg as gpr
import numpy as np
from pybads.bads.bads import BADS
import pytest

from pybads.bads.gaussian_process_train import (
    init_and_train_gp,
    _cov_identifier_to_covariance_function,
    _estimate_noise,
    _get_gp_training_options,
    _get_hyp_cov,
    _get_training_data,
    _meanfun_name_to_mean_function,
)
from scipy.stats import norm

def test_estimate_noise():
    # Back-up random number generator so as not to affect other tests
    # that might want to use different random numbers each run.
    state = np.random.get_state()
    np.random.seed(1234)

    N = 31
    D = 1
    X = -5 + np.random.rand(N, 1) * 10
    s2 = 0.05 * np.exp(0.5 * X)
    y = np.sin(X) + np.sqrt(s2) * norm.ppf(np.random.random_sample(X.shape))
    y[y < 0] = -np.abs(3 * y[y < 0]) ** 2

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(degree=3),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True
        ),
    )

    hyp = np.array([[-2.5, 1.7, -7.5, 0.3, 2.6, 1.2]])
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)

    noise_estimate = _estimate_noise(gp)

    np.random.set_state(state)

    # Value taken from MATLAB which only applies for this exact setup.
    # Change any part of the test and it will not apply.
    assert np.isclose(noise_estimate, 0.106582207806606)


def test_get_training_data_no_noise():
    D = 3
    f = lambda x: np.sum(x + 2, axis=1)
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D)) * 1

    bads = BADS(f, x0, None, None, plb, pub)

    # Make sure we get nothing out before data has not been added.
    X_train, y_train, s2_train, t_train = _get_training_data(
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
        bads.function_logger.fun_evaltime[sample_idx] = 1e-5

    # Then make sure we get that data back.
    X_train, y_train, s2_train, t_train = _get_training_data(
        bads.function_logger
    )

    assert np.all(X_train == Xs)
    assert np.all(y_train.flatten() == ys)
    assert s2_train is None
    assert np.all(t_train == 1e-5)


def test_get_training_data_noise():
    D = 3
    f = lambda x: np.sum(x + 2, axis=1)
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D)) * 1
    user_options = {"specifytargetnoise": True}

    bads = BADS(f, x0, None, None, plb, pub, user_options)

    # Make sure we get nothing out before data has not been added.
    X_train, y_train, s2_train, t_train = _get_training_data(
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
    ys = f(Xs)

    # Add dummy training data explicitly since function_logger
    # has a parameter transformer which makes everything hard.
    for sample_idx in range(sample_count):
        bads.function_logger.X_flag[sample_idx] = True
        bads.function_logger.X[sample_idx] = Xs[sample_idx]
        bads.function_logger.Y[sample_idx] = ys[sample_idx]
        bads.function_logger.S[sample_idx] = 1
        bads.function_logger.fun_evaltime[sample_idx] = 1e-5

    # Then make sure we get that data back.
    X_train, y_train, s2_train, t_train = _get_training_data(
        bads.function_logger
    )

    assert np.all(X_train == Xs)
    assert np.all(y_train.flatten() == ys)
    assert np.all(s2_train == 1)
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


def test_get_hyp_cov():
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    f = lambda x: np.sum(x + 2)
    bads = BADS(f, x0, lb, ub, plb, pub)
    hyp_dict = {"run_cov": 42}

    res1 = _get_hyp_cov(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict
    )

    assert res1 is None

    bads.optim_state["iter"] = 1
    bads.options["weightedhypcov"] = False
    res2 = _get_hyp_cov(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict
    )

    assert res2 == 42

    # TODO: figure out some sort of a set-up for testing this.
    #       currently I don't have reference values
    #       maybe something like checking whether the returned thing is
    #       a covariance matrix?
    # bads.options["weightedhypcov"] = True
    # res3 = _get_hyp_cov(bads.optim_state, bads.iteration_history,
    #                       bads.options, hyp_dict)


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
    bads.optim_state["n_eff"] = 10
    bads.optim_state["iter"] = 1
    bads.optim_state["recompute_var_post"] = True
    bads.options["weightedhypcov"] = False
    bads.iteration_history.record("rindex", 5, 0)

    res1 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
    )
    assert res1["sampler"] == "slicesample"

    bads.options["gphypsampler"] = "npv"
    res2 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
    )
    assert res2["sampler"] == "npv"

    bads.options["gphypsampler"] = "mala"
    bads.optim_state["gpmala_stepsize"] = 10
    res3 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
    )
    assert res3["sampler"] == "mala"
    assert res3["step_size"] == 10

    bads.options["gphypsampler"] = "slicelite"
    res4 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
    )
    assert res4["sampler"] == "slicelite"

    bads.options["gphypsampler"] = "splitsample"
    res5 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
    )
    assert res5["sampler"] == "splitsample"

    bads.options["gphypsampler"] = "covsample"
    res6 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
    )
    assert res6["sampler"] == "covsample"

    # Test too large rindex for covsample
    bads.iteration_history.record("rindex", 50, 0)
    res7 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
    )
    assert res7["sampler"] == "slicesample"

    res8 = _get_gp_training_options(
        bads.optim_state,
        bads.iteration_history,
        bads.options,
        hyp_dict_none,
        8,
    )
    assert res8["sampler"] == "covsample"

    # Test too small n_eff laplace sampler
    bads.options["gphypsampler"] = "laplace"
    res9 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
    )
    assert res9["sampler"] == "slicesample"

    # Test enough n_eff laplace sampler
    bads.optim_state["n_eff"] = 50
    bads.options["gphypsampler"] = "laplace"
    res10 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
    )
    assert res10["sampler"] == "laplace"

    # Test sampler that does not exist.
    bads.options["gphypsampler"] = "does_not_exist"
    with pytest.raises(ValueError):
        res11 = _get_gp_training_options(
            bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
        )


def test_get_gp_training_options_opts_N():
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    f = lambda x: np.sum(x + 2)
    bads = BADS(f, x0, lb, ub, plb, pub)

    bads.optim_state["n_eff"] = 10
    bads.optim_state["iter"] = 2
    bads.iteration_history.record("rindex", 5, 1)
    bads.options["weightedhypcov"] = False
    hyp_dict = {"run_cov": np.eye(3)}
    hyp_dict_none = {"run_cov": None}
    bads.options["gpretrainthreshold"] = 10
    bads.optim_state["recompute_var_post"] = True

    res1 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 0
    )
    assert res1["opts_N"] == 2

    
    bads.options["gphypsampler"] = "slicelite"
    bads.optim_state["recompute_var_post"] = False
    res2 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 0
    )
    assert res2["opts_N"] == 1

    res3 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8
    )
    assert res3["opts_N"] == 0

    bads.options["gpretrainthreshold"] = 1
    res4 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 0
    )
    assert res4["opts_N"] == 2
