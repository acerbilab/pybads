import gpyreg as gpr
import numpy as np
import pytest
from scipy.stats import norm

from pybads import BADS
from pybads.bads.gaussian_process_train import (
    _cov_identifier_to_covariance_function,
    _get_fevals_data,
    _get_gp_training_options,
    _meanfun_name_to_mean_function,
    init_and_train_gp,
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

    bads = BADS(f, x0, plausible_lower_bounds=plb, plausible_upper_bounds=pub, options=options)

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
       f_i, _ =  f(Xs[x_idx])
       ys.append(f_i)
    ys = np.array(ys)

    # Add dummy training data explicitly since function_logger
    # has a parameter transformer which makes everything hard.
    for sample_idx in range(sample_count):
        bads.function_logger.X_flag[sample_idx] = True
        bads.function_logger.X[sample_idx] = Xs[sample_idx]
        bads.function_logger.Y[sample_idx] = ys[sample_idx]
        bads.function_logger.S[sample_idx] = 1
        bads.function_logger.fun_eval_time[sample_idx] = 1e-5

    # Then make sure we get that data back.
    X_train, y_train, s2_train, t_train = _get_fevals_data(
        bads.function_logger
    )

    assert np.all(X_train == Xs)
    assert np.all(y_train.flatten() == ys.flatten())
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
    bads.optim_state['eff_starting_points'] = 10
    bads.optim_state["ntrain"] = 10
    bads.optim_state["iter"] = 1
    bads.options["weighted_hyp_cov"] = False

    res1 = _get_gp_training_options(
        bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 8, bads.function_logger
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

    bads.optim_state['eff_starting_points'] = 10
    bads.optim_state["ntrain"] = 10
    bads.optim_state["iter"] = 2
    bads.options["weighted_hyp_cov"] = False
    hyp_dict = {"run_cov": np.eye(3)}
    hyp_dict_none = {"run_cov": None}
    bads.options["gpretrainthreshold"] = 10

    res1 = _get_gp_training_options(
    bads.optim_state, bads.iteration_history, bads.options, hyp_dict, 0, bads.function_logger
    )
    assert res1["opts_N"] == 1
