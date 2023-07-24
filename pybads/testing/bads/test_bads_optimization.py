import gpyreg as gpr
import numpy as np
import pytest
from scipy.stats import norm

from pybads.bads import BADS


def get_test_opt_conf(D=3):
    x0 = np.ones((1, D)) * 4
    LB = -100*np.ones(D)                # Lower bound
    UB = 100*np.ones(D)                 # Upper bound
    PLB = -8*np.ones(D)                 # Plausible lower bound
    PUB = 12*np.ones(D)                 # Plausible upper bound
    tol_errs = np.array([0.1, 0.1, 1, 1.])
    return D, x0, LB, UB, PLB, PUB, tol_errs

def run_bads(fun, x0, LB, UB, PLB, PUB, tol_errs, f_min,
            oracle_fun=None, non_box_cons=None, uncertainty_handling=False,
            assert_flag=False, max_fun_evals=None):
    options = {}
    options["display"] = "full"#debug_flag = True

    if uncertainty_handling > 0:
        options["uncertainty_handling"] = True
        options["max_fun_evals"] = 200 if max_fun_evals is None  else max_fun_evals
        options["specify_target_noise"] = True
        if uncertainty_handling > 1:
            options["specify_target_noise"] = True
    else:
        options["max_fun_evals"] = 100 if max_fun_evals is None else max_fun_evals

    optimize_result = BADS(fun=fun, x0=x0, lower_bounds=LB,
                                    upper_bounds=UB, plausible_lower_bounds=PLB,
                                    plausible_upper_bounds=PUB, non_box_cons=non_box_cons,
                                    options=options).optimize()
    x = optimize_result['x']
    fval = optimize_result['fval']

    if oracle_fun is None:
        #print(f"Final value: {fval:.3f} (true value: {f_min}), with {optimize_result['func_count']} fun evals.")
        err = np.abs(fval - f_min)
    else:
        fval_true = oracle_fun(x)
        #print(f"Final value (not-noisy): {fval_true:.3f} (true value: {f_min}) with {optimize_result['func_count']} fun evals.")
        err = np.abs(fval_true - f_min)
    if assert_flag:
        assert np.any(err < tol_errs), f"Error {err} is not smaller than tolerance {tol_errs} when optimizing {fun.__name__}."
    
    return optimize_result, err
    
def test_ellipsoid_opt():
    D, x0, LB, UB, PLB, PUB, tol_errs = get_test_opt_conf()
    fun = lambda x: np.sum((np.atleast_2d(x) / np.arange(1, len(x) + 1) ** 2) ** 2)
    run_bads(fun, x0, LB, UB, PLB, PUB, tol_errs, f_min=0.0, assert_flag=True)

def test_1D_opt():
    D, x0, LB, UB, PLB, PUB, tol_errs = get_test_opt_conf(D=1)
    fun = lambda x: np.sum((np.atleast_2d(x) / np.arange(1, len(x) + 1) ** 2) ** 2)
    run_bads(fun, x0, LB, UB, PLB, PUB, tol_errs, f_min=0.0, assert_flag=True)

def test_high_dim_opt():
    D, x0, LB, UB, PLB, PUB, tol_errs = get_test_opt_conf(D=60)
    fun = lambda x: np.sum((np.atleast_2d(x) / np.arange(1, len(x) + 1) ** 2) ** 2)
    run_bads(fun, x0, LB, UB, PLB, PUB, tol_errs, f_min=0.0, assert_flag=False, max_fun_evals=200)

def test_sphere_opt():
    D, x0, LB, UB, PLB, PUB, tol_errs = get_test_opt_conf()
    x0 = np.zeros((1, D))
    fun = lambda x: np.sum(np.atleast_2d(x)**2, axis=1)
    non_box_cons = lambda x: np.atleast_2d(x)[:, 0] + np.atleast_2d(x)[:, 1] >= np.sqrt(2)     # Non-bound constraints
    print(non_box_cons(x0))
    run_bads(fun, x0, LB, UB, PLB, PUB, tol_errs, f_min=1.0, non_box_cons=non_box_cons, assert_flag=True)
    
def test_noisy_sphere_opt():
    D, x0, LB, UB, PLB, PUB, tol_errs = get_test_opt_conf()
    fun = lambda x: np.sum(np.atleast_2d(x)**2, axis=1) + np.random.randn()             # Noisy objective function
    oracle_fun = lambda x: np.sum(np.atleast_2d(x)**2, axis=1)                          # True objective function
    run_bads(fun, x0, LB, UB, PLB, PUB, tol_errs, f_min=0.0, oracle_fun=oracle_fun, assert_flag=True)
    
def he_noisy_sphere(x):
    y = np.sum(np.atleast_2d(x)**2, axis=1)
    s = 2 + 1*np.sqrt(y)
    y = y + s*np.random.randn()
    return y, s

def test_he_noisy_sphere_opt():
    D, x0, LB, UB, PLB, PUB, tol_errs = get_test_opt_conf()
    fun = he_noisy_sphere
    oracle_fun = lambda x: np.sum(np.atleast_2d(x)**2, axis=1)                          # True objective function
    run_bads(fun, x0, LB, UB, PLB, PUB, tol_errs, f_min=0.0, oracle_fun=oracle_fun, uncertainty_handling=2, assert_flag=True)