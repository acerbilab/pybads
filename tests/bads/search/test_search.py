
import numpy as np
import os
import sys
import pytest

from pybads.bads.options import Options
from pybads.bads.variables_transformer import VariableTransformer
from pybads.bads.gaussian_process_train import train_gp
from pybads.utils.iteration_history import IterationHistory
from pybads.function_logger import FunctionLogger
from pybads.function_examples import rosenbrocks, rosenbrocks_single_sample
from pybads.search.search_hedge import SearchESHedge
from pybads.utils.constraints_check import contraints_check


import gpyreg as gpr

def test_incumbent_constraint_check():
    
    # check duplicates
    U = np.random.normal(size=(10, 2))
    U = np.unique(U, axis=0)
    lb = np.array([[-5, -5]]) * 100
    ub = np.array([[5, 5]]) * 100
    f = FunctionLogger(rosenbrocks_single_sample, 2, False, 0)
    for i in range(len(U)):
        y, y_sd, idx_y = f(U[i])

    input = np.random.normal(size=(5, 2))
    U  = np.vstack((U, input))
    U_new = contraints_check(U, lb, ub, 1e-6, f, True)
    assert U_new.size != U.size
    print(f.func_count)
    assert U_new.shape[0] == U.shape[0] - f.func_count 
    print(U.shape)
    print(U_new.shape)

    # Check outliers and project them
    lb = np.array([[-0.5, -0.5]])
    ub = np.array([[0.5, 0.5]]) 
    outbounds = (U < lb) | (U > ub)
    assert(np.any(outbounds))
    U_new = contraints_check(U, lb, ub, 1e-6, f, True)
    inbounds = np.all(U_new >= lb) & np.all(U_new <= ub)
    assert inbounds
    


test_incumbent_constraint_check()

def test_search():
    x0 = np.array([[0, 0]]);        # Starting point
    lb = np.array([[-20, -20]])     # Lower bounds
    ub = np.array([[20, 20]])       # Upper bounds
    plb = np.array([[-5, -5]])      # Plausible lower bounds
    pub = np.array([[5, 5]])        # Plausible upper bounds
    D = 2
    # load basic and advanced options and validate the names
    pybads_path = "/home/gurjeet/Documents/UniPd/Helsinki/machine-human-intelligence/pybads/pybads/bads"
    basic_path = pybads_path + "/option_configs/basic_bads_options.ini"
    options = Options(
        basic_path,
        evaluation_parameters={"D": D},
        user_options=None,
    )
    advanced_path = (
        pybads_path + "/option_configs/advanced_bads_options.ini"
    )
    options.load_options_file(
        advanced_path,
        evaluation_parameters={"D": D},
    )
    options.validate_option_names([basic_path, advanced_path])
    
    search_fcn = [('ES-wcm', 1), ('ES-ell', 1)]
    search = SearchESHedge(search_fcn, options)

    var_transf = VariableTransformer(D, lb, ub,
            plb, pub)
    function_logger = FunctionLogger(rosenbrocks_single_sample, 2, False, 0)

    iteration_history = IterationHistory(
            [
                "iter",
                "fval",
                "ymu",
                "ys"
                "lcbmax",
                "gp",
                "gp_last_reset_idx"
                "gp_hyp_full",
                "Ns_gp",
                "timer",
                "optim_state",
                "func_count",
                "n_eff",
                "logging_action",
            ]
        )
    hyp_dict = {}
    #gp, Ns_gp, sn2hpd, hyp_dict = train_gp(hyp_dict, optim_state, function_logger, iteration_history, options,
    #        plausible_lower_bounds, plausible_upper_bounds)

    #search(x0, lb, ub, function_logger, gp , optim_state))



