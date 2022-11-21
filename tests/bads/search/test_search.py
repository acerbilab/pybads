
import numpy as np
import os
import sys
from pybads import function_logger
import pytest

from pybads import BADS
from pybads.search.es_search import ESSearchELL, ESSearchWM, ucov
from tests.bads.utils_test import load_options
from pybads.variable_transformer import VariableTransformer
from pybads.bads.gaussian_process_train import init_and_train_gp, get_grid_search_neighbors
from pybads.utils.iteration_history import IterationHistory
from pybads.function_logger import FunctionLogger, contraints_check
from pybads.function_examples import rosenbrocks_fcn
from pybads.search.search_hedge import ESSearchHedge

import gpyreg as gpr

def test_incumbent_constraint_check():
    
    # check duplicates
    D = 3
    U = np.random.normal(size=(10, D))
    U = np.unique(U, axis=0)
    lb = np.array([[-5]*D]) * 100
    ub = np.array([[5]*D]) * 100
    f = FunctionLogger(rosenbrocks_fcn, D, False, 0)
    for i in range(len(U)):
        y, y_sd, idx_y = f(U[i])

    input = np.random.normal(size=(5, D))
    U  = np.vstack((U, input))
    U_new = contraints_check(U, lb, ub, 1e-6, f, True)
    assert U_new.size != U.size
    print(f.func_count)
    assert U_new.shape[0] == U.shape[0] - f.func_count 
    print(U.shape)
    print(U_new.shape)

    # Check outliers and project them
    lb = np.array([[-0.5]*D])
    ub = np.array([[0.5]*D]) 
    U = np.vstack((U, np.array([[1]*D, [1]*D])))
    outbounds = (U < lb) | (U > ub)
    assert(np.any(outbounds))
    U_new = contraints_check(U, lb, ub, 1e-6, f, True)
    inbounds = np.all(U_new >= lb) & np.all(U_new <= ub)
    assert inbounds

def test_search():

    x0 = np.array([[0, 0, 0]]);        # Starting point
    lb = np.array([[-20, -20, -20]])     # Lower bounds
    ub = np.array([[20, 20, 20]])       # Upper bounds
    plb = np.array([[-5, -5, -5]])      # Plausible lower bounds
    pub = np.array([[5, 5, 5]])        # Plausible upper bounds
    D = 3

    options = load_options(D, "/home/gurjeet/Documents/UniPd/Helsinki/machine-human-intelligence/pybads/pybads/bads")

    bads = BADS(rosenbrocks_fcn, x0, lb, ub, plb, pub)
    bads.options['fun_eval_start'] = 0
    gp, Ns_gp, sn2hpd, hyp_dict = bads._init_optimization_()

    es_iter = bads.options['nsearchiter']
    mu = int(bads.options['nsearch'] / es_iter)
    lamb = mu
    search_es = ESSearchWM(mu, lamb, bads.options)
    us, z = search_es(bads.u, lb, ub, bads.function_logger, gp, bads.optim_state, True, None)

    assert us.size == 3 and (np.isscalar(z) or z.size == 1)
    assert np.all(gp.y >= z)

    search_es = ESSearchELL(mu, lamb, bads.options)
    us, z = search_es(bads.u, lb, ub, bads.function_logger, gp, bads.optim_state, True, None)
    assert us.size == 3 and (np.isscalar(z) or z.size == 1)
    assert np.all(gp.y >= z)

def test_search_selection_mask():

    D = 3
    mu = 1
    lamb = 2048
    options = load_options(D, "/home/gurjeet/Documents/UniPd/Helsinki/machine-human-intelligence/pybads/pybads/bads")
    search_es = ESSearchWM(mu, lamb, options)
    mask = search_es._get_selection_idx_mask_(mu, lamb)
    assert np.sum(mask) == 885072 
    assert np.min(mask + 1) == 1


def test_search_hedge(): 

    x0 = np.array([[0, 0, 0]]);        # Starting point
    lb = np.array([[-20, -20, -20]])     # Lower bounds
    ub = np.array([[20, 20, 20]])       # Upper bounds
    plb = np.array([[-5, -5, -5]])      # Plausible lower bounds
    pub = np.array([[5, 5, 5]])        # Plausible upper bounds
    D = 3

    bads = BADS(rosenbrocks_fcn, x0, lb, ub, plb, pub)
    bads.options['fun_eval_start'] = 0
    gp, Ns_gp, sn2hpd, hyp_dict = bads._init_optimization_()
    
    search_hedge = ESSearchHedge(bads.options['searchmethod'], bads.options)
    
    us, z = search_hedge(bads.u, lb, ub, bads.function_logger, gp, bads.optim_state)
    print(search_hedge.chosen_search_fun)
    assert us.size == 3 and (np.isscalar(z) or z.size == 1)
    assert np.all(gp.y >= z)

def test_u_cov():
    U = np.array([[0, 0, 0],[0.1172, 0.1328, 0.6641], [0, 0, 1], [0, 0, -1], [0.6172, -0.3672, 0.1641]])
    u0 = np.array([[0., 0., 0.]])
    ub = np.array([[4., 4., 4.]])
    lb = -ub
    w = np.array([0.4563, 0.2708, 0.1622, 0.0852, 0.0255])
    C = ucov(U, u0, w, ub, lb, 1)
    assert C.shape == (U.shape[1], U.shape[1])

def test_grid_search_neighbors():
    x0 = np.array([[0, 0]]);        # Starting point
    lb = np.array([[-20, -20]])     # Lower bounds
    ub = np.array([[20, 20]])       # Upper bounds
    plb = np.array([[-5, -5]])      # Plausible lower bounds
    pub = np.array([[5, 5]])        # Plausible upper bounds
    D = 2

    bads = BADS(rosenbrocks_fcn, x0, lb, ub, plb, pub)
    bads.options['fun_eval_start'] = 0
    gp, Ns_gp, sn2hpd, hyp_dict = bads._init_optimization_()
    gp.X = np.array([[0, 0], [-0.1055, 0.4570], [-0.3555, -0.7930]])
    f = FunctionLogger(rosenbrocks_fcn, D, False, 0)
    f.X = gp.X.copy()
    gp.y = np.array([1, 405.1637, 5.082e3])
    f.Y = gp.y.copy()

    f.X_max_idx = 3
    
    gp.temporary_data['len_scale'] = 1.
    bads.optim_state['scale'] = 1.
    bads.options['gpradius'] = 3
    
    result = get_grid_search_neighbors(f, np.array([[0, 0]]), gp, bads.options, bads.optim_state)[0]
    assert result[0, 0] == 0. and np.isclose(result[1, 0], -0.1055, 1e-3) and np.isclose(result[2, 0], -0.3555, 1e-3)