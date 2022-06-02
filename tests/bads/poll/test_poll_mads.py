
import numpy as np
import os
import sys
import pytest

from pybads.bads.bads import BADS
from pybads.search.es_search import SearchESELL, SearchESWM
from tests.bads.utils_test import load_options
from pybads.bads.variables_transformer import VariableTransformer
from pybads.bads.gaussian_process_train import train_gp
from pybads.utils.iteration_history import IterationHistory
from pybads.function_logger import FunctionLogger
from pybads.function_examples import rosenbrocks_fcn
from pybads.utils.constraints_check import contraints_check
from pybads.poll.poll_mads_2n import poll_mads_2n

import gpyreg as gpr

def poll_mads_2n_ones_test():
    
    poll_scale = np.ones((1, 3))
    search_mesh_size = 9.7656e-4
    mesh_size = 1.
    D = 3
    
    B = poll_mads_2n(D , poll_scale, search_mesh_size, mesh_size)
    assert B.shape[0] == 6 and B.shape[1]==D

def poll_mads_2n_test():
    
    poll_scale = np.array([[0.5133, 0.493, 3.9511]])
    search_mesh_size = 9.7656e-4
    mesh_size = 0.0312
    D = 3
    
    B = poll_mads_2n(D , poll_scale, search_mesh_size, mesh_size)
    assert B.shape[0] == 6 and B.shape[1]==D

poll_mads_2n_ones_test()
poll_mads_2n_test()
