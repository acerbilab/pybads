from multiprocessing.sharedctypes import Value
from typing import Callable
import numpy as np

from pybads.function_logger import FunctionLogger

def contraints_check(U:np.ndarray, lb:np.ndarray, ub:np.ndarray, tol_mesh, function_logger:FunctionLogger, proj=True, nonbondcons:Callable=None):
    """
        Return a new incumbent that satisfies the boundaries. Projection is applied in case of constraint violations
    """

    if proj:
        # Project vectors outside bounds on search mesh points closest to bounds
        U_new = np.maximum(np.minimum(U, ub), lb)
    else:
        idx = np.any(U > ub, axis=1)  | np.any( U < lb, axis=1)
        U_new = U[~idx]
    
    # Remove duplicate vectors and preserve the initial order
    _, idx_sort = np.unique(U_new, axis=0, return_index=True)
    U_new = U_new[np.sort(idx_sort), :].copy()

    # Remove previously evaluated vectors (within TolMesh)
    if U_new.size > 0:
        tol = tol_mesh / 2.
        u1 = np.round(U_new / tol)
        X_max_idx = function_logger.X_max_idx
        u2 = np.round(function_logger.X[:X_max_idx + 1] / tol)
        # Set difference using distance L1 matrix and checks if is 0,
        # since the setdiff(u1,u2,'rows') Matlab function is not implemented in numpy.
        l1 = np.abs(u1[:, np.newaxis, :] - u2).sum(axis=2)
        idx = np.where(np.isclose(l1, 0.))[0]
        mask = np.ones(len(u1)).astype(bool)
        mask[idx] = False
        
        U_new = U_new[mask]
    
    if nonbondcons is not None:
        if function_logger is None:
            raise ValueError("contraints_check: function_logger not passed, non bondcons requires it.")
        X = function_logger.variable_transformer.inverse_transf(U)
        C = nonbondcons(X)
        idx = C <=0
        U_new = U_new[idx]

    return U_new