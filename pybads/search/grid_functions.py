from matplotlib.pyplot import axis
import numpy as np
from pybads.bads.variables_transformer import VariableTransformer
from scipy.spatial.distance import cdist

from pybads.function_logger.function_logger import FunctionLogger

def force_to_grid(x, search_mesh_size, tol = None):

    if tol is None: 
        tol = search_mesh_size

    return tol*np.round(x/tol)


def grid_units(x, var_trans : VariableTransformer = None, x0 = None, scale=None):
    '''
    grid_units convert vector(s) coordinates to grid-normalized units
    '''
    if var_trans is not None:
        if len(x) == 1:
            u = var_trans(x)
        else:
            u = np.zeros(x.shape)
            for i in range(0, len(x)):
                u[i,:] = var_trans(x[i,:])

    else:
        u = (x-x0)/ scale
    return u


def get_grid_search_neighbors(function_logger: FunctionLogger, u, gp, options, optim_state):
    """ Retrieve a sorted matrix based on distance from the current incumbent `u`.
    Return
    ----------
        tuple : (U, Y, S)
            U (np.ndarray) : nearest neighbors from the incumbent `u`
            Y (np.ndarray) : predicted values at U
            S (np.ndarray) : estimated variance at U
    """
    # get the training set by retrieving the sorted NEAREST neighbors from u
    U_max_idx = function_logger.X_max_idx
    U = function_logger.X[0:U_max_idx+1].copy()
    Y = function_logger.Y[0:U_max_idx+1].copy()

    if function_logger.noise_flag:
        S = function_logger.S[0:U_max_idx+1]
    
    dist = udist(U, u, gp.temporary_data["len_scale"],
        optim_state["lb"], optim_state["ub"], optim_state['scale'],
            optim_state['periodic_vars'])
    if dist.ndim > 1:
        dist = np.min(dist, axis=1) 
    sort_idx = np.argsort(dist) # Ascending sort

    # Keep only points within a certain (rescale) radius from target
    radius = options["gpradius"] * gp.temporary_data["effective_radius"]
    ntrain = np.minimum(options["ndata"], np.sum(dist<=radius**2))

    # Minimum number of point to keep
    ntrain = np.max([options["minndata"], options["ndata"] - options["bufferndata"], ntrain])

    # Up to the maximum number of available points
    ntrain = np.minimum(ntrain, function_logger.X_max_idx)
    
    # Take points closest to reference points
    res_S = None
    if function_logger.noise_flag:
        res_S = function_logger.S[sort_idx[0:ntrain]]
    return (U[sort_idx[0:ntrain]], Y[sort_idx[0:ntrain]], res_S)


def udist(U, u2, len_scale, lb, ub, bound_scale, periodic_vars):
    '''

    Parameters
    --------

    periodic_vars: bool array
    '''
    idx_periods = np.nonzero(periodic_vars)[0]
    if len(idx_periods) > 0:
        # Can be improved by using cdist, but we have to be careful with the scaling and
        A = np.atleast_2d(U)
        B = np.atleast_2d(u2)
        A2 = np.sum(A**2, axis=-1)
        B2 = np.sum(B**2, axis=-1)
        diff = A2[:, None] - 2* A @ B.T + B2[None, :]

        w_s =   (ub - lb) / bound_scale #scaled width
        diff[idx_periods] = np.minimum(np.abs(diff[idx_periods]),
                                np.sum(np.atleast_2d(w_s) - np.abs(diff)[idx_periods], axis=-1)**2)
        
        diff = (diff/(len_scale)**2)
        return diff

    else:
        dist = cdist(np.atleast_2d(U)/len_scale, np.atleast_2d(u2)/len_scale)
        return dist**2

