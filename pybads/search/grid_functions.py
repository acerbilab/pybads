from matplotlib.pyplot import axis
import numpy as np
from pybads.bads.variables_transformer import VariableTransformer
from scipy.spatial.distance import cdist
#from scipy.spatial.distance import pdist

def force_to_grid(x, search_mesh_size, tol = None):

    if tol is None: 
        tol = search_mesh_size

    return tol*np.round(x/tol)


def grid_units(x, var_trans : VariableTransformer = None, x0 = None, scale=None):
    '''
    GRIDUNITS Convert vector(s) coordinates to grid-normalized units
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

def get_grid_search_neighbors(function_logger, u, gp, options, optim_state):
    # get the training set by retrieving the sorted NEAREST neighbors from u
                    
    U_max_idx = function_logger.X_max_idx
    U = function_logger.X[0:U_max_idx]
    Y = function_logger.Y[0:U_max_idx]

    if 'S' in optim_state["S"]:
        S = optim_state["S"][0:U_max_idx]
    
    dist = udist(U, u, gp.temporary_data["lenscale"],
        optim_state["lb"], optim_state["ub"], optim_state["scale"],
            optim_state["periodicvars"])
    dist = np.minimum(dist, axis=1) 
    sort_idx = np.argsort(dist) # Ascending sort

    # Keep only points within a certain (rescale) radius from target
    radius = options["gpradius"] * gp.temporary_data["effective_radius"]
    ntrain = np.minimum(options["ndata"], np.sum(dist<=radius**2))

    # Minimum number of point to keep
    ntrain = np.max([options["minndata"], options["ndata"] - options["bufferndata"], ntrain])

    # Up to the maximum number of available points
    ntrain = np.minimum(ntrain, function_logger.X_max_idx)
    
    # Take points closest to reference points
    result = (U[sort_idx[0:ntrain+1]], Y[sort_idx[0:ntrain+1]], None)
    if 'S' in optim_state["S"]:
        result[2] = S[0:ntrain+1, :]
    return result

def udist(U, u2, lenscale, lb, ub, bound_scale, periodic_vars):
    '''

    Parameters
    --------

    periodic_vars: bool array
    '''
    idx_periods = np.nonzero(periodic_vars)
    if len(idx_periods) > 0:
        # Can be improved by using cdist, but we have to be careful with the scaling and
        rows = u2.shape[0]
        if len(u2.shape) > 1:
            rows = np.maximum(rows, u2.shape[0])
        diff = U[:, np.newaxis] - u2[np.newaxis, :]
        diff = diff.reshape((rows, U.shape[1]))

        w_s = (ub - lb) / bound_scale #scaled width (w_s)
        diff[idx_periods] = np.minimum(np.abs(diff[idx_periods]),
                                w_s - np.abs(diff)[idx_periods])
        diff = (diff/lenscale)**2
        return np.sum(diff, axis=1)

    else:
        dist = cdist(U/lenscale, u2/lenscale)
        return dist**2

