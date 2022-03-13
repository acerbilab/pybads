import numpy as np
from pybads.bads.variables_transformer import VariableTransformer

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
            u = var_trans.direct_transf(x)
        else:
            u = np.zeros(x.shape)
            for i in range(0, len(x)):
                u[i,:] = var_trans.direct_transf(x[i,:])

    else:
        u = (x-x0)/ scale
    return u

