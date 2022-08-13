import numpy as np
from numpy import random  as rnd

from gpyreg.gaussian_process import GP


def poll_mads_2n(dim_x, poll_scale, search_mesh_size, mesh_size):
    """
    POLLMADS2N Poll 2N random basis vectors (mesh-adaptive direct search).

    """    
    n_max = np.maximum(1, np.round(search_mesh_size / mesh_size))

    if n_max > 0:
        D = rnd.randint(1, n_max*2, size=(dim_x, dim_x)) - n_max
        D = np.tril(D, -1)
    else:
        D = np.zeros((dim_x), dtype='float')

    diag = n_max * 2 * (rnd.randint(1, 3, dim_x) - 1.5)
    D = D + np.eye(dim_x) * diag

    # Random permutation of rows and columns and then transpose
    D = np.transpose(rnd.permutation(D))

    # Counteract subsequent multiplication by pollscale
    D = D/poll_scale
    B_new = np.vstack((D, -D))
    return B_new
