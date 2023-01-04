import numpy as np
from gpyreg.gaussian_process import GP
from numpy import random as rnd


def poll_mads_2n(dim_x, poll_scale, search_mesh_size, mesh_size):
    """
    Practical Implementation of the MADS POLL method -- LTMADS [1].
    
    It retrieves random basis vectors which are dense refining directions in the hypertangent cone. The methods is the core method of the MADS framework for the theoretical convergence guarantees towards the Clarke's local stationary point.
    
    References
        ----------
        [1] Audet, Charles, Kwassi Joseph Dzahini, Michael Kokkolaras, and Sébastien Le Digabel. ‘Stochastic Mesh Adaptive Direct Search for Blackbox Optimization Using Probabilistic Estimates’. Computational Optimization and Applications 79, no. 1 (May 2021): 1–34. https://doi.org/10.1007/s10589-020-00249-0.
    """
    n_max = np.maximum(1, np.round(search_mesh_size / mesh_size))

    if n_max > 0:
        D = rnd.randint(1, n_max * 2, size=(dim_x, dim_x)) - n_max
        D = np.tril(D, -1)
    else:
        D = np.zeros((dim_x), dtype="float")

    diag = n_max * 2 * (rnd.randint(1, 3, dim_x) - 1.5)
    D = D + np.eye(dim_x) * diag

    # Random permutation of rows and columns and then transpose
    D = np.transpose(rnd.permutation(D))

    # Counteract subsequent multiplication by pollscale
    D = D / poll_scale
    B_new = np.vstack((D, -D))
    return B_new
