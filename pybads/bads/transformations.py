from cmath import inf
import numpy as np

def hypercube_trans(D, lower_bounds, upper_bounds, plausible_lower_bounds=None, plausible_upper_bounds=None) :
    """
    Standardize variables via linear or nonlinear transformation.
    The standardized transform maps PLB and PUB to the hypercube [-1,1]^D.
    If PLB and/or PUB are empty, LB and/or UB are used instead. Note that
    at least one among LB, PLB and one among UB, PUB needs to be nonempty.

    Parameters
    ----------

    D : scalar
        dimension
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    plausible_lower_bounds: np.ndarray
    plausible_upper_bounds: np.ndarray

    
    """
    
    if (lower_bounds is None and plausible_lower_bounds is None) or (upper_bounds is None and upper_bounds is None):
        raise ValueError("""hypercube_trans: At least one among LB, PLB and one among UB, PUB needs to be nonempty.""")

    
    plb = lower_bounds.copy() if (plausible_lower_bounds is None) else plausible_lower_bounds.copy()
    pub = upper_bounds.copy() if (plausible_upper_bounds is None) else plausible_upper_bounds.copy()

    lb =  lower_bounds.copy() if lower_bounds is not None else -np.inf
    ub =  upper_bounds.copy() if upper_bounds is not None else np.inf

    # TODO: Convert scalar inputs to row vectors

    # TODO: Check finiteness of plausible range

    # TODO: Check that the order of bounds is respected

    # Nonlinear log transform
    logct = None #TODO: logct??
    if logct is None:
        logct = np.full((1, D), np.NaN)
    elif np.isscalar(logct):
        logct = logct * np.ones((1, D))

    
    