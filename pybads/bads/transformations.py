from cmath import inf
from this import d
import numpy as np

def hypercube_trans(D, lower_bounds, upper_bounds, plausible_lower_bounds=None, plausible_upper_bounds=None, apply_log_t = None) :
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
    # TODO: add update of transformation variables

    if (lower_bounds is None and plausible_lower_bounds is None) or (upper_bounds is None and plausible_upper_bounds is None):
        raise ValueError("""hypercube_trans: At least one among LB, PLB and one among UB, PUB needs to be nonempty.""")

    
    plb = lower_bounds.copy() if (plausible_lower_bounds is None) else plausible_lower_bounds.copy()
    pub = upper_bounds.copy() if (plausible_upper_bounds is None) else plausible_upper_bounds.copy()

    lb =  lower_bounds.copy() if lower_bounds is not None else -np.inf
    ub =  upper_bounds.copy() if upper_bounds is not None else np.inf

    if np.isscalar(lb): lb = lb * np.ones(1, D)
    if np.isscalar(ub): ub = ub * np.ones(1, D)
    if np.isscalar(plb): plb = plb * np.ones(1, D)
    if np.isscalar(pub): pub = pub * np.ones(1, D)

    # Check finiteness of plausible range
    assert np.all(np.isfinite(np.concatenate([plb, pub]))), 'Plausible interval ranges PLB and PUB need to be finite.'

    # Check that the order of bounds is respected
    assert np.all(lb <= plb and plb < pub and pub <= ub), 'Interval bounds needs to respect the order LB <= PLB < PUB <= UB for all coordinates.'

    # Nonlinear log transform
    if apply_log_t is None:
        apply_log_t = np.full((1, D), np.NaN)
    elif np.isscalar(apply_log_t):
        apply_log_t = apply_log_t * np.ones((1, D))

    # A variable is converted to log scale if all bounds are positive and 
    # the plausible range spans at least one order of magnitude
    for i in np.find(np.isnan(apply_log_t)):
        apply_log_t[i] = np.all(np.concatenate([lb[i], ub[i], plb[i], pub[i]]) > 0) & (pub[i]/plb[i] >= 10)       
    apply_log_t = apply_log_t.astype(bool)


    lb[apply_log_t] = np.log(lb[apply_log_t])
    ub[apply_log_t] = np.log(ub[apply_log_t])
    plb[apply_log_t] = np.log(plb[apply_log_t])
    pub[apply_log_t] = np.log(pub[apply_log_t])

    #TODO: if returning a class object (trinfo), save also non-transformed variables
    mu = 0.5 * (plb + pub)
    gamma = 0.5 * (pub - plb)

    z = lambda x: maskindex((x - mu)/gamma, ~ apply_log_t)
    zlog = lambda x: maskindex((np.log(np.abs(x) + (x == 0)) - mu)/gamma, apply_log_t)

    apply_log_t_sum = np.sum(apply_log_t)
    if apply_log_t_sum == 0:
        g = lambda x: z(x)
        ginv = lambda y: gamma * y + mu

    elif apply_log_t_sum == D:
        g = lambda x: zlog(x)
        ginv = lambda y: min(np.finfo(np.float64).max, np.exp(gamma * y + mu) )
    else:
        g = lambda x: z(x) + zlog(x)
        ginv = lambda y: maskindex(gamma * y + mu, ~apply_log_t) \
                    + maskindex(min(np.finfo(np.float64).max, np.exp(gamma * y + mu)), apply_log_t)

    #check that the transform works correctly in the range
    lbtest = lb.copy()
    eps = np.spacing(1.0)
    lbtest[~np.isfinite(lb)] = -1/np.sqrt(eps)
    
    ubtest = ub.copy()
    ubtest[~np.isfinite(ub)] = 1/np.sqrt(eps)
    ubtest[~np.isfinite(ub) and apply_log_t] = 1e6

    numeps = 1e-6 #accepted numerical error
    tests = np.zeros(4)
    tests[0] = all(abs(ginv(g(lbtest)) - lbtest) < numeps)
    tests[1] = all(abs(ginv(g(ubtest)) - ubtest) < numeps)
    tests[2] = all(abs(ginv(g(plb)) - plb) < numeps)
    tests[3] = all(abs(ginv(g(pub)) - pub) < numeps)
    assert all(tests), 'Cannot invert the transform to obtain the identity at the provided boundaries.'

    return (g(lb), g(ub), g(plb), g(pub))


    
def maskindex (vector, bool_index):
    """
        Mask non-indexed elements in vector
    """
    result = vector.copy()
    result[~bool_index] = 0
    return result