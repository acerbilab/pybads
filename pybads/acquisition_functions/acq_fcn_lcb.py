import numpy as np

import gpyreg as gpr

from pybads.function_logger.function_logger import FunctionLogger

def acq_fcn_lcb(xi, func_logger:FunctionLogger, gp:gpr.GP, sqrt_beta=None):
    """
        This method corresponds to the Lower Confidence Bound (LCB) acquisition function.

        Parameters
        ==========
        xi: np.ndarray
            current points
        func_logger: FunctionLogger
            function value
        gp: GP
            Gaussian process
        sqrt_beta: float
            LCB parameter
        
        Returns
        ==========
        z: lower confidence bound
        f_mu: GP prediction at xi
        f_s: GP variance
    """
    # Returns z, dz,ymu,ys,fmu,fs,*fpi*

    n = xi.shape[0]
    n_vars = xi.shape[1]
    t = func_logger.func_count + 1
    if sqrt_beta is None:
        delta, nu = 0.1, 0.2
        sqrt_beta = np.sqrt(nu*2*np.log(n_vars * t**2 * np.pi**2 / (6 * delta) ))
    elif callable(sqrt_beta):
        sqrt_beta = sqrt_beta(t, n_vars)
    elif ~np.isfinite(sqrt_beta) or sqrt_beta.size > 1:
        raise ValueError("acq_lcb: The SQRTBETAT parameter of the acquisition \
            function needs to be a scalar or a function handle/name to an annealing schedule.")

    f_mu, f_s2 = gp.predict(xi)
    f_s = np.sqrt(f_s2)

    # Lower confidence bound
    z = f_mu - sqrt_beta * f_s


    return z, f_mu, f_s