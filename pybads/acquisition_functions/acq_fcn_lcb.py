import numpy as np

import gpyreg as gpr

from pybads.function_logger.function_logger import FunctionLogger

def acq_fcn_lcb(xi, func_logger:FunctionLogger, gp:gpr.GP, optim_state, sqrt_beta=None):
    # Returns z, dz,ymu,ys,fmu,fs,*fpi*

    n = xi.shape[0]
    n_vars = xi.shape[1]
    t = func_logger.func_count + 1
    if sqrt_beta is None or sqrt_beta.size == 0:
        delta, nu = 0.1, 0.2
        sqrt_beta_t = np.sqrt(nu*2*np.log(n_vars * t**2 * np.pi**2 / (6 * delta) ))
    elif callable(sqrt_beta):
        sqrt_beta_t = sqrt_beta(t, n_vars)
    elif ~np.isfinite(sqrt_beta_t) or sqrt_beta_t.size > 1:
        raise ValueError("acq_lcb: The SQRTBETAT parameter of the acquisition \
            function needs to be a scalar or a function handle/name to an annealing schedule.")

    f_mu, f_s = gp.predict(xi) # hyp weight average is already done by gpyreg.

    f_s = np.sqrt(f_s)

    # Lower confidence bound
    z = f_mu - sqrt_beta * f_s


    return z, f_mu, f_s