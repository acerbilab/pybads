import gpyreg as gpr
import numpy as np


def acq_fcn_lcb(xi, func_count: int, gp: gpr.GP, sqrt_beta=None):
    """
    Lower Confidence Bound (LCB) acquisition function.
    It retrieves the point at the lower confidence bound of the GP surrogate model.

    Parameters
    ==========
    xi: np.ndarray
        Input points
    func_count: int
        Number of function evaluations
    gp: GP
        Gaussian process
    sqrt_beta: None, callable or float
        LCB parameter, the multiplier of the GP's SD. If ``None``, the
        schedule of Srinivas et al. (2010), with an empirical correction; a
        callable is called as ``sqrt_beta(t, n_vars)``, with
        ``t = func_count + 1``; otherwise a positive finite real number (a
        Python or NumPy scalar, or an array of one element).

    Returns
    ==========
    z: lower confidence bound
        Lower confidence bound at xi.
    f_mu: GP prediction at xi
        GP mean at xi.
    f_s: GP standard deviation
        GP standard deviation at xi.

    Raises
    ==========
    ValueError
        If ``sqrt_beta`` is none of the values above.
    """
    # Returns z, dz,ymu,ys,fmu,fs,*fpi*

    n = xi.shape[0]
    n_vars = xi.shape[1]
    t = func_count + 1
    if sqrt_beta is None:
        delta, nu = 0.1, 0.2
        sqrt_beta = np.sqrt(
            nu * 2 * np.log(n_vars * t**2 * np.pi**2 / (6 * delta))
        )
    elif callable(sqrt_beta):
        sqrt_beta = sqrt_beta(t, n_vars)
    else:
        sqrt_beta_array = np.asarray(sqrt_beta)
        if not (
            sqrt_beta_array.size == 1
            and sqrt_beta_array.dtype.kind in "iuf"
            and np.isfinite(sqrt_beta_array).item()
            and sqrt_beta_array.item() > 0
        ):
            raise ValueError(
                "acq_fcn_lcb: sqrt_beta needs to be None (the default "
                "schedule), a callable sqrt_beta(t, n_vars) or a positive "
                f"finite real number, not {sqrt_beta!r}."
            )
        sqrt_beta = sqrt_beta_array.item()

    f_mu, f_s2 = gp.predict(xi)
    f_s = np.sqrt(f_s2)

    # Lower confidence bound
    z = f_mu - sqrt_beta * f_s

    return z, f_mu, f_s
