import numpy as np
from scipy.stats.qmc import Sobol


def init_sobol(u0=np.ndarray, lb=np.ndarray, ub=np.ndarray, plb=np.ndarray, pub=np.ndarray, fun_eval_start=int):
    """
    Initialize the Sobol sequence.
    This method relies on the scipy.stats.qmc.Sobol class for generating the Sobol sequence (Roy et. al 2023).
    You can find more information about the Sobol sequence in the documentation of the Sobol class.

    Roy et al., (2023). Quasi-Monte Carlo Methods in Python. Journal of Open Source Software, 8(84), 5309, https://doi.org/10.21105/joss.05309

    Parameters
    ----------
    u0 : array_like
        Initial point.
    lb : array_like
        Lower bounds.
    ub : array_like
        Upper bounds.
    plb : array_like
        Lower bounds for the parameters.
    pub : array_like
        Upper bounds for the parameters.
    fun_eval_start : int
        Number of initial function evaluations.

    Returns
    -------
    u_init : array_like
        Initial points.
    n_samples : int
        Number of samples used for the initialization.
    """

    max_seed = 997
    if np.all(np.isfinite(u0)):
        # Seed depends on u0
        str_seed = u0[0 : np.minimum(11, len(u0))].astype(np.uint64)
        if str_seed.ndim == 1:
            str_seed = np.array2string(str_seed)[1:-1]
        else:
            str_seed = np.array2string(str_seed)[2:-2]
        str_seed = np.array([ord(ch) for ch in str_seed])
        seed = np.prod(str_seed)
        seed = np.mod(seed, max_seed) + 1
    else:
        seed = np.random.randint(1, high=max_seed + 1)

    # Sobol’ sequences are a quadrature rule and they lose their balance properties
    # if one uses a sample size that is not a power of 2, or skips the first point,
    # or thins the sequence (Art B. Owen, “On dropping the first Sobol’ point.” arXiv:2008.08051, 2020.).
    sobol_sampler = Sobol(u0.size, seed=seed)

    # n_samples = fun_eval_start
    # samples = sobol_sampler.random(n_samples)
    n_samples = int(np.ceil(np.log2(fun_eval_start)))
    if 2**n_samples == u0.size:
        n_samples += 1
    samples = sobol_sampler.random_base2(n_samples)

    u_init = plb + samples * (pub - plb)

    return u_init, n_samples
