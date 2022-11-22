import numpy as np
from scipy.stats.qmc import Sobol


def init_sobol(u0, lb, ub, plb, pub, fun_eval_start):

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
