import numpy as np
from scipy.stats.qmc import Sobol

from pybads.rng import get_rng


def init_sobol(
    u0,
    lb,
    ub,
    plb,
    pub,
    fun_eval_start,
    rng=None,
):
    """
    Initialize the Sobol sequence.
    This method relies on the scipy.stats.qmc.Sobol class for generating the Sobol sequence (Roy et. al 2023).
    You can find more information about the Sobol sequence in the documentation of the Sobol class.

    Roy et al., (2023). Quasi-Monte Carlo Methods in Python. Journal of Open Source Software, 8(84), 5309, https://doi.org/10.21105/joss.05309

    The design has ``2**ceil(log2(fun_eval_start))`` points, twice as many
    when that number equals the dimension ``D``, scaled to the plausible box.

    Parameters
    ----------
    u0 : np.ndarray
        The starting point, of shape ``(D,)``.
    lb : np.ndarray
        The lower bounds (unused).
    ub : np.ndarray
        The upper bounds (unused).
    plb : np.ndarray
        The plausible lower bounds, which the design spans.
    pub : np.ndarray
        The plausible upper bounds, which the design spans.
    fun_eval_start : int
        The number of points asked of the design, which is rounded up as
        above.
    rng : numpy.random.Generator, optional
        Draws the seed of the Sobol sequence when ``u0`` is not all finite;
        otherwise the seed derives from the integer parts of the first 11
        coordinates of ``u0``. If ``None``, a
        generator is derived from NumPy's global random state
        (``pybads.rng.get_rng``).

    Returns
    -------
    u_init : np.ndarray
        The points of the design, of shape ``(n_samples, D)``.
    n_samples : int
        The number of points of the design.
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
        seed = get_rng(rng).integers(1, max_seed + 1)

    # Sobol’ sequences are a quadrature rule and they lose their balance properties
    # if one uses a sample size that is not a power of 2, or skips the first point,
    # or thins the sequence (Art B. Owen, “On dropping the first Sobol’ point.” arXiv:2008.08051, 2020.).
    sobol_sampler = Sobol(u0.size, seed=seed)

    # n_samples = fun_eval_start
    # samples = sobol_sampler.random(n_samples)
    m = int(np.ceil(np.log2(fun_eval_start)))
    if 2**m == u0.size:
        m += 1
    samples = sobol_sampler.random_base2(m)
    n_samples = samples.shape[0]

    u_init = plb + samples * (pub - plb)

    return u_init, n_samples
