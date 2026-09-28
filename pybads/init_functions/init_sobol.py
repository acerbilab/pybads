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
    This method relies on the scipy.stats.qmc.Sobol class for generating the
    Sobol sequence (Roy et. al 2023). You can find more information about the
    Sobol sequence in the documentation of the Sobol class.

    Roy et al., (2023). Quasi-Monte Carlo Methods in Python. Journal of Open
    Source Software, 8(84), 5309, https://doi.org/10.21105/joss.05309

    The design has ``2**ceil(log2(fun_eval_start))`` points, twice as many
    when that number equals the dimension ``D``, scaled to the plausible box.
    Its scrambling is seeded by one draw of ``rng``, so that the generator
    decides the design, whatever the starting point.

    Parameters
    ----------
    u0 : np.ndarray
        The starting point, of shape ``(D,)``, of which only the size is read.
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
        The generator that draws the seed of the scrambling. If ``None``, a
        generator is derived from NumPy's global random state
        (``pybads.rng.get_rng``).

    Returns
    -------
    u_init : np.ndarray
        The points of the design, of shape ``(n_samples, D)``.
    n_samples : int
        The number of points of the design.
    """

    # The seed of the scrambling is one draw of the run's generator, so that
    # the generator advances by one draw whatever scipy's Sobol draws from
    # the generator it seeds. MATLAB BADS derives a skip index into the
    # unscrambled sequence from the digits of u0 instead (initSobol.m:9-15),
    # so that its design follows from the start alone, with no random draw
    seed = get_rng(rng).integers(2**63)

    # Sobol’ sequences are a quadrature rule and they lose their balance
    # properties if one uses a sample size that is not a power of 2, or skips
    # the first point, or thins the sequence (Art B. Owen, “On dropping the
    # first Sobol’ point.” arXiv:2008.08051, 2020.).
    sobol_sampler = Sobol(u0.size, seed=seed)

    m = int(np.ceil(np.log2(fun_eval_start)))
    if 2**m == u0.size:
        m += 1
    samples = sobol_sampler.random_base2(m)
    n_samples = samples.shape[0]

    u_init = plb + samples * (pub - plb)

    return u_init, n_samples
