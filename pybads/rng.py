"""The random-number generator of a PyBADS run."""

import numpy as np


def get_rng(seed=None) -> np.random.Generator:
    """Return a NumPy random ``Generator``.

    Parameters
    ----------
    seed : None, int, array_like[int], SeedSequence, BitGenerator or \
Generator, optional
        Anything accepted by ``numpy.random.default_rng``. A ``Generator`` is
        returned unchanged, so that one generator can be shared. If ``None``
        (default), a new ``Generator`` is seeded from four ``uint32`` draws of
        NumPy's global random state, so that a preceding
        ``np.random.seed(...)`` makes the result reproducible; the global
        state advances by those draws and is not reseeded.

    Returns
    -------
    rng : np.random.Generator
    """
    if seed is None:
        seed = np.random.randint(0, 2**32, size=4, dtype=np.uint32)
    return np.random.default_rng(seed)
