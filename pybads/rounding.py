"""Rounding to the nearest integer, as MATLAB rounds."""

import numpy as np


def round_half_away(x):
    """Round to the nearest integer, taking halves away from zero.

    This is MATLAB's ``round``, exactly: the integer part and the fraction
    that ``np.modf`` splits ``x`` into are exact, and the integer part
    moves away from zero when the fraction is at least one half in
    magnitude. ``np.round`` takes a half to the even integer, and
    ``np.sign(x) * np.floor(np.abs(x) + 0.5)`` takes the largest double
    below one half to 1.

    Parameters
    ----------
    x : array_like
        The values to round.

    Returns
    -------
    r : np.ndarray
        The rounded values, as floats.
    """
    frac, r = np.modf(x)
    return r + np.sign(frac) * (np.abs(frac) >= 0.5)
