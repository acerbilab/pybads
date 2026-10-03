"""How film.html receives a surface: a GP's mean and standard deviation at
the nodes of an n x n grid (rows along x2, columns along x1).

The mean is quantized to 16 bits between the lowest and the highest mean of
all the surfaces of a file, and the square root of the standard deviation to
8 bits up to the highest one. Each quantized surface is stored as
differences, which git compresses well: every value less its left neighbour
along its row, and the first column's less the value above it, as integers
that wrap around. A 16-bit surface is written as its low bytes, then its
high bytes. ``decodeStates`` in film.html sums them back.
"""

import base64

import numpy as np


def differences(q):
    """The integer grid q as differences, wrapping around."""
    d = q.copy()
    d[:, 1:] = q[:, 1:] - q[:, :-1]
    d[1:, 0] = q[1:, 0] - q[:-1, 0]
    return d


def pack(q):
    """The bytes of a quantized grid, as film.html reads them."""
    d = differences(q)
    if d.dtype == np.uint16:
        b = d.astype("<u2").view(np.uint8).reshape(-1, 2)
        return np.concatenate([b[:, 0], b[:, 1]])
    return d.ravel()


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def encode(mus, sds, n):
    """The surfaces (one per row of mus and sds, n * n values each, rows of
    the grid in order) as the grid's description and, for each surface, the
    strings of its mean and of its standard deviation."""
    mus = np.asarray(mus, float).reshape(-1, n, n)
    sds = np.asarray(sds, float).reshape(-1, n, n)
    z_lo, z_hi, sd_hi = float(mus.min()), float(mus.max()), float(sds.max())
    mu16 = np.round((mus - z_lo) / (z_hi - z_lo) * 65535).astype(np.uint16)
    sd8 = np.round(np.sqrt(sds / sd_hi) * 255).astype(np.uint8)
    grid = dict(n=n, z_lo=z_lo, z_hi=z_hi, sd_hi=sd_hi, code="differences")
    return grid, [(b64(pack(m)), b64(pack(s))) for m, s in zip(mu16, sd8)]
