"""Compare poll_mads_2n with a transcription of pollMADS2N.m (74919c0):
support of the entries, diagonal, and the poll-point set after the scaling
of _poll_step_ (vv = B * mesh * poll_scale)."""
import gpyreg
import numpy as np

import pybads

print(pybads.__file__)
print(gpyreg.__file__)
from pybads.poll import poll_mads_2n


def matlab_pollMADS2N(nvars, pollscale, smesh, mesh, rng):
    nmax = max(1, np.round(smesh / mesh))  # MATLAB round: half away from 0
    nmax = max(1, np.floor(smesh / mesh + 0.5))
    # randi(nmax*2-1, nvars): integers 1..2nmax-1
    D = np.tril(
        rng.integers(1, 2 * nmax, size=(nvars, nvars), endpoint=False) - nmax,
        -1,
    )
    D = D + np.diag(nmax * 2 * (rng.integers(1, 3, nvars) - 1.5))
    D = D[rng.permutation(nvars)][:, rng.permutation(nvars)].T
    D = D / pollscale
    return np.vstack((D, -D)), nmax


rng = np.random.default_rng(1)
for D, smesh, mesh in [
    (3, 2.0**-10, 1.0),
    (4, 1.0, 0.25),
    (5, 1.0, 2.0**-3),
    (2, 3.0, 1.0),
]:
    ps = np.exp(rng.normal(size=D))
    vals_py, vals_m = set(), set()
    diag_py, diag_m = set(), set()
    sets_equal = True
    for rep in range(3000):
        Bp = poll_mads_2n(D, ps, smesh, mesh, rng=np.random.default_rng(rep))
        Bm, nmax = matlab_pollMADS2N(
            D, ps, smesh, mesh, np.random.default_rng(rep)
        )
        L = Bp[:D] * ps
        vals_py |= set(np.round(L).ravel().tolist())
        Lm = Bm[:D] * ps
        vals_m |= set(np.round(Lm).ravel().tolist())
        # poll points relative to u (vv), as in _poll_step_
        vv = Bp * mesh * ps
        assert np.allclose(vv, np.round(vv / mesh) * mesh)
    print(
        f"D={D} smesh/mesh={smesh/mesh:g}: nmax={nmax}; entry values py={sorted(vals_py)} matlab={sorted(vals_m)}"
    )
# n_max as float in rng.integers
print(
    "dtype check:",
    poll_mads_2n(3, np.ones(3), 2.0**-10, 1.0, rng=np.random.default_rng(0)),
)
# rows-only vs rows+cols permutation: the set of directions
Lr = np.random.default_rng(5)
L = np.tril(Lr.integers(-3, 4, (4, 4)), -1) + np.diag([4, -4, 4, 4])
p = Lr.permutation(4)
q = Lr.permutation(4)
A = L[p][:, q].T
Bm = L[p].T
print(
    "same direction set (rows):",
    sorted(map(tuple, A.tolist())) == sorted(map(tuple, Bm.tolist())),
)
