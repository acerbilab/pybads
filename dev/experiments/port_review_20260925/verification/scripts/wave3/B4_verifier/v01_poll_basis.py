"""I-F1, I-F2: the port's poll basis against a transcription of MATLAB's
pollMADS2N; n_max over every reachable default mesh state; the cancellation
of poll_scale; and the directions in one default run."""
import numpy as np
from vhdr import box, ellipsoid, rosen

import pybads.bads.bads as bm
from pybads import BADS
from pybads.poll.poll_mads_2n import poll_mads_2n


def matlab_pollMADS2N(nvars, pollscale, searchmeshsize, meshsize, rng):
    # bads/poll/pollMADS2N.m, with MATLAB's randi(k) = integers in 1..k
    nmax = max(1, round(searchmeshsize / meshsize))
    if nmax > 0:
        D = np.tril(rng.integers(1, 2 * nmax, size=(nvars, nvars)) - nmax, -1)
    else:
        D = np.zeros((nvars, nvars))
    D = D + np.diag(nmax * 2 * (rng.integers(1, 3, nvars) - 1.5))
    D = D[rng.permutation(nvars)][:, rng.permutation(nvars)].T
    D = D / pollscale
    return np.vstack((D, -D)), nmax


# 1. n_max over the reachable default states: msi <= max_poll_grid_number = 0,
#    ssi = min(0, 2*msi - 10) (init and every failed poll), poll mesh 2**msi,
#    search mesh 2**ssi
nm = set()
for msi in range(0, -40, -1):
    ssi = min(0, 2 * msi - 10)
    nm.add(int(max(1, np.round(2.0**ssi / 2.0**msi))))
print("n_max over msi in [-39, 0]:", nm)

# 2. same structure as MATLAB for n_max > 1 (non-default mesh ratio)
rng = np.random.default_rng(0)
for nmax_target in (1, 3, 8):
    D = 4
    ps = np.exp(rng.normal(size=D))
    Bp = poll_mads_2n(
        D, ps, nmax_target * 0.25, 0.25, rng=np.random.default_rng(1)
    )
    Bm, nmax = matlab_pollMADS2N(
        D, ps, nmax_target * 0.25, 0.25, np.random.default_rng(1)
    )
    Ip, Im = np.round(Bp * ps).astype(int), np.round(Bm * ps).astype(int)

    # canonical form: set of directions as sorted rows
    def canon(I):
        return (
            np.sort(np.abs(I).max(axis=1)).tolist(),
            sorted(np.unique(np.abs(I)).tolist()),
            abs(round(np.linalg.det(I[:D]))),
        )

    print(
        f"n_max={nmax}: port (maxabs rows, abs entries, |det|)={canon(Ip)}; matlab={canon(Im)}"
    )

# 3. cancellation: vv = B_new * mesh * poll_scale is integer * mesh
ps = np.array([0.01, 3.0, 50.0])
B = poll_mads_2n(3, ps, 2.0**-12, 2.0**-1, rng=np.random.default_rng(3))
vv = B * 2.0**-1 * ps
print("vv / mesh =\n", vv / 2.0**-1)

# 4. one default run: record n_max and whether each poll set is +-mesh*e_i
rec = []
orig = bm.poll_mads_2n


def spy(D, poll_scale, sms, ms, rng=None):
    B = orig(D, poll_scale, sms, ms, rng=rng)
    I = B * poll_scale
    is_coord = bool(
        np.all(np.isclose(np.abs(I).sum(axis=1), 1))
        and np.all(np.isclose(np.abs(I).max(axis=1), 1))
    )
    rec.append(
        (
            int(max(1, np.round(sms / ms))),
            is_coord,
            float(np.ptp(np.log(poll_scale))),
        )
    )
    return B


bm.poll_mads_2n = spy
for name, f, D in (("rosen", rosen, 3), ("ellipsoid", ellipsoid, 4)):
    rec.clear()
    lb, ub, plb, pub = box(D)
    r = BADS(
        f,
        1.5 * np.ones(D),
        lb,
        ub,
        plb,
        pub,
        options=dict(random_seed=1, display="off", max_fun_evals=200),
    ).optimize()
    print(
        f"{name} D={D}: polls={len(rec)} n_max set={set(a for a,_,_ in rec)} "
        f"all coordinate={all(b for _,b,_ in rec)} max ptp(log poll_scale)={max(c for *_,c in rec):.2f} "
        f"fval={r['fval']:.3g} nfev={r['func_count']}"
    )
