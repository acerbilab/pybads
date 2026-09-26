"""K5 / I-F3 / C-F2: initial design against a small max_fun_evals.
MATLAB (transcribed from evalinitmesh.m:38-50, 93-104 and bads.m:441-442):
evaluations of the initialization = 1 + test + min(Ninit', MaxFunEvals - 1),
Ninit' = min(max(20, Ninit), MaxFunEvals) when noisy; nfs = min(nfs, MFE - count)."""
import common
import numpy as np

from pybads import BADS


class Count:
    def __init__(self, f):
        self.f = f
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return self.f(x)


def matlab_init(D, mfe, noisy, test=True, ninit=None, nfs=10):
    ninit = D if ninit is None else ninit
    count = 1 + (1 if test else 0)
    if mfe == 1:
        return count, None, None
    if noisy:
        ninit = min(max(20, ninit), mfe)
    n = min(ninit, mfe - 1)
    count += n
    if noisy:
        nfs = min(nfs, mfe - count)
        mfe2 = mfe - nfs
    else:
        mfe2 = mfe
        nfs = None
    return count, nfs, mfe2


def run(D, mfe, noisy, seed=0):
    nrng = np.random.default_rng(seed)
    base = (
        (lambda x: float(np.sum(np.ravel(x) ** 2) + nrng.standard_normal()))
        if noisy
        else (lambda x: float(np.sum(np.ravel(x) ** 2)))
    )
    f = Count(base)
    b = BADS(
        f,
        np.full(D, 0.5),
        np.full(D, -5.0),
        np.full(D, 5.0),
        np.full(D, -2.0),
        np.full(D, 2.0),
        options=dict(display="off", random_seed=seed, max_fun_evals=mfe),
    )
    r = b.optimize()
    m = matlab_init(D, mfe, noisy)
    print(
        f"D={D} {'noisy' if noisy else 'det  '} max_fun_evals={mfe:3d}: PyBADS target calls {f.n:3d}, "
        f"func_count {r['func_count']:3d}, design points {b.optim_state['eff_starting_points']-1:3d}, "
        f"nfs after {b.options['noise_final_samples']}, max_fun_evals after {b.options['max_fun_evals']}, "
        f"iterations {r['iterations']}, fsd {r['fsd']}, yval_vec {None if r['yval_vec'] is None else np.size(r['yval_vec'])} "
        f"| MATLAB init evals {m[0]}, nfs {m[1]}, MaxFunEvals after {m[2]}"
    )


for D, mfe in [(2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (5, 7), (5, 9)]:
    run(D, mfe, False)
for D, mfe in [(2, 10), (2, 25), (2, 34), (2, 38), (2, 40), (2, 44), (2, 50)]:
    run(D, mfe, True)
