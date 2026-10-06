import numpy as np

import pybads.bads.bads as bm
from pybads import BADS

D = 3


def sphere(x):
    return float(np.sum(np.atleast_2d(x) ** 2))


def rosen(x):
    x = np.ravel(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ell(x):
    x = np.ravel(x)
    return float(np.sum((10.0 ** np.arange(len(x))) * x**2))


def ridge(x):
    x = np.ravel(x)
    return float(10 * np.sum(np.abs(np.diff(x))) + abs(np.sum(x)))


of = bm.local_gp_fitting
os_ = BADS._search_step_
op = BADS._poll_step_
fits = []
steps = []


def fitting(*a, **k):
    fits.append(a[6])
    return of(*a, **k)


def step(orig, kind):
    def w(self, gp):
        u = self.u_best.copy()
        n = len(fits)
        out = orig(self, gp)
        steps.append((kind, not np.array_equal(u, self.u_best), fits[n:]))
        return out

    return w


bm.local_gp_fitting = fitting
BADS._search_step_ = step(os_, "search")
BADS._poll_step_ = step(op, "poll")


def count():
    am = au = 0
    pm = None
    prev = None
    nmoves = 0
    for kind, moved, sf in steps:
        if kind == "poll":
            pm = moved
            nmoves += moved
        elif pm is not None:
            first = prev[0] == "poll"
            if not first and not prev[1] and not any(sf):
                if pm:
                    am += 1
                else:
                    au += 1
        prev = (kind, moved)
    return am, au, nmoves


for name, f in [
    ("sphere", sphere),
    ("rosen", rosen),
    ("ell", ell),
    ("ridge", ridge),
]:
    for seed in [1, 2, 3]:
        for mfe in [100, 150]:
            fits.clear()
            steps.clear()
            b = BADS(
                f,
                np.ones(D) * 4,
                -100 * np.ones(D),
                100 * np.ones(D),
                -8 * np.ones(D),
                12 * np.ones(D),
                options={
                    "display": "off",
                    "max_fun_evals": mfe,
                    "random_seed": seed,
                },
            )
            r = b.optimize()
            print(name, seed, mfe, count(), r["func_count"])
